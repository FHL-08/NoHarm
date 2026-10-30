from __future__ import annotations

import argparse
import bisect
import json
import subprocess
from pathlib import Path

import huggingface_hub
import imageio.v2 as imageio
import mujoco
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from harmless.embodiments.g1 import G1Embodiment
from harmless.policies import serve
from harmless.policies.gr00t import Gr00tVLAPolicy

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_OUT = Path.home() / ".cache" / "harmless" / "checkpoints" / "g1_gr00t"
_PROJECTORS = ("state_encoder.", "action_encoder.", "action_decoder.")
_HEAD_PREFIX = "action_head."
_CAMERA = "observation.images.head_cam"


class _Episodes:
    def __init__(self, root, space):
        self.root = Path(root)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.episodes = [json.loads(line) for line in
                         (self.root / "meta/episodes.jsonl").read_text().splitlines()
                         if line.strip()]
        self.tasks = {item["task_index"]: item["task"] for item in
                      (json.loads(line) for line in
                       (self.root / "meta/tasks.jsonl").read_text().splitlines()
                       if line.strip())}
        self.ends = np.cumsum([episode["length"] for episode in self.episodes])
        if not self.episodes or any(episode["length"] <= 0 for episode in self.episodes):
            raise ValueError("dataset must contain nonempty recorded episodes")
        self.order = {}
        for key in ("observation.state", "observation.velocity", "action"):
            names = self.info["features"][key]["names"]
            if len(names) != len(space.joints) or set(names) != set(space.joints):
                raise ValueError(f"{key} must name every G1 action joint once")
            self.order[key] = [names.index(joint) for joint in space.joints]
        self.policy = Gr00tVLAPolicy(action_space=space)
        self.mid = np.array([sum(space.bounds[joint]) / 2
                             for joint in space.joints], dtype=np.float32)
        self.half = np.array([(space.bounds[joint][1] - space.bounds[joint][0]) / 2
                              for joint in space.joints], dtype=np.float32)
        self.loaded = None
        self.reader = None

    def __len__(self):
        return int(self.ends[-1])

    def close(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None
        self.loaded = None

    def __getitem__(self, index):
        episode_slot = bisect.bisect_right(self.ends, index)
        episode = self.episodes[episode_slot]
        frame = index - (int(self.ends[episode_slot - 1]) if episode_slot else 0)
        if self.loaded != episode_slot:
            self.close()
            args = {"episode_index": episode["episode_index"],
                    "episode_chunk": episode["episode_index"] // self.info["chunks_size"],
                    "video_key": _CAMERA}
            table = pq.read_table(self.root / self.info["data_path"].format(**args))
            if len(table) != episode["length"]:
                raise ValueError("episode length does not match its parquet rows")
            self.vectors = {}
            for key, order in self.order.items():
                values = np.asarray(table[key].to_pylist(), dtype=np.float32)
                if values.shape != (len(table), len(order)) or not np.isfinite(values).all():
                    raise ValueError(f"invalid episode vectors: {key}")
                self.vectors[key] = values[:, order]
            self.task_indices = table["task_index"].to_pylist()
            self.reader = imageio.get_reader(
                self.root / self.info["video_path"].format(**args))
            self.loaded = episode_slot
        observation = {
            "joint_pos": self.vectors["observation.state"][frame],
            "joint_vel": self.vectors["observation.velocity"][frame],
            "image": self.reader.get_data(frame),
            "instruction": self.tasks[self.task_indices[frame]],
        }
        future = np.minimum(np.arange(frame, frame + serve.ACTION_HORIZON),
                            episode["length"] - 1)
        targets = np.clip((self.vectors["action"][future] - self.mid) / self.half,
                          -1.0, 1.0)
        action = torch.zeros(1, serve.ACTION_HORIZON, serve.ACTION_DIM)
        action[0, :, :targets.shape[1]] = torch.from_numpy(targets)
        return (self.policy._state_tensor(observation),
                self.policy._vl_tensor(observation, None), action)


def _load_head(snapshot, device):
    with torch.device("meta"):
        head = serve.FlowmatchingActionHead()
    expected = dict(head.named_parameters())
    loaded = set()
    for shard in sorted(snapshot.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as weights:
            for key in weights.keys():
                if not key.startswith(_HEAD_PREFIX):
                    continue
                name = key.removeprefix(_HEAD_PREFIX)
                if name not in expected or name in loaded:
                    raise ValueError(f"unexpected or duplicate action-head tensor: {key}")
                tensor = weights.get_slice(key)
                if tuple(tensor.get_shape()) != tuple(expected[name].shape):
                    raise ValueError(f"incompatible action-head tensor: {key}")
                trainable = name.startswith(_PROJECTORS)
                if trainable:
                    value = tensor[serve.NEW_EMBODIMENT_ID:serve.NEW_EMBODIMENT_ID + 1]
                    value = value.to(device=device, dtype=torch.float32).clone()
                else:
                    value = weights.get_tensor(key).to(device=device)
                module_name, parameter_name = name.rsplit(".", 1)
                head.get_submodule(module_name).register_parameter(
                    parameter_name, torch.nn.Parameter(value, requires_grad=trainable))
                loaded.add(name)
    if loaded != set(expected):
        raise ValueError(f"missing action-head tensors: {sorted(set(expected) - loaded)}")
    return head.eval()


def _loss(head, state, context, actions, joints):
    embodiment = torch.zeros(actions.shape[0], dtype=torch.long, device=actions.device)
    with torch.no_grad():
        context = head.process_backbone_output(context.to(dtype=head.vlln.weight.dtype))
    state_tokens = head.state_encoder(state, embodiment)
    noise = torch.randn_like(actions)
    sample = torch.distributions.Beta(1.5, 1.0).sample((actions.shape[0],))
    time = ((0.999 - sample) / 0.999).to(device=actions.device)
    mixed = (1 - time[:, None, None]) * noise + time[:, None, None] * actions
    timesteps = (time * serve.NUM_TIMESTEP_BUCKETS).long()
    action_tokens = head.action_encoder(mixed, timesteps, embodiment)
    positions = torch.arange(serve.ACTION_HORIZON, device=actions.device)
    action_tokens = action_tokens + head.position_embedding(positions).unsqueeze(0)
    tokens = torch.cat((state_tokens,
                        head.future_tokens.weight.unsqueeze(0).expand(actions.shape[0], -1, -1),
                        action_tokens), dim=1)
    hidden = head.model(tokens, context, timesteps)
    predicted = head.action_decoder(hidden, embodiment)[:, -serve.ACTION_HORIZON:]
    return F.mse_loss(predicted[..., :joints].float(),
                      (actions - noise)[..., :joints].float())


def _destination(out_dir):
    output = Path(out_dir).expanduser().resolve()
    for parent in (output, *output.parents):
        if parent.name == ".git":
            raise ValueError("checkpoint cannot be written inside .git")
        if (parent / ".git").exists():
            repository = subprocess.run(
                ["git", "-C", str(parent), "rev-parse", "--show-toplevel"],
                check=False, capture_output=True, text=True)
            if repository.returncode != 0:
                continue
            ignored = subprocess.run(
                ["git", "-C", str(parent), "check-ignore", "-q", "--", str(output)],
                check=False).returncode == 0
            tracked = subprocess.run(
                ["git", "-C", str(parent), "ls-files", "--", str(output)],
                check=True, capture_output=True, text=True).stdout
            if not ignored or tracked:
                raise ValueError("checkpoint must be outside git or in an ignored directory")
            break
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"checkpoint destination must be empty: {output}")
    return output


def _save_head(head, snapshot, output):
    parameters = dict(head.named_parameters())
    state = {}
    for shard in sorted(snapshot.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as weights:
            for key in weights.keys():
                if not key.startswith(_HEAD_PREFIX):
                    continue
                name = key.removeprefix(_HEAD_PREFIX)
                value = parameters[name].detach().cpu()
                if name.startswith(_PROJECTORS):
                    released = weights.get_tensor(key).clone()
                    released[serve.NEW_EMBODIMENT_ID] = value[0].to(released.dtype)
                    value = released
                state[key] = value.contiguous()
    output.mkdir(parents=True, exist_ok=True)
    pending = output / "model.safetensors.tmp"
    try:
        save_file(state, str(pending))
        pending.replace(output / "model.safetensors")
    finally:
        pending.unlink(missing_ok=True)


def finetune(dataset, out_dir=_DEFAULT_OUT, *, steps=1000, batch_size=1,
             repo_id=serve.DEFAULT_REPO_ID, revision=None, device=None,
             learning_rate=1e-4, seed=0):
    """Fit G1 projectors with frozen VL/DiT weights; return the checkpoint directory."""
    if steps < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("steps, batch_size, and learning_rate must be positive")
    output = _destination(out_dir)
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type not in ("cpu", "cuda"):
        raise ValueError("training supports CPU or one CUDA device")
    model = mujoco.MjModel.from_xml_path(str(_REPO / "kitchen.xml"))
    space = G1Embodiment(model=model).action_space
    del model
    if len(space.joints) > serve.ACTION_DIM or 2 * len(space.joints) > serve.MAX_STATE_DIM:
        raise ValueError("G1 state/action dimensions exceed the released head")
    episodes = _Episodes(dataset, space)
    try:
        snapshot = Path(huggingface_hub.snapshot_download(
            repo_id=repo_id, revision=revision, local_files_only=True))
    except huggingface_hub.errors.LocalEntryNotFoundError:
        snapshot = Path(huggingface_hub.snapshot_download(repo_id=repo_id, revision=revision))
    if output.is_relative_to(snapshot.resolve()):
        raise ValueError("checkpoint destination cannot modify the released snapshot")
    dtype = (torch.float16 if device.type == "cuda" and not torch.cuda.is_bf16_supported()
             else torch.bfloat16)
    head = _load_head(snapshot, device)
    parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
    generator = np.random.default_rng(seed)
    losses = []
    try:
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            torch.manual_seed(seed)
            for update in range(steps):
                optimizer.zero_grad(set_to_none=True)
                total = 0.0
                for _ in range(batch_size):
                    state, context, actions = episodes[int(generator.integers(len(episodes)))]
                    with torch.autocast(device.type, dtype=dtype):
                        loss = _loss(head, state.to(device), context.to(device),
                                     actions.to(device), len(space.joints))
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite flow-matching loss")
                    scaler.scale(loss / batch_size).backward()
                    total += loss.detach().item() / batch_size
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                losses.append(total)
                print(f"step {update + 1}/{steps} loss={total:.6f}", flush=True)
        _save_head(head, snapshot, output)
        metadata = {
            "base_repo_id": repo_id, "base_revision": snapshot.name,
            "tuning": "g1_category_projectors", "frozen_context": "adapter_fixed_projections",
            "embodiment_id": serve.NEW_EMBODIMENT_ID, "joints": list(space.joints),
            "joint_bounds": space.bounds, "dataset": str(Path(dataset).resolve()),
            "steps": steps, "batch_size": batch_size, "learning_rate": learning_rate,
            "seed": seed, "loss": losses,
        }
        (output / "training.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return output
    finally:
        episodes.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Post-train released GR00T N G1 projectors; freeze VL layers and denoiser.")
    parser.add_argument("--dataset", required=True, help="POL-5 LeRobot episode directory")
    parser.add_argument("--out", dest="out_dir", default=str(_DEFAULT_OUT),
                        help="empty checkpoint directory outside git (or gitignored)")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="gradient accumulation count; microbatch size is always one")
    parser.add_argument("--repo-id", default=serve.DEFAULT_REPO_ID)
    parser.add_argument("--revision")
    parser.add_argument("--device", help="cpu or a single CUDA device, e.g. cuda:0")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    print(finetune(**vars(parser.parse_args(argv))))


if __name__ == "__main__":
    main()
