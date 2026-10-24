"""Isaac GR00T N policy adapter: a served VLA on the joint_pos contract.

Gr00tVLAPolicy fills the same act(observation) slot every policy fills:
it binds to an embodiment's declared joint_pos action_space, fetches the
released nvidia/GR00T-N1.5-3B weights into the local Hugging Face cache
through harmless.policies.serve, loads the released flow-matching action
head, and runs its denoising loop over each observation — proprioceptive
state through the released state encoder, camera frames and the language
instruction through a fixed projection standing in for the VLM backbone
features the action head cross-attends to — then maps the emitted
16x32 action chunk onto the G1's declared per-joint bounds.
"""
from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from harmless.policies import serve
from harmless.policy import Policy

INSTRUCTION_KEYS = (
    "instruction", "subgoal", "task", "prompt", "goal",
    "language_instruction", "command",
)
IMAGE_PATCH = 16
IMAGE_GRID = 4
SEED_VIS = 20260309
SEED_TXT = 20260310


class Gr00tVLAPolicy(Policy):
    """Policy adapter serving released Isaac GR00T N weights."""

    name = "gr00t"
    description = (
        "served Isaac GR00T N1.5 flow-matching action head emitting "
        "joint_pos chunks"
    )

    def __init__(self, embodiment=None, action_space=None, model_id=None,
                 repo_id=None, checkpoint=None, instruction=None, **_unused):
        if isinstance(embodiment, (str, os.PathLike)) and embodiment is not None:
            model_id = model_id or str(embodiment)
            embodiment = None
        super().__init__(embodiment=embodiment, action_space=action_space)
        self.repo_id = (
            repo_id or model_id or self._as_repo_id(checkpoint)
            or serve.DEFAULT_REPO_ID)
        self.checkpoint_path = (
            str(checkpoint)
            if checkpoint is not None and Path(str(checkpoint)).exists()
            else None)
        self.instruction = instruction
        self.model = None
        self._w_vis = None
        self._w_txt = None

    @staticmethod
    def _as_repo_id(value):
        if isinstance(value, str) and "/" in value \
                and not Path(value).exists():
            return value
        return None

    @property
    def action_joints(self):
        space = self.action_space
        return list(space.joints) if space is not None else []

    def load(self):
        return self.ensure_weights()

    def ensure_weights(self):
        if self.model is None:
            self.model = serve.load_action_head(
                repo_id=self.repo_id, checkpoint=self.checkpoint_path)
            self.checkpoint_path = str(
                serve.checkpoint_dir or self.checkpoint_path or "")
        return self.model

    def close(self):
        self.model = None

    def _joint_values(self, observation, key):
        joints = self.action_joints
        vector = observation.get(key)
        if isinstance(vector, np.ndarray) and vector.size >= len(joints) \
                and joints:
            return np.asarray(vector, dtype=float)[: len(joints)]
        measured = observation.get("joints")
        out = []
        for j in joints:
            try:
                out.append(float(measured[j]))
            except Exception:
                out.append(0.0)
        return np.asarray(out, dtype=float)

    def _state_tensor(self, observation):
        joints = self.action_joints
        space = self.action_space
        pos = self._joint_values(observation, "joint_pos")
        vel = self._joint_values(observation, "joint_vel")
        state = np.zeros(serve.MAX_STATE_DIM, dtype=np.float32)
        n = len(joints)
        if space is not None and n:
            lo = np.array([space.bounds[j][0] for j in joints])
            hi = np.array([space.bounds[j][1] for j in joints])
            half = np.maximum((hi - lo) / 2.0, 1e-6)
            state[:n] = np.clip((pos - (hi + lo) / 2.0) / half, -1.0, 1.0)
            state[n:2 * n] = np.tanh(vel / 5.0)
        else:
            state[:n] = pos
            state[n:2 * n] = np.tanh(vel)
        return torch.from_numpy(state).view(1, 1, -1)

    def _instruction_text(self, observation, instruction):
        if isinstance(instruction, str) and instruction:
            return instruction
        if isinstance(observation, Mapping):
            for key in INSTRUCTION_KEYS:
                text = observation.get(key)
                if isinstance(text, str) and text:
                    return text
        return self.instruction or ""

    def _projector(self, in_dim, seed):
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(in_dim, serve.BACKBONE_EMBEDDING_DIM,
                           generator=gen) / math.sqrt(in_dim)

    def _image_tokens(self, image):
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., :3].mean(axis=-1)
        if arr.size == 0:
            return None
        if arr.max() > 2.0:
            arr = arr / 255.0
        side = IMAGE_GRID * IMAGE_PATCH
        img = torch.from_numpy(arr).view(1, 1, arr.shape[0], arr.shape[1])
        img = torch.nn.functional.interpolate(
            img, size=(side, side), mode="area").view(side, side)
        patches = img.view(IMAGE_GRID, IMAGE_PATCH,
                           IMAGE_GRID, IMAGE_PATCH)
        patches = patches.permute(0, 2, 1, 3).reshape(
            IMAGE_GRID * IMAGE_GRID, IMAGE_PATCH * IMAGE_PATCH)
        if self._w_vis is None:
            self._w_vis = self._projector(
                IMAGE_PATCH * IMAGE_PATCH, SEED_VIS)
        return patches @ self._w_vis

    def _text_token(self, text):
        counts = np.zeros(256, dtype=np.float32)
        for byte in text.encode("utf-8", errors="ignore"):
            counts[byte] += 1.0
        total = counts.sum()
        if total > 0:
            counts /= total
        if self._w_txt is None:
            self._w_txt = self._projector(256, SEED_TXT)
        return torch.from_numpy(counts) @ self._w_txt

    def _vl_tensor(self, observation, instruction):
        tokens = []
        if isinstance(observation, Mapping):
            cameras = observation.get("cameras")
            image = None
            if isinstance(cameras, Mapping) and cameras:
                image = cameras.get("head_cam")
                if image is None:
                    image = next(iter(cameras.values()))
            elif isinstance(observation.get("image"), np.ndarray):
                image = observation["image"]
            if image is not None:
                img_tokens = self._image_tokens(image)
                if img_tokens is not None:
                    tokens.append(img_tokens)
            text = self._instruction_text(observation, instruction)
            if text:
                tokens.append(self._text_token(text).view(1, -1))
        if not tokens:
            tokens.append(torch.zeros(1, serve.BACKBONE_EMBEDDING_DIM))
        return torch.cat(tokens, dim=0).unsqueeze(0)

    def _contract_actions(self, actions):
        joints = self.action_joints
        space = self.action_space
        n = min(len(joints), actions.shape[-1])
        chunk = actions[:, :n].float()
        peak = chunk.abs().max().clamp(min=1e-6)
        normalized = chunk / peak
        if space is None:
            return normalized.numpy()
        lo = np.array([space.bounds[j][0] for j in joints])
        hi = np.array([space.bounds[j][1] for j in joints])
        mid = (hi + lo) / 2.0
        half = (hi - lo) / 2.0
        targets = torch.from_numpy(mid) + normalized * torch.from_numpy(half)
        return torch.clamp(
            targets, torch.from_numpy(lo), torch.from_numpy(hi)).numpy()

    def act(self, observation, instruction=None):
        head = self.ensure_weights()
        if not isinstance(observation, Mapping):
            observation = {}
        state = self._state_tensor(observation)
        vl = self._vl_tensor(observation, instruction)
        embodiment = torch.tensor(
            [serve.NEW_EMBODIMENT_ID], dtype=torch.long)
        actions = head.get_action(vl, state, embodiment)
        return self._contract_actions(actions[0])


try:
    from harmless import registry as _registry

    for _alias in ("gr00t", "vla", "isaac_gr00t", "gr00t_n1_5"):
        _registry.register_policy(_alias, Gr00tVLAPolicy)
except Exception:
    pass
