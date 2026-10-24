"""Served Isaac GR00T N weights: checkpoint download and model runtime.

This module is the serving layer behind the gr00t policy adapter. It
fetches the released nvidia/GR00T-N1.5-3B checkpoint through
huggingface_hub.snapshot_download into the local HF cache (never through
git, never vendored into the repository), rebuilds the released
flow-matching action head — the category-specific state/action encoders,
the interleaved self/cross-attention DiT, and the action decoder — loads
the released safetensors weights into it strictly, and runs the released
Euler denoising loop that turns state and vision-language context
features into a 16-step action chunk.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn

DEFAULT_REPO_ID = "nvidia/GR00T-N1.5-3B"
NEW_EMBODIMENT_ID = 31

ACTION_DIM = 32
ACTION_HORIZON = 16
HIDDEN_SIZE = 1024
INPUT_EMBEDDING_DIM = 1536
BACKBONE_EMBEDDING_DIM = 2048
MAX_STATE_DIM = 64
MAX_SEQ_LEN = 1024
NUM_EMBODIMENTS = 32
NUM_VISION_TOKENS = 32
NUM_TIMESTEP_BUCKETS = 1000
NUM_INFERENCE_STEPS = 4
DIT_LAYERS = 16
DIT_HEADS = 32
DIT_HEAD_DIM = 48
DIT_OUT_DIM = 1024
DROPOUT = 0.2
VL_LAYERS = 4
VL_HEADS = 32
VL_HEAD_DIM = 64

checkpoint_dir = None


def ensure_checkpoint(repo_id=DEFAULT_REPO_ID, revision=None, cache_dir=None):
    global checkpoint_dir
    path = snapshot_download(
        repo_id=repo_id, revision=revision, cache_dir=cache_dir)
    checkpoint_dir = Path(path)
    return checkpoint_dir


def _swish(x):
    return x * torch.sigmoid(x)


class _CategoryLinear(nn.Module):
    def __init__(self, num_categories, in_dim, out_dim):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(num_categories, in_dim, out_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, out_dim))

    def forward(self, x, cat_ids):
        w = self.W[cat_ids]
        b = self.b[cat_ids]
        return torch.bmm(x, w) + b.unsqueeze(1)


class _CategoryMLP(nn.Module):
    def __init__(self, num_categories, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layer1 = _CategoryLinear(num_categories, in_dim, hidden_dim)
        self.layer2 = _CategoryLinear(num_categories, hidden_dim, out_dim)

    def forward(self, x, cat_ids):
        return self.layer2(F.relu(self.layer1(x, cat_ids)), cat_ids)


class _SinusoidalTau(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        t = timesteps.float()
        half = self.dim // 2
        exponent = -torch.arange(half, dtype=torch.float, device=t.device) * (
            math.log(10000.0) / half)
        freqs = t.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class _ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_categories):
        super().__init__()
        self.W1 = _CategoryLinear(num_categories, action_dim, hidden_size)
        self.W2 = _CategoryLinear(num_categories, 2 * hidden_size, hidden_size)
        self.W3 = _CategoryLinear(num_categories, hidden_size, hidden_size)
        self.pos_encoding = _SinusoidalTau(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        b, t_len, _ = actions.shape
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(1).expand(-1, t_len)
        a_emb = self.W1(actions, cat_ids)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = _swish(self.W2(x, cat_ids))
        return self.W3(x, cat_ids)


class _Attention(nn.Module):
    def __init__(self, dim, heads, head_dim, cross_dim=None, bias=True):
        super().__init__()
        inner = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.to_q = nn.Linear(dim, inner, bias=bias)
        self.to_k = nn.Linear(cross_dim or dim, inner, bias=bias)
        self.to_v = nn.Linear(cross_dim or dim, inner, bias=bias)
        self.to_out = nn.ModuleList(
            [nn.Linear(inner, dim), nn.Dropout(0.0)])

    def forward(self, hidden, context=None):
        ctx = hidden if context is None else context
        b = hidden.shape[0]
        q = self.to_q(hidden).view(b, -1, self.heads, self.head_dim)
        k = self.to_k(ctx).view(b, -1, self.heads, self.head_dim)
        v = self.to_v(ctx).view(b, -1, self.heads, self.head_dim)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        out = out.transpose(1, 2).reshape(b, -1, self.heads * self.head_dim)
        return self.to_out[1](self.to_out[0](out))


class _ApproxGeluProj(nn.Module):
    def __init__(self, dim, inner):
        super().__init__()
        self.proj = nn.Linear(dim, inner)

    def forward(self, x):
        return F.gelu(self.proj(x), approximate="tanh")


class _FeedForward(nn.Module):
    def __init__(self, dim, inner, dropout):
        super().__init__()
        self.net = nn.ModuleList([
            _ApproxGeluProj(dim, inner),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
        ])

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class _AdaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 2 * dim)
        self.norm = nn.LayerNorm(dim, 1e-5, elementwise_affine=False)

    def forward(self, x, temb):
        scale, shift = self.linear(F.silu(temb)).chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class _Block(nn.Module):
    def __init__(self, dim, heads, head_dim, dropout, cross_dim=None,
                 ada_norm=True, affine=True):
        super().__init__()
        self.self_attention = cross_dim is None
        self.norm1 = (
            _AdaLayerNorm(dim) if ada_norm else
            nn.LayerNorm(dim, 1e-5, elementwise_affine=affine))
        self.attn1 = _Attention(dim, heads, head_dim, cross_dim=cross_dim)
        self.norm3 = nn.LayerNorm(dim, 1e-5, elementwise_affine=affine)
        self.ff = _FeedForward(dim, 4 * dim, dropout)
        self.final_dropout = nn.Dropout(dropout)

    def forward(self, hidden, context=None, temb=None):
        if isinstance(self.norm1, _AdaLayerNorm):
            normed = self.norm1(hidden, temb)
        else:
            normed = self.norm1(hidden)
        attn = self.attn1(normed, None if self.self_attention else context)
        hidden = hidden + self.final_dropout(attn)
        return hidden + self.ff(self.norm3(hidden))


class _Timesteps(nn.Module):
    def __init__(self, dim=256, flip_sin_to_cos=True, freq_shift=1):
        super().__init__()
        self.dim = dim
        self.flip = flip_sin_to_cos
        self.shift = freq_shift

    def forward(self, timesteps):
        t = timesteps.float()
        half = self.dim // 2
        exponent = -math.log(10000.0) * torch.arange(
            half, dtype=torch.float, device=t.device) / (half - self.shift)
        emb = t[:, None] * exponent.exp()[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.flip:
            emb = torch.cat([emb[:, half:], emb[:, :half]], dim=-1)
        return emb


class _TimestepEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, out_dim)
        self.linear_2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return self.linear_2(F.silu(self.linear_1(x)))


class _TimestepEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.time_proj = _Timesteps(256, flip_sin_to_cos=True, freq_shift=1)
        self.timestep_embedder = _TimestepEmbedding(256, dim)

    def forward(self, timesteps):
        return self.timestep_embedder(self.time_proj(timesteps))


class _DiT(nn.Module):
    def __init__(self):
        super().__init__()
        inner = DIT_HEADS * DIT_HEAD_DIM
        self.timestep_encoder = _TimestepEncoder(inner)
        self.transformer_blocks = nn.ModuleList([
            _Block(
                inner, DIT_HEADS, DIT_HEAD_DIM, DROPOUT,
                cross_dim=(
                    None if i % 2 == 1 else BACKBONE_EMBEDDING_DIM),
                ada_norm=True, affine=False)
            for i in range(DIT_LAYERS)
        ])
        self.norm_out = nn.LayerNorm(inner, 1e-6, elementwise_affine=False)
        self.proj_out_1 = nn.Linear(inner, 2 * inner)
        self.proj_out_2 = nn.Linear(inner, DIT_OUT_DIM)

    def forward(self, hidden, context, timestep):
        temb = self.timestep_encoder(timestep)
        for block in self.transformer_blocks:
            hidden = block(hidden, context, temb)
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden = self.norm_out(hidden) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(hidden)


class _SelfAttentionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        inner = VL_HEADS * VL_HEAD_DIM
        self.transformer_blocks = nn.ModuleList([
            _Block(inner, VL_HEADS, VL_HEAD_DIM, DROPOUT,
                   cross_dim=None, ada_norm=False, affine=True)
            for _ in range(VL_LAYERS)
        ])

    def forward(self, hidden):
        for block in self.transformer_blocks:
            hidden = block(hidden)
        return hidden


class FlowmatchingActionHead(nn.Module):
    """The released GR00T N1.5 flow-matching action head.

    Mirrors NVIDIA's n1.5-release FlowmatchingActionHead: state and noisy
    action tokens are encoded by category-specific MLPs, fused with the
    learned future tokens, denoised by a 16-layer DiT that alternates
    cross-attention to vision-language features with self-attention, and
    decoded back to action_dim through the category-specific decoder.
    `get_action` is the released Euler-integration inference loop.
    """

    def __init__(self):
        super().__init__()
        self.state_encoder = _CategoryMLP(
            NUM_EMBODIMENTS, MAX_STATE_DIM, HIDDEN_SIZE, INPUT_EMBEDDING_DIM)
        self.action_encoder = _ActionEncoder(
            ACTION_DIM, INPUT_EMBEDDING_DIM, NUM_EMBODIMENTS)
        self.action_decoder = _CategoryMLP(
            NUM_EMBODIMENTS, DIT_OUT_DIM, HIDDEN_SIZE, ACTION_DIM)
        self.future_tokens = nn.Embedding(
            NUM_VISION_TOKENS, INPUT_EMBEDDING_DIM)
        self.position_embedding = nn.Embedding(
            MAX_SEQ_LEN, INPUT_EMBEDDING_DIM)
        self.vlln = nn.LayerNorm(BACKBONE_EMBEDDING_DIM)
        self.vl_self_attention = _SelfAttentionTransformer()
        self.model = _DiT()

    def process_backbone_output(self, backbone_features):
        features = self.vlln(backbone_features)
        return self.vl_self_attention(features)

    @torch.no_grad()
    def get_action(self, backbone_features, state, embodiment_id):
        self.eval()
        vl_embs = self.process_backbone_output(backbone_features)
        state_features = self.state_encoder(state, embodiment_id)
        batch = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            batch, ACTION_HORIZON, ACTION_DIM,
            dtype=vl_embs.dtype, device=device)
        dt = 1.0 / NUM_INFERENCE_STEPS
        for step in range(NUM_INFERENCE_STEPS):
            t_discretized = int(
                step / NUM_INFERENCE_STEPS * NUM_TIMESTEP_BUCKETS)
            timesteps = torch.full(
                (batch,), t_discretized, device=device, dtype=torch.long)
            action_features = self.action_encoder(
                actions, timesteps, embodiment_id)
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features \
                + self.position_embedding(pos_ids).unsqueeze(0)
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(
                batch, -1, -1)
            sa_embs = torch.cat(
                [state_features, future_tokens, action_features], dim=1)
            model_output = self.model(sa_embs, vl_embs, timesteps)
            pred = self.action_decoder(model_output, embodiment_id)
            actions = actions + dt * pred[:, -ACTION_HORIZON:]
        return actions


def _head_state_dict(checkpoint):
    state = {}
    for shard in sorted(Path(checkpoint).glob("*.safetensors")):
        for name, tensor in load_file(str(shard)).items():
            if name.startswith("action_head."):
                state[name[len("action_head."):]] = tensor
    return state


def load_action_head(repo_id=DEFAULT_REPO_ID, revision=None,
                     checkpoint=None, cache_dir=None):
    root = Path(checkpoint) if checkpoint else ensure_checkpoint(
        repo_id=repo_id, revision=revision, cache_dir=cache_dir)
    state = _head_state_dict(root)
    if not state:
        raise RuntimeError(
            f"no action_head tensors found in checkpoint {root}")
    head = FlowmatchingActionHead()
    head.load_state_dict(state, strict=True)
    return head.float().eval()
