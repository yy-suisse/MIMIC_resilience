"""ModernBERT-style encoder for patient trajectories, with rotary position embeddings (RoPE).

Input of each position i (all summed, then normalised):
    x_i = E_token(id_i) + E_source(src_i) + A(age_days_i) + T(seconds_in_day_i) + 1[value visible] V(value_i)
  A: sine/cosine of the age at 16 periods from 1 day to 100 years, then a linear layer ("no time" vector if missing)
  T: sine/cosine of the time of day at 8 harmonics of 24 h, then a linear layer ("no time" vector if missing)
  V: small network on the scaled value, only at [VALUE] positions whose number is visible (D3)
No learned position table: positions enter through RoPE inside attention, restarting at 0 in each segment.

Encoder blocks (pre-norm, as ModernBERT): attention with RoPE, then a GeGLU feed-forward layer, no biases.
Attention mask: a token only attends to tokens of its own patient segment (seg), so packed patients never mix.

Heads (applied only where a target exists):
  concept   -> scores over the model vocabulary
  quantile  -> 10 scores (Q1..Q10)
  unit      -> scores over the units
  value     -> one number (D3)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate(x, cos, sin):
    """Rotary embedding on the last dimension: pairs (first half, second half) rotated by the position angle."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Block(nn.Module):
    def __init__(self, dim, heads, ff, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.norm1, self.norm2 = nn.LayerNorm(dim, bias=False), nn.LayerNorm(dim, bias=False)
        self.qkv, self.out = nn.Linear(dim, 3 * dim, bias=False), nn.Linear(dim, dim, bias=False)
        self.ff_in, self.ff_out = nn.Linear(dim, 2 * ff, bias=False), nn.Linear(ff, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cos, sin, mask):
        B, L, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.out(a.transpose(1, 2).reshape(B, L, D)))
        gate, up = self.ff_in(self.norm2(x)).chunk(2, dim=-1)
        return x + self.drop(self.ff_out(F.gelu(gate) * up))


class TrajectoryBert(nn.Module):
    def __init__(self, vocab_size, n_sources, n_units, dim=256, layers=4, heads=4, dropout=0.1,
                 use_value=False, rope_base=10000.0):
        super().__init__()
        ff = int(round(8 * dim / 3 / 64) * 64)              # GeGLU width, as in ModernBERT (704 for dim 256)
        self.tok = nn.Embedding(vocab_size, dim)
        self.src = nn.Embedding(n_sources + 1, dim)         # 0 = padding
        # absolute clock: age in days, 16 periods from 1 day to 100 years
        self.register_buffer("age_periods", torch.logspace(0, math.log10(36525), 16), persistent=False)
        self.age_proj, self.age_none = nn.Linear(32, dim), nn.Parameter(torch.zeros(dim))
        # time of day: harmonics 1..8 of 24 h
        self.register_buffer("tod_harmonics", torch.arange(1, 9, dtype=torch.float32), persistent=False)
        self.tod_proj, self.tod_none = nn.Linear(16, dim), nn.Parameter(torch.zeros(dim))
        self.use_value = use_value
        if use_value:
            self.value_mlp = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.emb_norm, self.emb_drop = nn.LayerNorm(dim, bias=False), nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(dim, heads, ff, dropout) for _ in range(layers))
        self.final_norm = nn.LayerNorm(dim, bias=False)
        # rotary angles per head dimension pair
        head_dim = dim // heads
        self.register_buffer("inv_freq", 1.0 / rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim),
                             persistent=False)
        # prediction heads: a shared transform, then one output layer per target
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.concept_out = nn.Linear(dim, vocab_size)
        self.quantile_out = nn.Linear(dim, 10)
        self.unit_out = nn.Linear(dim, max(n_units, 1))
        self.value_out = nn.Linear(dim, 1)

    def embed(self, b):
        x = self.tok(b["ids"]) + self.src(b["src"])
        age = b["age"].float()
        ang = 2 * math.pi * age[..., None] / self.age_periods
        a = self.age_proj(torch.cat([ang.sin(), ang.cos()], dim=-1))
        x = x + torch.where((b["age"] >= 0)[..., None], a, self.age_none)
        ang = 2 * math.pi * b["sec"].float()[..., None] / 86400 * self.tod_harmonics
        t = self.tod_proj(torch.cat([ang.sin(), ang.cos()], dim=-1))
        x = x + torch.where((b["sec"] >= 0)[..., None], t, self.tod_none)
        if self.use_value:
            v = self.value_mlp(b["value"][..., None])
            x = x + v * b["value_on"][..., None].to(v.dtype)
        return self.emb_drop(self.emb_norm(x))

    def forward(self, b):
        """Final hidden state of every position, shape (batch, row length, dim)."""
        x = self.embed(b)
        ang = b["pos"].float()[..., None] * self.inv_freq                 # (B, L, head_dim / 2)
        cos, sin = ang.cos()[:, None], ang.sin()[:, None]                  # (B, 1, L, head_dim / 2)
        mask = (b["seg"][:, :, None] == b["seg"][:, None, :])[:, None]     # (B, 1, L, L): own segment only
        for block in self.blocks:
            x = block(x, cos.to(x.dtype), sin.to(x.dtype), mask)
        return self.final_norm(x)
