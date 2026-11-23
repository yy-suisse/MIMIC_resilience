"""JEPA model: student and teacher encoders, predictor Z, pooling, EMA update, anti-collapse terms.

  student  = BERT encoder (D1, D2 or D3, from its best.pt) + a segment embedding (static / history / recent)
             initialised at zero, so at step 0 the student is exactly the BERT encoder
  teacher  = copy of the student, updated only as an exponential moving average (EMA) of the student
  c        = mean of the student's outputs over the context tokens (gender and padding excluded)
  y        = teacher's outputs averaged per day, then over the days of the target, then layer-normalised
  y_hat    = Z(c, f(delta), f(L_obs))      f = sine and cosine of log(1 + days) at a few frequencies
"""
import copy
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bert"))
from bert_model import TrajectoryBert  # noqa: E402

BERT_HEADS = ("head", "concept_out", "quantile_out", "unit_out", "value_out")   # not used by JEPA


def load_bert(run_dir, checkpoint="best.pt", device="cpu"):
    """The trained BERT encoder of a run folder, and that run's config (model, cut-off, sizes)."""
    ck = torch.load(Path(run_dir) / checkpoint, map_location=device, weights_only=False)
    cfg = ck["config"]
    a = cfg["args"]
    bert = TrajectoryBert(cfg["vocab_size"], cfg["n_sources"], cfg["n_units"], a["dim"], a["layers"], a["heads"],
                          a["dropout"], use_value=(a["model"] == "D3"))
    bert.load_state_dict(ck["model"])
    for name in BERT_HEADS:                       # the masked-token heads are not trained any more
        getattr(bert, name).requires_grad_(False)
    return bert, cfg


class JepaEncoder(nn.Module):
    """BERT encoder + segment embedding (0 static, 1 history, 2 recent), added to the input."""

    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.segment = nn.Embedding(3, bert.tok.embedding_dim)
        nn.init.zeros_(self.segment.weight)      # step 0: identical to the BERT encoder

    def forward(self, b):
        return self.bert(b, extra=self.segment(b["segment"]))


def time_features(days, max_days, n):
    """Sine and cosine of log(1 + days), scaled to [0, 1] by max_days, at n frequencies: shape (N, 2n)."""
    u = torch.log1p(days.float().clamp_min(0)) / math.log1p(max_days)
    angle = math.pi * u[:, None] * torch.arange(1, n + 1, device=days.device)
    return torch.cat([angle.sin(), angle.cos()], dim=-1)


class Predictor(nn.Module):
    """Z: context vector + gap + observed length -> predicted vector of the future window."""

    def __init__(self, dim, n_time, max_delta, max_length):
        super().__init__()
        self.n_time, self.max_delta, self.max_length = n_time, max_delta, max_length
        self.net = nn.Sequential(
            nn.Linear(dim + 4 * n_time, 2 * dim), nn.GELU(),
            nn.Linear(2 * dim, 2 * dim), nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, c, delta, length):
        z = torch.cat([c, time_features(delta, self.max_delta, self.n_time),
                       time_features(length, self.max_length, self.n_time)], dim=-1)
        return self.net(z)


# ---- pooling -------------------------------------------------------------------------------------
def group_mean(values, keys):
    """Mean of the rows of values per key. Returns (sorted unique keys, means)."""
    uniq, inv = torch.unique(keys, return_inverse=True)
    sums = torch.zeros(len(uniq), values.shape[-1], device=values.device, dtype=values.dtype).index_add_(0, inv, values)
    counts = torch.zeros(len(uniq), device=values.device, dtype=values.dtype).index_add_(
        0, inv, torch.ones_like(keys, dtype=values.dtype))
    return uniq, sums / counts[:, None]


def token_keys(b):
    """Key of the sample each token belongs to: row * (row_len + 1) + seg (0 for padding)."""
    B, L = b["seg"].shape
    return b["seg"] + torch.arange(B, device=b["seg"].device)[:, None] * (L + 1)


def pool_context(h, b, wanted):
    """c: mean over the context tokens (segment 0 = gender and padding are excluded), in the order of wanted."""
    m = (b["seg"] > 0) & (b["segment"] > 0)
    uniq, means = group_mean(h[m].float(), token_keys(b)[m])
    return means[torch.searchsorted(uniq, wanted)]


def pool_target(h, b, wanted):
    """y: mean per day, then mean over the days of each target (every day counts the same), layer-normalised."""
    m = b["seg"] > 0
    key, day = token_keys(b)[m], b["age"][m]
    day_keys, day_means = group_mean(h[m].float(), key * 100_000 + day)     # one row per (target, day)
    uniq, means = group_mean(day_means, day_keys // 100_000)
    y = means[torch.searchsorted(uniq, wanted)]
    return F.layer_norm(y, y.shape[-1:])


# ---- teacher and anti-collapse -----------------------------------------------------------------------
def make_teacher(student):
    teacher = copy.deepcopy(student)
    teacher.requires_grad_(False)
    return teacher


@torch.no_grad()
def ema_update(teacher, student, momentum):
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.mul_(momentum).add_(s.detach(), alpha=1 - momentum)


def variance_covariance(c, eps=1e-4):
    """VICReg terms on the batch of context vectors: keep each dimension spread, and dimensions decorrelated."""
    if len(c) < 2:                       # no spread to measure with one vector (var would be NaN)
        zero = c.sum() * 0
        return zero, zero
    c = c - c.mean(dim=0)
    std = torch.sqrt(c.var(dim=0) + eps)
    var_term = F.relu(1 - std).mean()
    cov = (c.T @ c) / max(1, len(c) - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_term = (off_diag ** 2).sum() / c.shape[-1]
    return var_term, cov_term


@torch.no_grad()
def effective_rank(x):
    """exp(entropy of the normalised singular values): about how many directions the vectors really use."""
    s = torch.linalg.svdvals(x.float() - x.float().mean(dim=0))
    p = s / s.sum().clamp_min(1e-12)
    return float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))
