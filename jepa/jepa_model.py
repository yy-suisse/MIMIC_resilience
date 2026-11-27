"""JEPA model: student and teacher encoders, predictor Z, pooling, EMA update, anti-collapse terms.

  student  = BERT encoder (D1, D2 or D3, from its best.pt) + a segment embedding (static / history / recent)
             initialised at zero, so at step 0 the student is exactly the BERT encoder
  teacher  = copy of the student, updated only as an exponential moving average (EMA) of the student
  c        = mean of the student's outputs over the context tokens (gender and padding excluded)
  y        = teacher's outputs averaged per day, then over the days of the target, then layer-normalised
  y_hat    = Z(c, f(delta), f(L_obs))      f = sine and cosine of log(1 + days) at a few frequencies

Three variants of how the context reaches Z (--variant of train_jepa.py, recorded in the run's config):
  base           c = mean over all context tokens; Z = MLP on [c, f(delta), f(L_obs)]
  concat_hist    c = [mean of the history tokens ; mean over the recent days of the per-day means] (2 x 256),
                 plus a 0/1 "no history" flag; Z = the same MLP, wider input
  z_transformer  Z = 2 cross-attention layers: one query per (delta, L) pair, built from f(delta), f(L_obs),
                 reads every context token; y_hat = the query's output. c = mean of the tokens is kept
                 for the anti-collapse terms and the resilience map
  hist_z         the encoder reads [gender + recent window] and [history] as two separate sequences (same
                 weights); Z = 3 cross-attention layers whose query is built from f(delta), f(L_obs), f(age at T)
                 and a no-history flag, and reads the recent tokens, the history tokens and one learned empty
                 token. c = recent window pooled like the target (per day, then over days)
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
STATIC, HISTORY, RECENT = 0, 1, 2              # values of the segment channel (as SEGMENT in jepa_data.py)
VARIANTS = ("base", "concat_hist", "z_transformer", "hist_z")


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

    def __init__(self, dim, n_time, max_delta, max_length, in_dim=None):
        super().__init__()
        self.n_time, self.max_delta, self.max_length = n_time, max_delta, max_length
        in_dim = in_dim or dim                   # concat_hist: 2 * dim + 1 (both halves and the no-history flag)
        self.net = nn.Sequential(
            nn.Linear(in_dim + 4 * n_time, 2 * dim), nn.GELU(),
            nn.Linear(2 * dim, 2 * dim), nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, c, delta, length, extra=None):
        parts = [c] + ([extra] if extra is not None else [])
        z = torch.cat(parts + [time_features(delta, self.max_delta, self.n_time),
                               time_features(length, self.max_length, self.n_time)], dim=-1)
        return self.net(z)


class QueryLayer(nn.Module):
    """One layer of the transformer Z: the queries read the context tokens (cross-attention), then a feed-forward.
    Only the queries change; the context tokens are keys and values."""

    def __init__(self, dim, heads, ff_mult):
        super().__init__()
        self.norm_q, self.norm_kv, self.norm_ff = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, ff_mult * dim), nn.GELU(), nn.Linear(ff_mult * dim, dim))

    def forward(self, q, H, pad):
        kv = self.norm_kv(H)
        q = q + self.attn(self.norm_q(q), kv, kv, key_padding_mask=pad, need_weights=False)[0]
        return q + self.ff(self.norm_ff(q))


class TransformerPredictor(nn.Module):
    """Z as a small transformer (variant z_transformer).

    q    = q0 + MLP([f(delta), f(L_obs)])            one query per (delta, L) pair
    q    <- q + Attention(q, context tokens);  q <- q + FFN(q)     (layers times)
    y_hat = Linear(LayerNorm(q))
    With equal attention weights 1/n and no FFN this is the mean pooling of the base variant.
    """

    def __init__(self, dim, n_time, max_delta, max_length, layers=2, heads=4, ff_mult=2):
        super().__init__()
        self.n_time, self.max_delta, self.max_length = n_time, max_delta, max_length
        self.q0 = nn.Parameter(torch.randn(dim) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(4 * n_time, dim), nn.GELU(), nn.Linear(dim, dim))
        self.layers = nn.ModuleList(QueryLayer(dim, heads, ff_mult) for _ in range(layers))
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, H, pad, pair_ctx, delta, length):
        """H, pad: context tokens (contexts x tokens x dim) and padding mask (True = padding);
        pair_ctx: the context of each pair. Returns y_hat, one row per pair."""
        t = torch.cat([time_features(delta, self.max_delta, self.n_time),
                       time_features(length, self.max_length, self.n_time)], dim=-1)
        q = (self.q0 + self.time_mlp(t))[:, None, :]          # (pairs, 1, dim)
        H, pad = H[pair_ctx], pad[pair_ctx]                                 # each query reads its own context
        for layer in self.layers:
            q = layer(q, H, pad)
        return self.out(q[:, 0])


class HistZPredictor(nn.Module):
    """Z of the variant hist_z: a small transformer whose query reads the recent tokens, the history tokens and
    one learned empty token (where attention can go when nothing fits).

    q     = q0 + MLP([f(delta), f(L_obs), f(age at T), no-history flag])      one query per (delta, L) pair
    keys  = [recent tokens + e_recent ; history tokens + e_history ; e_empty]
    q    <- q + Attention(q, keys);  q <- q + FFN(q)      (layers times)
    y_hat = Linear(LayerNorm(q))
    """

    def __init__(self, dim, n_time, max_delta, max_length, max_age, layers=3, heads=4, ff_mult=2):
        super().__init__()
        self.n_time, self.max_delta, self.max_length, self.max_age = n_time, max_delta, max_length, max_age
        self.q0 = nn.Parameter(torch.randn(dim) * 0.02)
        self.query_mlp = nn.Sequential(nn.Linear(6 * n_time + 1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.kind = nn.Parameter(torch.randn(2, dim) * 0.02)       # 0 = recent token, 1 = history token
        self.empty = nn.Parameter(torch.randn(dim) * 0.02)
        self.layers = nn.ModuleList(QueryLayer(dim, heads, ff_mult) for _ in range(layers))
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, inputs, pair_ctx, delta, length, return_query=False):
        """inputs: R, pad_R (recent tokens), H, pad_H (history tokens), age, no_history, one row per context.
        Returns y_hat, one row per pair (and the final query before the output layer if return_query)."""
        R, pad_R, H, pad_H = inputs["R"], inputs["pad_R"], inputs["H"], inputs["pad_H"]
        n = len(R)
        empty = self.empty.to(R.dtype).expand(n, 1, -1)
        keys = torch.cat([R + self.kind[0].to(R.dtype), H + self.kind[1].to(R.dtype), empty], dim=1)
        pad = torch.cat([pad_R, pad_H, torch.zeros(n, 1, dtype=torch.bool, device=R.device)], dim=1)
        t = torch.cat([time_features(delta, self.max_delta, self.n_time),
                       time_features(length, self.max_length, self.n_time),
                       time_features(inputs["age"][pair_ctx], self.max_age, self.n_time),
                       inputs["no_history"][pair_ctx, None]], dim=-1)
        q = (self.q0 + self.query_mlp(t))[:, None, :]
        keys, pad = keys[pair_ctx], pad[pair_ctx]                      # each query reads its own context
        for layer in self.layers:
            q = layer(q, keys, pad)
        y_hat = self.out(q[:, 0])
        return (y_hat, q[:, 0]) if return_query else y_hat


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


def pool_context_concat(h, b, wanted):
    """Variant concat_hist: [mean of the history tokens ; mean over the recent days of the per-day means].

    Each recent day counts the same (as in pool_target). A context without history gets zeros for that half
    and flag 1. Returns c (contexts x 2 dim) and flag (contexts), in the order of wanted.
    """
    key, ok, d = token_keys(b), b["seg"] > 0, h.shape[-1]
    hist = torch.zeros(len(wanted), d, device=h.device)
    flag = torch.ones(len(wanted), device=h.device)
    m = ok & (b["segment"] == HISTORY)
    if m.any():
        uniq, means = group_mean(h[m].float(), key[m])
        at = torch.searchsorted(uniq, wanted).clamp(max=len(uniq) - 1)
        has = uniq[at] == wanted
        hist[has] = means[at[has]]
        flag = (~has).float()
    m = ok & (b["segment"] == RECENT)
    day_keys, day_means = group_mean(h[m].float(), key[m] * 100_000 + b["age"][m])   # one row per (context, day)
    uniq, means = group_mean(day_means, day_keys // 100_000)
    recent = means[torch.searchsorted(uniq, wanted)]
    return torch.cat([hist, recent], dim=-1), flag


def pool_days(h, b, wanted):
    """Mean per day, then over the days, of the recent-window tokens of each sample: the pooling of the target,
    without its layer norm. Variant hist_z: c of the recent window. Only recent tokens: gender and history have
    no day (age -1), which would shift them to the previous key in the division below."""
    m = (b["seg"] > 0) & (b["segment"] == RECENT)
    day_keys, day_means = group_mean(h[m].float(), token_keys(b)[m] * 100_000 + b["age"][m])
    uniq, means = group_mean(day_means, day_keys // 100_000)
    return means[torch.searchsorted(uniq, wanted)]


def history_tokens(h, b, n_contexts):
    """Variant hist_z: the history tokens of each context, padded (contexts without history: all padding)."""
    has = b["hist_key"] >= 0
    if not has.any():
        return h.new_zeros(n_contexts, 0, h.shape[-1]), torch.ones(n_contexts, 0, dtype=torch.bool, device=h.device)
    Hh, pad_h = context_tokens(h, b, b["hist_key"][has])
    H = Hh.new_zeros(n_contexts, Hh.shape[1], Hh.shape[2])
    pad = torch.ones(n_contexts, Hh.shape[1], dtype=torch.bool, device=h.device)
    H[has], pad[has] = Hh, pad_h
    return H, pad


def context_tokens(h, b, wanted):
    """Variant z_transformer: the context tokens of each sample as a padded matrix (gender included).
    Returns H (contexts x longest context x dim) and pad (True = padding), in the order of wanted."""
    m = b["seg"] > 0
    keys, hv = token_keys(b)[m], h[m]
    order = torch.argsort(keys, stable=True)                 # tokens of one sample together, in their order
    keys, hv = keys[order], hv[order]
    uniq, counts = torch.unique_consecutive(keys, return_counts=True)
    start = torch.cumsum(counts, 0) - counts
    group = torch.repeat_interleave(torch.arange(len(uniq), device=h.device), counts)
    rank = torch.arange(len(keys), device=h.device) - start[group]
    H = hv.new_zeros(len(uniq), int(counts.max()), hv.shape[-1])
    pad = torch.ones(len(uniq), int(counts.max()), dtype=torch.bool, device=h.device)
    H[group, rank] = hv
    pad[group, rank] = False
    at = torch.searchsorted(uniq, wanted)
    return H[at], pad[at]


def make_predictor(dim, jcfg, variant):
    """Z of a variant (base, concat_hist, z_transformer)."""
    sizes = (jcfg["time_features"], jcfg["delta_days"]["max"], jcfg["length_days"]["max"])
    if variant == "concat_hist":
        return Predictor(dim, *sizes, in_dim=2 * dim + 1)
    if variant == "z_transformer":
        z = jcfg.get("z_transformer", {})
        return TransformerPredictor(dim, *sizes, layers=z.get("layers", 2), heads=z.get("heads", 4),
                                    ff_mult=z.get("ff_mult", 2))
    if variant == "hist_z":
        z = jcfg.get("hist_z", {})
        return HistZPredictor(dim, *sizes, max_age=z.get("max_age_days", 40000), layers=z.get("layers", 3),
                              heads=z.get("heads", 4), ff_mult=z.get("ff_mult", 2))
    return Predictor(dim, *sizes)


def context_inputs(variant, h, b, wanted):
    """c (for the anti-collapse terms and the map) and what Z reads besides it, for one batch of contexts."""
    if variant == "concat_hist":
        c, flag = pool_context_concat(h, b, wanted)
        return c, {"extra": flag[:, None]}
    if variant == "hist_z":
        R, pad_R = context_tokens(h, b, wanted)              # gender + recent window
        H, pad_H = history_tokens(h, b, len(wanted))
        return pool_days(h, b, wanted), {"R": R, "pad_R": pad_R, "H": H, "pad_H": pad_H, "age": b["age_T"],
                                         "no_history": (b["hist_key"] < 0).float()}
    c = pool_context(h, b, wanted)
    if variant == "z_transformer":
        H, pad = context_tokens(h, b, wanted)
        return c, {"H": H, "pad": pad}
    return c, {}


def run_predictor(variant, predictor, c, inputs, pair_ctx, delta, length, return_query=False):
    """y_hat of every (context, delta, L) pair; pair_ctx = the context of each pair.
    hist_z with return_query: also the final query of Z (the patient vector that combines recent, history, age)."""
    if variant == "hist_z":
        return predictor(inputs, pair_ctx, delta, length, return_query=return_query)
    if variant == "z_transformer":
        return predictor(inputs["H"], inputs["pad"], pair_ctx, delta, length)
    extra = inputs["extra"][pair_ctx] if "extra" in inputs else None
    return predictor(c[pair_ctx], delta, length, extra=extra)


def map_vector(variant, c):
    """The vector of a point on the resilience map. concat_hist: each half scaled to unit length, so the history
    and the recent days weigh the same in the cosine (a context without history keeps a zero half)."""
    if variant != "concat_hist":
        return c
    d = c.shape[-1] // 2
    return torch.cat([F.normalize(c[:, :d], dim=-1), F.normalize(c[:, d:], dim=-1)], dim=-1)


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
