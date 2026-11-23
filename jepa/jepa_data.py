"""Data for JEPA training: arrays with episodes and landmarks, then (context, target) samples, packed into rows.

Two parts:
  1. prepare_arrays   master table -> flat arrays per split, as for BERT, plus the episode of each token and
                      the landmarks (last token of each episode). One folder per (model, cut-off).
  2. JepaBatches      for each patient: a landmark T, a context (gender + history + recent window) and
                      n_pairs targets (the events of (T + delta, T + delta + L]), packed into rows.

The arrays are the BERT ones (see bert/bert_data.py) plus:
  episode       int32   episode number of each token within its patient (-1 = gender). An admission or an ED
                        visit is one episode; outpatient events of the same day form one episode.
  landmark_idx  int64   index of the last token of each episode, patient after patient
  landmark_ptr  int64   where the landmarks of each patient start in landmark_idx

Time is handled in days (age_days): the target holds the events whose day is in (T + delta, T + delta + L].
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bert"))
from bert_data import (FIELDS as BERT_FIELDS, SPLITS, load_yaml, make_remap, model_token_ids,  # noqa: E402
                       to_model_rows, value_scaling_tables)

FIELDS = {**BERT_FIELDS, "episode": np.int32}
SEGMENT = {"static": 0, "history": 1, "recent": 2}     # value of the segment channel
CODE = 1                                               # ttype of a concept token


# ---- 1. master table -> arrays ----------------------------------------------------------------------
def add_episode(rows):
    """Episode number per token: admission or ED visit (episode_id), else one episode per outpatient day."""
    key = pl.coalesce(pl.col("episode_id"), pl.concat_str(pl.lit("O"), pl.col("age").cast(pl.String)))
    number = (key.rank("dense").over("subject_id") - 1).cast(pl.Int32)
    return rows.with_columns(episode=pl.when(pl.col("ttype") == 0).then(-1).otherwise(number).cast(pl.Int32))


def prepare_arrays(traj_cfg, model, min_patients, arrays_dir):
    """Master table -> JEPA arrays for one model and one cut-off. Bucket by bucket, so memory stays low."""
    out = Path(traj_cfg["paths"]["out"])
    master_dir = out / traj_cfg["paths"]["master_dir"]
    files = sorted(master_dir.glob(f"master_n{traj_cfg['n_buckets']}_b*.parquet"))
    if not files:
        raise FileNotFoundError(f"no master files in {master_dir}: run bert/build_bert_master.ipynb first")

    vocab = pl.read_parquet(out / traj_cfg["paths"]["vocab"])
    remap = make_remap(vocab, min_patients)                # same ids as the BERT encoder of this cut-off
    ids = model_token_ids(vocab, remap)
    sources = sorted(vocab["source"].drop_nulls().unique().to_list())
    unknown_full = vocab.filter((pl.col("source") == "quantile") & (pl.col("label") == "unknown"))["token_id"].item()
    per_unit, per_concept = value_scaling_tables(vocab, pl.read_parquet(out / "vocabulary" / "value_stats.parquet"))

    tmp = arrays_dir / "_chunks"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    chunks = {s: [] for s in SPLITS}

    for i, f in enumerate(files):
        rows = to_model_rows(pl.read_parquet(f), traj_cfg["models"][model]["types"], remap, sources,
                             per_unit, per_concept, unknown_full, 5.0, keep=("episode_id",))
        rows = add_episode(rows)
        for s in SPLITS:
            sub = rows.filter(pl.col("split") == s)
            if sub.height == 0:
                continue
            d = tmp / s / f"c{i:04d}"
            d.mkdir(parents=True)
            for name, dtype in FIELDS.items():
                np.save(d / f"{name}.npy", sub[name].to_numpy().astype(dtype))
            lengths = sub.group_by("subject_id", maintain_order=True).len()["len"].to_numpy().astype(np.int64)
            # landmark = last token of each episode (rows are in time order within a patient)
            landmarks = (sub.with_row_index("i")
                         .filter(pl.col("episode") >= 0)
                         .group_by(["subject_id", "episode"]).agg(pl.col("i").max())
                         .sort("i")["i"].to_numpy().astype(np.int64))
            ptr = np.concatenate([[0], np.cumsum(lengths)])
            n_landmarks = np.searchsorted(landmarks, ptr[1:]) - np.searchsorted(landmarks, ptr[:-1])
            np.save(d / "lengths.npy", lengths)
            np.save(d / "landmarks.npy", landmarks)
            np.save(d / "n_landmarks.npy", n_landmarks.astype(np.int64))
            chunks[s].append((d, sub.height))
        print(f"jepa arrays: bucket file {i + 1}/{len(files)} done", flush=True)

    n_tokens, n_patients = {}, {}
    for s in SPLITS:
        d_out = arrays_dir / s
        d_out.mkdir(parents=True, exist_ok=True)
        total = sum(n for _, n in chunks[s])
        for name, dtype in FIELDS.items():
            mm = np.lib.format.open_memmap(d_out / f"{name}.npy", mode="w+", dtype=dtype, shape=(total,))
            at = 0
            for d, n in chunks[s]:
                mm[at:at + n] = np.load(d / f"{name}.npy")
                at += n
            mm.flush()
            del mm
        lengths, landmarks, n_landmarks, offset = [], [], [], 0
        for d, n in chunks[s]:
            lengths.append(np.load(d / "lengths.npy"))
            landmarks.append(np.load(d / "landmarks.npy") + offset)      # chunk index -> index in the split
            n_landmarks.append(np.load(d / "n_landmarks.npy"))
            offset += n
        lengths = np.concatenate(lengths) if lengths else np.zeros(0, np.int64)
        np.save(d_out / "patient_ptr.npy", np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64))
        np.save(d_out / "landmark_idx.npy", np.concatenate(landmarks) if landmarks else np.zeros(0, np.int64))
        np.save(d_out / "landmark_ptr.npy",
                np.concatenate([[0], np.cumsum(np.concatenate(n_landmarks) if n_landmarks else [])]).astype(np.int64))
        n_tokens[s], n_patients[s] = int(total), int(len(lengths))
    shutil.rmtree(tmp, ignore_errors=True)

    meta = {"model": model, "min_patients": min_patients, "sources": sources,
            "n_tokens": n_tokens, "n_patients": n_patients, **ids}
    with open(arrays_dir / "meta.json", "w") as f:
        json.dump(meta, f)
    return meta


# ---- 2. samples ---------------------------------------------------------------------------------
class JepaArrays:
    """The arrays of one split, memory-mapped, plus lookups by source for history and target."""

    def __init__(self, folder, meta, jcfg):
        folder = Path(folder)
        self.f = {name: np.load(folder / f"{name}.npy", mmap_mode="r") for name in FIELDS}
        self.patient_ptr = np.load(folder / "patient_ptr.npy")
        self.landmark_idx = np.load(folder / "landmark_idx.npy")
        self.landmark_ptr = np.load(folder / "landmark_ptr.npy")
        self.n_patients = len(self.patient_ptr) - 1
        self.unk_id = meta["unk_id"]
        # source id -> True if it is a history source / a source allowed in the target (id 0 = padding)
        names = ["<pad>"] + meta["sources"]
        self.is_history = np.array([n in jcfg["history_sources"] for n in names])
        self.in_target = np.array([n not in jcfg["target_exclude"] and n != "<pad>" for n in names])


def log_uniform(rng, lo, hi):
    """A number of days between lo and hi, uniform on a log scale (log(1 + x))."""
    return float(np.expm1(rng.uniform(np.log1p(lo), np.log1p(hi))))


def keep_whole_pieces(idx, piece, cap):
    """Keep at most cap token indices from the start of idx without cutting an event (piece) in half."""
    if len(idx) <= cap:
        return idx
    p = piece[idx]
    end = cap
    while end > 0 and p[end] == p[end - 1]:          # move back to the start of the event that would be cut
        end -= 1
    return idx[:end]


def build_sample(a, p, jcfg, caps, rng, use_history=True):
    """One training sample for patient p: a context and up to n_pairs targets. None if impossible."""
    f = a.f
    s, e = a.patient_ptr[p], a.patient_ptr[p + 1]
    age, piece, source, ttype = f["age"], f["piece"], f["source"], f["ttype"]
    last_day = int(age[e - 1])

    # landmark T: the end of an episode that has at least one later day with data
    landmarks = a.landmark_idx[a.landmark_ptr[p]:a.landmark_ptr[p + 1]]
    landmarks = landmarks[age[landmarks] < last_day]
    if len(landmarks) == 0:
        return None
    t_idx = int(rng.choice(landmarks))
    t_day = int(age[t_idx])

    # recent window: tokens of the last recent_days up to T (rows are in time order)
    days = np.asarray(age[s + 1:t_idx + 1])
    lo = s + 1 + int(np.searchsorted(days, t_day - jcfg["recent_days"], side="left"))

    # history: concepts of the history sources before the recent window, each once at its last mention
    hist = np.zeros(0, np.int64)
    if lo > s + 1:
        idx = np.arange(s + 1, lo)
        # concepts of the history sources; [UNK] (any rare concept) is left out: it says nothing as a history item
        idx = idx[(ttype[idx] == CODE) & a.is_history[source[idx]] & (np.asarray(f["tokens"][idx]) != a.unk_id)]
        if len(idx):
            tokens = np.asarray(f["tokens"][idx])
            _, first_from_end = np.unique(tokens[::-1], return_index=True)
            last_mention = np.sort(len(idx) - 1 - first_from_end)          # positions in idx, in time order
            hist = idx[last_mention][-jcfg["history_max"]:]

    # the recent window keeps its most recent tokens, without cutting an event
    room = caps["context_cap"] - 1 - len(hist)
    if t_idx + 1 - lo > room:
        lo = t_idx + 1 - room
        while lo <= t_idx and piece[lo] == piece[lo - 1]:
            lo += 1
    recent = np.arange(lo, t_idx + 1)
    if not use_history:          # history switched off (check): same recent window, no history block
        hist = hist[:0]

    context = {
        "idx": np.concatenate([[s], hist, recent]).astype(np.int64),
        "segment": np.concatenate([[SEGMENT["static"]], np.full(len(hist), SEGMENT["history"]),
                                   np.full(len(recent), SEGMENT["recent"])]),
        "timeless": np.concatenate([[True], np.ones(len(hist), bool), np.zeros(len(recent), bool)]),
    }

    # targets: events of (T + delta, T + delta + L], bedside readings excluded
    after = np.arange(t_idx + 1, e)
    after = after[a.in_target[source[after]]]
    after_days = np.asarray(age[after])
    targets = []
    for _ in range(jcfg["n_pairs"]):
        for _ in range(jcfg["max_pair_tries"]):
            max_delta = min(jcfg["delta_days"]["max"], last_day - t_day - 1)
            delta = int(log_uniform(rng, jcfg["delta_days"]["min"], max_delta))
            length = max(1, int(log_uniform(rng, jcfg["length_days"]["min"], jcfg["length_days"]["max"])))
            start, end = t_day + delta, t_day + delta + length
            idx = after[(after_days > start) & (after_days <= end)]
            if len(idx):
                idx = keep_whole_pieces(idx, piece, caps["target_cap"])
                observed = min(length, last_day - start)          # the window may run past the patient's data
                targets.append({"idx": idx, "delta": delta, "length": observed})
                break
    if not targets:
        return None
    return {"context": context, "targets": targets}


# ---- 3. packing -------------------------------------------------------------------------------------
def pack(items, arrays, row_len, value_id):
    """Pack token lists into rows of row_len. Each item gets its own segment number (seg) in its row.

    items: dicts with "idx" (token indices) and optionally "segment" and "timeless" (no time channels).
    Returns the tensors and, for each item, its key = row * (row_len + 1) + seg (unique in the batch).
    """
    f = arrays.f
    rows, row, used = [], [], 0
    for item in items:
        if used + len(item["idx"]) > row_len:
            rows.append(row)
            row, used = [], 0
        row.append(item)
        used += len(item["idx"])
    if row:
        rows.append(row)

    B, L = len(rows), row_len
    t = {"ids": np.zeros((B, L), np.int64), "src": np.zeros((B, L), np.int64),
         "age": np.full((B, L), -1, np.int64), "sec": np.full((B, L), -1, np.int64),
         "value": np.zeros((B, L), np.float32), "value_on": np.zeros((B, L), bool),
         "seg": np.zeros((B, L), np.int64), "pos": np.zeros((B, L), np.int64),
         "segment": np.zeros((B, L), np.int64)}
    keys = []
    for b, row in enumerate(rows):
        at = 0
        for k, item in enumerate(row, start=1):
            idx = item["idx"]
            n = len(idx)
            sl = slice(at, at + n)
            value = np.asarray(f["value"][idx])
            timeless = item.get("timeless", np.zeros(n, bool))
            t["ids"][b, sl] = f["tokens"][idx]
            t["src"][b, sl] = f["source"][idx]
            t["age"][b, sl] = np.where(timeless, -1, f["age"][idx])
            t["sec"][b, sl] = np.where(timeless, -1, f["sec"][idx])
            t["value"][b, sl] = np.nan_to_num(value)
            t["value_on"][b, sl] = (np.asarray(f["tokens"][idx]) == value_id) & ~np.isnan(value)
            t["seg"][b, sl], t["pos"][b, sl] = k, np.arange(n)
            t["segment"][b, sl] = item.get("segment", np.full(n, SEGMENT["recent"]))
            keys.append(b * (L + 1) + k)
            at += n
    return {k: torch.from_numpy(v) for k, v in t.items()}, torch.tensor(keys)


class JepaBatches(torch.utils.data.IterableDataset):
    """Endless batches: batch_size patients shuffled per epoch, each giving one context and its targets."""

    def __init__(self, folder, meta, jcfg, model, batch_size, seed, shuffle=True, epochs=None, use_history=True):
        super().__init__()
        self.folder, self.meta, self.jcfg = str(folder), meta, jcfg
        self.caps, self.row_len = jcfg["models"][model], jcfg["row_len"]
        self.batch_size, self.seed, self.shuffle, self.epochs = batch_size, seed, shuffle, epochs
        self.use_history = use_history

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        arrays = JepaArrays(self.folder, self.meta, self.jcfg)
        rng = np.random.default_rng([self.seed, wid])
        epoch = 0
        samples = []                     # kept across epochs: the end of an epoch fills the first batch of the next one
        while self.epochs is None or epoch < self.epochs:
            order = rng.permutation(arrays.n_patients) if self.shuffle else np.arange(arrays.n_patients)
            for p in order[wid::nw]:
                sample = build_sample(arrays, int(p), self.jcfg, self.caps, rng, self.use_history)
                if sample is not None:
                    samples.append(sample)
                if len(samples) == self.batch_size:
                    yield self.collate(samples, arrays)
                    samples = []
            epoch += 1
        # only a finite run (validation) ends; a batch of 1 context has no spread, so it is dropped
        if len(samples) > 1:
            yield self.collate(samples, arrays)

    def collate(self, samples, arrays):
        value_id = self.meta["value_id"]
        ctx, ctx_key = pack([s["context"] for s in samples], arrays, self.row_len, value_id)
        pairs = [(i, t) for i, s in enumerate(samples) for t in s["targets"]]
        tgt, tgt_key = pack([t for _, t in pairs], arrays, self.row_len, value_id)
        return {
            "ctx": ctx, "ctx_key": ctx_key,
            "tgt": tgt, "tgt_key": tgt_key,
            "pair_ctx": torch.tensor([i for i, _ in pairs]),                       # context of each target
            "delta": torch.tensor([t["delta"] for _, t in pairs], dtype=torch.float32),
            "length": torch.tensor([t["length"] for _, t in pairs], dtype=torch.float32),
            "n_history": sum(int((s["context"]["segment"] == SEGMENT["history"]).sum()) for s in samples),
        }


def fixed_batches(folder, meta, jcfg, model, batch_size, n_batches, use_history=True, seed=12345):
    """The same validation batches at every evaluation (fixed seed), kept in memory.

    With use_history=False the batches hold the same landmarks and targets, only without the history block:
    no random draw depends on the history, so both versions are drawn identically.
    """
    data = JepaBatches(folder, meta, jcfg, model, batch_size, seed, shuffle=True, epochs=1, use_history=use_history)
    out = []
    for batch in data:
        out.append(batch)
        if len(out) == n_batches:
            break
    return out
