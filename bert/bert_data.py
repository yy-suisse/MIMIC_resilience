"""Data for BERT pre-training (D1, D2, D3).

Three parts:
  1. make_remap       full vocabulary id -> model id, for a vocabulary cut-off (min_patients)
  2. prepare_arrays   master table -> flat arrays per split (train / val / test), one folder per (model, cut-off)
  3. PackedBatches    patient segments -> masked segments -> rows packed with several patients -> torch tensors

The arrays are memory-mapped .npy files: one entry per token, patients one after the other.
  tokens  int32    model id of the token
  source  int16    source id (0 = padding)
  age     int32    age in days, -1 when missing (gender)
  sec     int32    seconds in the day, -1 when missing (gender, date-only events)
  value   float32  scaled value on [VALUE] tokens, NaN elsewhere
  ttype   int8     0 gender, 1 code, 2 quantile, 3 value, 4 unit
  piece   int32    event number within the patient (0 = gender)
  kind    int8     0 gender, 1 event without a number, 2 numeric event with an "unknown" bin, 3 numeric event
plus patient_ptr (start of each patient), day_starts + day_ptr (first token of each day with data).
"""
import inspect
import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml

KEY = ["source", "ontology_version", "label", "code"]
TYPE_IDS = {"gender": 0, "code": 1, "quantile": 2, "value": 3, "unit": 4}
FIELDS = {"tokens": np.int32, "source": np.int16, "age": np.int32, "sec": np.int32,
          "value": np.float32, "ttype": np.int8, "piece": np.int32, "kind": np.int8}
SPLITS = ["train", "val", "test"]

# the argument that lets null keys match in a join was renamed in recent Polars versions
NULLS = {"nulls_equal": True} if "nulls_equal" in inspect.signature(pl.DataFrame.join).parameters \
    else {"join_nulls": True}


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


# ---- 1. vocabulary cut-off --------------------------------------------------------------------
def make_remap(vocab, min_patients):
    """Kept tokens renumbered without gaps; the others -> [UNK]. Special tokens keep their ids."""
    always = pl.col("source").is_in(["special", "quantile", "patients", "unit"])  # patients = gender, death
    remap = (vocab
             .select(pl.col("token_id").cast(pl.Int64),
                     (always | (pl.col("n_patients") >= min_patients)).fill_null(False).alias("kept"))
             .with_columns(model_id=pl.col("kept").cast(pl.Int64).cum_sum() - 1))
    unk_full = vocab.filter((pl.col("source") == "special") & (pl.col("label") == "[UNK]"))["token_id"].item()
    unk_model = remap.filter(pl.col("token_id") == unk_full)["model_id"].item()
    return remap.with_columns(model_id=pl.when(pl.col("kept")).then(pl.col("model_id")).otherwise(unk_model))


def model_token_ids(vocab, remap):
    """Model ids of the special tokens, the quantile bins, the units and the concepts."""
    v = vocab.with_columns(pl.col("token_id").cast(pl.Int64)).join(remap, on="token_id")
    kept = v.filter(pl.col("kept"))
    special = dict(kept.filter(pl.col("source") == "special").select("label", "model_id").iter_rows())
    q = dict(kept.filter(pl.col("source") == "quantile").select("label", "model_id").iter_rows())
    is_gender = (pl.col("source") == "patients") & pl.col("label").is_in(["female", "male"])
    concepts = kept.filter(~pl.col("source").is_in(["special", "quantile", "unit"]) & ~is_gender)
    return {
        "vocab_size": int(kept.height),
        "pad_id": special["[PAD]"], "unk_id": special["[UNK]"], "mask_id": special["[MASK]"],
        "value_id": special["[VALUE]"],
        "q_ids": [q[f"Q{k}"] for k in range(1, 11)],        # Q1..Q10 in order: index = bin - 1
        "q_unknown_id": q["unknown"],
        "unit_ids": kept.filter(pl.col("source") == "unit")["model_id"].sort().to_list(),
        "concept_pool": concepts["model_id"].sort().to_list(),  # random replacement of a concept
    }


# ---- 2. master table -> arrays ----------------------------------------------------------------
def value_scaling_tables(vocab, value_stats):
    """Median and IQR per (concept, unit) and per concept, keyed by the full token id of the concept."""
    as_text = [pl.col(c).cast(pl.String) for c in KEY]          # an all-empty column would otherwise have type Null
    stats = (value_stats.with_columns(*as_text, pl.col("unit").cast(pl.String))
             .join(vocab.select(*as_text, pl.col("token_id").cast(pl.Int64)), on=KEY, how="inner", **NULLS))
    per_unit = (stats.filter(pl.col("level") == "unit")
                .select(pl.col("token_id").alias("concept_full"), "unit",
                        pl.col("median").alias("med_u"), pl.col("iqr").alias("iqr_u")))
    per_concept = (stats.filter(pl.col("level") == "token")
                   .select(pl.col("token_id").alias("concept_full"),
                           pl.col("median").alias("med_t"), pl.col("iqr").alias("iqr_t")))
    return per_unit, per_concept


def to_model_rows(master, model_types, remap, sources, per_unit, per_concept, unknown_full, clip):
    """One master bucket -> the rows and columns of the arrays, for one model."""
    by_piece = ["subject_id", "piece"]
    is_q = pl.col("type") == "quantile"
    df = master.with_columns(
        # full id of the event's concept, used to scale its value
        concept_full=pl.col("token_id").filter(pl.col("type") == "code").first().over(by_piece),
        # kind of event: 0 gender, 1 no number, 2 numeric with an "unknown" bin, 3 numeric
        kind=pl.when(pl.col("type") == "gender").then(0)
               .when((is_q & (pl.col("token_id") == unknown_full)).any().over(by_piece)).then(2)
               .when(is_q.any().over(by_piece)).then(3)
               .otherwise(1).cast(pl.Int8),
    ).filter(pl.col("type").is_in(model_types))

    # scaled value: per (concept, unit), else per concept; IQR 0 -> 1
    df = (df.join(per_unit, on=["concept_full", "unit"], how="left", **NULLS)
            .join(per_concept, on="concept_full", how="left")
            .sort(["subject_id", "pos"]))                                  # joins do not keep the order
    median = pl.coalesce("med_u", "med_t", pl.lit(0.0))
    iqr = (pl.when(pl.col("iqr_u") > 0).then(pl.col("iqr_u"))
             .when(pl.col("iqr_t") > 0).then(pl.col("iqr_t"))
             .otherwise(1.0))
    scaled = ((pl.col("value") - median) / iqr).clip(-clip, clip)

    # first token of each day with data: a new event (piece) on a new age_days
    new_piece = pl.col("piece").ne_missing(pl.col("piece").shift(1).over("subject_id"))
    new_day = pl.col("age_days").ne_missing(pl.col("age_days").shift(1).over("subject_id"))
    return df.select(
        "subject_id", "split",
        pl.col("token_id").replace_strict(remap["token_id"], remap["model_id"]).cast(pl.Int32).alias("tokens"),
        pl.col("source").replace_strict(sources, list(range(1, len(sources) + 1)), default=0)
          .cast(pl.Int16).alias("source"),
        pl.col("age_days").fill_null(-1).cast(pl.Int32).alias("age"),
        pl.col("seconds_in_day").fill_null(-1).cast(pl.Int32).alias("sec"),
        pl.when(pl.col("type") == "value").then(scaled).cast(pl.Float32).fill_null(float("nan")).alias("value"),
        pl.col("type").replace_strict(TYPE_IDS).cast(pl.Int8).alias("ttype"),
        pl.col("piece").cast(pl.Int32),
        "kind",
        ((pl.col("type") != "gender") & new_piece & new_day).alias("day_start"),
    )


def prepare_arrays(traj_cfg, model, min_patients, arrays_dir):
    """Master table -> arrays for one model and one cut-off. Bucket by bucket, so memory stays low."""
    out = Path(traj_cfg["paths"]["out"])
    master_dir = out / traj_cfg["paths"]["master_dir"]
    files = sorted(master_dir.glob(f"master_n{traj_cfg['n_buckets']}_b*.parquet"))
    if not files:
        raise FileNotFoundError(f"no master files in {master_dir}: run bert/build_bert_master.ipynb first")
    model_types = traj_cfg["models"][model]["types"]

    vocab = pl.read_parquet(out / traj_cfg["paths"]["vocab"])
    remap = make_remap(vocab, min_patients)
    ids = model_token_ids(vocab, remap)
    sources = sorted(vocab["source"].drop_nulls().unique().to_list())
    unknown_full = vocab.filter((pl.col("source") == "quantile") & (pl.col("label") == "unknown"))["token_id"].item()
    per_unit, per_concept = value_scaling_tables(vocab, pl.read_parquet(out / "vocabulary" / "value_stats.parquet"))
    clip = 5.0

    tmp = arrays_dir / "_chunks"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    chunks = {s: [] for s in SPLITS}   # per split: list of (chunk folder, n_tokens)

    for i, f in enumerate(files):
        rows = to_model_rows(pl.read_parquet(f), model_types, remap, sources, per_unit, per_concept,
                             unknown_full, clip)
        for s in SPLITS:
            sub = rows.filter(pl.col("split") == s)
            if sub.height == 0:
                continue
            d = tmp / s / f"c{i:04d}"
            d.mkdir(parents=True)
            for name, dtype in FIELDS.items():
                np.save(d / f"{name}.npy", sub[name].to_numpy().astype(dtype))
            per_patient = sub.group_by("subject_id", maintain_order=True).agg(
                pl.len().alias("n"), pl.col("day_start").sum().alias("n_days"))
            np.save(d / "subjects.npy", per_patient["subject_id"].to_numpy().astype(np.int64))
            np.save(d / "lengths.npy", per_patient["n"].to_numpy().astype(np.int64))
            np.save(d / "n_days.npy", per_patient["n_days"].to_numpy().astype(np.int64))
            np.save(d / "day_local.npy", np.flatnonzero(sub["day_start"].to_numpy()).astype(np.int64))
            chunks[s].append((d, sub.height))
        print(f"arrays: bucket file {i + 1}/{len(files)} done", flush=True)

    # join the chunks of each split into one memory-mapped file per field
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
        lengths, subjects, n_days, day_starts, offset = [], [], [], [], 0
        for d, n in chunks[s]:
            lengths.append(np.load(d / "lengths.npy"))
            subjects.append(np.load(d / "subjects.npy"))
            n_days.append(np.load(d / "n_days.npy"))
            day_starts.append(np.load(d / "day_local.npy") + offset)
            offset += n
        lengths = np.concatenate(lengths) if lengths else np.zeros(0, np.int64)
        np.save(d_out / "patient_ptr.npy", np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64))
        np.save(d_out / "subjects.npy", np.concatenate(subjects) if subjects else np.zeros(0, np.int64))
        np.save(d_out / "day_ptr.npy", np.concatenate([[0], np.cumsum(np.concatenate(n_days) if n_days else [])])
                .astype(np.int64))
        np.save(d_out / "day_starts.npy", np.concatenate(day_starts) if day_starts else np.zeros(0, np.int64))
        n_tokens[s], n_patients[s] = int(total), int(len(lengths))
    shutil.rmtree(tmp, ignore_errors=True)

    meta = {"model": model, "min_patients": min_patients, "types": model_types, "sources": sources,
            "value_clip": clip, "n_tokens": n_tokens, "n_patients": n_patients, **ids}
    with open(arrays_dir / "meta.json", "w") as f:
        json.dump(meta, f)
    remap.write_parquet(arrays_dir / "vocab_remap.parquet")
    return meta


# ---- 3. segments, masking, packing --------------------------------------------------------------
class SplitArrays:
    """The arrays of one split, memory-mapped (read from disk only when a token is needed)."""

    def __init__(self, folder):
        folder = Path(folder)
        self.f = {name: np.load(folder / f"{name}.npy", mmap_mode="r") for name in FIELDS}
        self.patient_ptr = np.load(folder / "patient_ptr.npy")
        self.day_ptr = np.load(folder / "day_ptr.npy")
        self.day_starts = np.load(folder / "day_starts.npy")
        self.n_patients = len(self.patient_ptr) - 1

    def segment(self, p, cap, rng):
        """Token indices of one segment of patient p: gender + events from a random day, at most cap tokens.

        A patient that fits is shown whole. Otherwise the segment starts at the first event of a day drawn
        at random (every day counts the same) and never cuts an event in half.
        """
        s, e = self.patient_ptr[p], self.patient_ptr[p + 1]
        if e - s <= cap:
            return np.arange(s, e)
        days = self.day_starts[self.day_ptr[p]:self.day_ptr[p + 1]]
        d = days[rng.integers(len(days))]
        end = min(e, d + cap - 1)
        piece = self.f["piece"]
        while d < end < e and piece[end] == piece[end - 1]:
            end -= 1
        return np.concatenate([[s], np.arange(d, end)])


def soft_label_matrix(eps):
    """Row q = target distribution for true bin q: 1 - 2 eps on q, eps on each neighbour (renormalised at the edges)."""
    m = np.zeros((10, 10), np.float32)
    for q in range(10):
        m[q, q] = 1 - 2 * eps
        for n in (q - 1, q + 1):
            if 0 <= n < 10:
                m[q, n] = eps
        m[q] /= m[q].sum()
    return torch.from_numpy(m)


class Masker:
    """Masks one segment: chooses events, chooses parts (plan B), applies 80/10/10, writes the targets."""

    def __init__(self, mcfg, meta):
        self.rate, self.span_mean, self.span_max = mcfg["mask_rate"], mcfg["span"]["mean"], mcfg["span"]["max"]
        self.part_prob = mcfg["part_prob"]
        self.p_mask, self.p_rand = mcfg["replace"]["mask"], mcfg["replace"]["random"]
        self.mask_id = meta["mask_id"]
        self.concept_pool = np.array(meta["concept_pool"], np.int32)
        self.q_ids = np.array(meta["q_ids"], np.int32)
        self.unit_ids = np.array(meta["unit_ids"], np.int32)
        # lookups: model id -> bin index (0..9) and model id -> unit index, -1 elsewhere
        self.qbin = np.full(meta["vocab_size"], -1, np.int64)
        self.qbin[self.q_ids] = np.arange(10)
        self.unit_index = np.full(meta["vocab_size"], -1, np.int64)
        self.unit_index[self.unit_ids] = np.arange(len(self.unit_ids))

    def __call__(self, seg, rng):
        orig = seg["tokens"]
        tokens, value = orig.copy(), np.nan_to_num(seg["value"]).astype(np.float32)
        ttype, piece, kind = seg["ttype"], seg["piece"], seg["kind"]
        L = len(tokens)
        out = {"tgt_concept": np.full(L, -100, np.int64), "tgt_cpart": np.zeros(L, np.int64),
               "tgt_q": np.full(L, -100, np.int64), "tgt_unit": np.full(L, -100, np.int64),
               "tgt_value": np.full(L, np.nan, np.float32)}
        value_on = (ttype == 3) & ~np.isnan(seg["value"])
        pool = value[value_on]                                   # scaled values for the random replacement

        starts = np.flatnonzero(np.r_[True, piece[1:] != piece[:-1]])   # first token of each event
        ends = np.r_[starts[1:], L]
        is_event = kind[starts] != 0                                    # the gender is never masked
        ev_starts, ev_ends = starts[is_event], ends[is_event]
        n = len(ev_starts)
        n_masked = n_numeric = 0
        if n > 0:
            chosen = np.zeros(n, bool)
            n_sel = max(1, int(round(self.rate * n)))
            while chosen.sum() < n_sel:                                 # spans of consecutive events
                length = min(int(rng.geometric(1 / self.span_mean)), self.span_max)
                a = int(rng.integers(n))
                chosen[a:a + length] = True
            for j in np.flatnonzero(chosen):
                self._mask_event(int(ev_starts[j]), int(ev_ends[j]), orig, tokens, value, value_on, pool,
                                 ttype, kind, out, rng)
            n_masked = int(chosen.sum())
            n_numeric = int((kind[ev_starts[chosen]] >= 2).sum())
        out.update(tokens=tokens, value=value, value_on=value_on)
        return out, n_masked, n_numeric

    def _mask_event(self, a, b, orig, tokens, value, value_on, pool, ttype, kind, out, rng):
        tt = ttype[a:b]
        cpos = a + int(np.flatnonzero(tt == 1)[0])
        qpos, vpos, upos = (a + np.flatnonzero(tt == t) for t in (2, 3, 4))
        k = int(kind[a])
        if k == 3 and len(qpos):          # numeric event: each part masked with part_prob, at least one
            avail = ["concept", "magnitude"] + (["unit"] if len(upos) else [])
            while True:
                picked = rng.random(len(avail)) < self.part_prob
                if picked.any():
                    break
            parts = [p for p, x in zip(avail, picked) if x]
        else:                             # no number, or an "unknown" bin: the concept (and its unit)
            parts = ["concept"] + (["unit"] if len(upos) else [])

        for part in parts:
            r = rng.random()              # < p_mask: [MASK]; < p_mask + p_rand: random same type; else kept
            if part == "concept":
                out["tgt_concept"][cpos] = orig[cpos]
                out["tgt_cpart"][cpos] = 2 if k >= 2 else 1          # 2 = concept_measure, 1 = concept_event
                if r < self.p_mask:
                    tokens[cpos] = self.mask_id
                elif r < self.p_mask + self.p_rand:
                    tokens[cpos] = rng.choice(self.concept_pool)
            elif part == "unit":
                u = int(upos[0])
                out["tgt_unit"][u] = self.unit_index[orig[u]]
                if r < self.p_mask:
                    tokens[u] = self.mask_id
                elif r < self.p_mask + self.p_rand:
                    tokens[u] = rng.choice(self.unit_ids)
            else:                         # magnitude: the quantile, and in D3 the [VALUE] with its number
                q = int(qpos[0])
                out["tgt_q"][q] = self.qbin[orig[q]]
                v = int(vpos[0]) if len(vpos) else None
                if v is not None:
                    out["tgt_value"][v] = value[v]
                if r < self.p_mask:
                    tokens[q] = self.mask_id
                    if v is not None:
                        tokens[v], value_on[v] = self.mask_id, False
                elif r < self.p_mask + self.p_rand:
                    tokens[q] = rng.choice(self.q_ids)
                    if v is not None:
                        value[v] = rng.choice(pool) if len(pool) else np.float32(rng.normal())


class PackedBatches(torch.utils.data.IterableDataset):
    """Endless batches of packed rows for training: one segment per patient per epoch, patients shuffled.

    Several patient segments share a row of row_len positions. Each segment gets its own number (seg),
    and positions restart at 0 in each segment; the model only lets a token attend to its own segment.
    """

    def __init__(self, folder, mcfg, meta, model, batch_size, seed, shuffle=True, epochs=None):
        super().__init__()
        self.folder, self.mcfg, self.meta = str(folder), mcfg, meta
        self.cap = mcfg["models"][model]["max_segment"]
        self.row_len, self.batch_size = mcfg["row_len"], batch_size
        self.seed, self.shuffle, self.epochs = seed, shuffle, epochs

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        arrays, masker = SplitArrays(self.folder), Masker(self.mcfg, self.meta)
        rng = np.random.default_rng([self.seed, wid])
        epoch = 0
        while self.epochs is None or epoch < self.epochs:
            order = rng.permutation(arrays.n_patients) if self.shuffle else np.arange(arrays.n_patients)
            rows, row, used = [], [], 0
            for p in order[wid::nw]:                   # each worker takes its share of the patients
                idx = arrays.segment(int(p), self.cap, rng)
                if used + len(idx) > self.row_len:
                    rows.append(row)
                    row, used = [], 0
                    if len(rows) == self.batch_size:
                        yield self.collate(rows, arrays, masker, rng)
                        rows = []
                row.append(idx)
                used += len(idx)
            if row:
                rows.append(row)
            if rows:
                yield self.collate(rows, arrays, masker, rng)
            epoch += 1

    def collate(self, rows, arrays, masker, rng):
        B, L = len(rows), self.row_len
        t = {"ids": np.zeros((B, L), np.int64), "src": np.zeros((B, L), np.int64),
             "age": np.full((B, L), -1, np.int64), "sec": np.full((B, L), -1, np.int64),
             "value": np.zeros((B, L), np.float32), "value_on": np.zeros((B, L), bool),
             "seg": np.zeros((B, L), np.int64), "pos": np.zeros((B, L), np.int64),
             "tgt_concept": np.full((B, L), -100, np.int64), "tgt_cpart": np.zeros((B, L), np.int64),
             "tgt_q": np.full((B, L), -100, np.int64), "tgt_unit": np.full((B, L), -100, np.int64),
             "tgt_value": np.full((B, L), np.nan, np.float32)}
        n_masked = n_numeric = 0
        for b, row in enumerate(rows):
            at = 0
            for k, idx in enumerate(row, start=1):
                seg = {name: np.asarray(arrays.f[name][idx]) for name in FIELDS}
                m, nm, nn = masker(seg, rng)
                n_masked, n_numeric = n_masked + nm, n_numeric + nn
                sl = slice(at, at + len(idx))
                t["ids"][b, sl], t["src"][b, sl] = m["tokens"], seg["source"]
                t["age"][b, sl], t["sec"][b, sl] = seg["age"], seg["sec"]
                t["value"][b, sl], t["value_on"][b, sl] = m["value"], m["value_on"]
                t["seg"][b, sl], t["pos"][b, sl] = k, np.arange(len(idx))
                for name in ("tgt_concept", "tgt_cpart", "tgt_q", "tgt_unit", "tgt_value"):
                    t[name][b, sl] = m[name]
                at += len(idx)
        batch = {k: torch.from_numpy(v) for k, v in t.items()}
        batch["n_masked"], batch["n_numeric"] = n_masked, n_numeric
        return batch


def fixed_batches(folder, mcfg, meta, model, batch_size, n_batches, seed=12345):
    """The same validation batches at every evaluation (one pass, fixed seed), kept in memory."""
    data = PackedBatches(folder, mcfg, meta, model, batch_size, seed, shuffle=True, epochs=1)
    out = []
    for batch in data:
        out.append(batch)
        if len(out) == n_batches:
            break
    return out
