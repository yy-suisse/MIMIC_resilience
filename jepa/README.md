# MIMIC JEPA training: context, target and predictor Z

How the JEPA stage is built, what a training sample looks like, and the commands to run it.
Last updated 2026-11-23. Plan and decisions: `claude/jepa_plan.md` in the claude.ai project "Mimic Jepa".

## Overview

JEPA continues the training of a BERT encoder (D1, D2 or D3) with a new objective: from a patient's past up to a landmark T, **predict the embedding of a future window**, separated from T by a gap Δ that the predictor is told. No token is reconstructed; the prediction happens in the embedding space.

| Part | What it is |
| --- | --- |
| Student encoder | the BERT encoder + a segment embedding (static / history / recent); reads the context |
| Teacher encoder | an EMA copy of the student; reads the target; never trained by gradient |
| Context vector c | mean of the student's outputs over the context tokens |
| Target vector y | teacher's outputs averaged per day, then over days, layer-normalised |
| Predictor Z | ŷ = Z(c, Δ, L_obs): the expected embedding of the window (T + Δ, T + Δ + L] |

Two things are kept after training: the student encoder (c, a summary of the patient up to T) and student + Z (ŷ, the expected future at a chosen gap and length). Each gives its own latent space for the resilience score.

## Pipeline

JEPA reads the same data as BERT, so steps 1–8 of `bert/README.md` come first (MIMIC tables → thinned, binned events → master token table), plus a trained BERT run (step 10).

| Step | Code | What it does | Output |
| --- | --- | --- | --- |
| 1–8 | see `bert/README.md` | events, vocabulary, bins, thinning, master table | `D:\MIMIC\trajectory_out\bert\master\` |
| 10 | `bert\train_bert.py` | BERT pre-training D1, D2, D3 | `D:\MIMIC\bert_model\<run>\best.pt` |
| 11 | `jepa\jepa_data.py` (`prepare_arrays`), run automatically | master table → JEPA arrays with episodes and landmarks | `D:\MIMIC\trajectory_out\jepa\arrays\<model>_min<N>\{train,val,test}` |
| 12 | `jepa\train_jepa.py` | JEPA training from a BERT run | `D:\MIMIC\jepa_model\<run>\` |
| 13 | to do | embed points with c and ŷ; gate; resilience score | |

The model (D1, D2, D3) and the vocabulary cut-off (`min_patients`) are read from the BERT run, so the arrays and the encoder always match. The recent window and the target come from the thinned master table, so they are thinned exactly as in BERT pre-training.

## JEPA arrays (step 11)

The BERT arrays (`tokens`, `source`, `age`, `sec`, `value`, `ttype`, `piece`, `kind`, `patient_ptr`) plus:

| Array | Content |
| --- | --- |
| `episode` | episode number of each token within its patient (−1 = gender). An admission or an ED visit is one episode; the outpatient events of one day form one episode |
| `landmark_idx` | index of the last token of each episode: the candidate landmarks |
| `landmark_ptr` | where each patient's landmarks start in `landmark_idx` |

Built once per (model, cut-off), the first time `train_jepa.py` runs (or with `--prepare-only`).

## One training sample

Drawn on the fly, one per patient per epoch, so the same patient gives many splits over the epochs.

1. **Landmark T**: the end of a randomly drawn episode of any type (admission, ED visit, outpatient day) that has at least one later day with data.
2. **Context** (student encoder), at most `context_cap` tokens:
    - **Static**: the gender token.
    - **History**: diagnoses, drugs and procedures (`history_sources`) before the recent window. A bag of concepts: each concept once, at its **last mention**, ordered by last mention, so the most recent sit next to the recent window. No time channels ("no time" vector), no Q, `[VALUE]` or unit, no shuffling. `[UNK]` is left out (it stands for any rare concept). At most `history_max` items, the most recent kept.
    - **Recent**: every token of the last `recent_days` up to T, with full detail (concept, Q, `[VALUE]`, unit) and real age and time of day, as in BERT. The landmark visit's billing codes are included. If too long, the most recent tokens are kept, never cutting an event.
3. **Targets** (teacher encoder), `n_pairs` per context:
    - Δ drawn on a log scale in [0, 730] days (but leaving at least one later day of data); L drawn on a log scale in [1, 365] days.
    - Target = the events whose day is in (T + Δ, T + Δ + L], bedside readings excluded (`target_exclude`), no gender, no history, real age kept, at most `target_cap` tokens (the earliest kept, never cutting an event).
    - The target must hold at least one event; otherwise Δ and L are drawn again (`max_pair_tries`).
    - Z receives the **observed** length: L_obs = min(L, last day of the patient − (T + Δ)).

Time is counted in days for Δ and L: with Δ = 0, the target starts the day after T.

**Multiple views of a patient trajectory**: each (landmark, Δ, L) draw is one view, a different (context, target) split of the same trajectory.

- **Many landmarks per trajectory:** a new T is drawn each time the patient comes up (about once per epoch), so the same trajectory is split into context and target at many points (about 12 landmarks per patient over a D3 run).
- **Many Δ per landmark:** each landmark gets `n_pairs` = 4 pairs, each with its own Δ, plus new draws when the landmark comes back in a later epoch.
- **Many L_obs per target trajectory:** for the same T and Δ, each pair draws its own L, so the same future is seen over many horizons. L_obs = min(L, d_last − (T + Δ)): L_obs = L when the data covers the window, shorter only when the record ends inside it.
- Example: T = day 1,000, Δ = 40, data until day 2,500 → L = 30 gives L_obs = 30 (next month), L = 365 gives L_obs = 365 (next year), two nested windows with different targets y.

## Example (D3 encoder)

A woman, about 68 years old (age_days ≈ 24,850). T = discharge from an admission, day 24,850 at 14:00. `recent_days` = 30.

**Context**

| pos | token | type | source | value (scaled) | age_days | seconds_in_day | segment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | female | gender | patients | – | – | – | static |
| 1 | hypertension (I10) | code | diagnoses_icd | – | – | – | history |
| 2 | type 2 diabetes (E11.9) | code | diagnoses_icd | – | – | – | history |
| 3 | metformin | code | prescriptions | – | – | – | history |
| 4 | chest pain (R07.9) | code | diagnosis | – | – | – | history |
| 5 | creatinine | code | labevents | – | 24,847 | 30,600 | recent |
| 6 | Q8 | quantile | labevents | – | 24,847 | 30,600 | recent |
| 7 | [VALUE] | value | labevents | 0.67 | 24,847 | 30,600 | recent |
| 8 | mg/dL | unit | labevents | – | 24,847 | 30,600 | recent |
| 9 | heart rate | code | chartevents | – | 24,848 | 7,200 | recent |
| 10 | Q10 | quantile | chartevents | – | 24,848 | 7,200 | recent |
| 11 | [VALUE] | value | chartevents | 2.15 | 24,848 | 7,200 | recent |
| 12 | bpm | unit | chartevents | – | 24,848 | 7,200 | recent |
| 13 | furosemide | code | prescriptions | – | 24,848 | 39,600 | recent |
| 14 | heart failure, unspec. (I50.9) | code | diagnoses_icd | – | 24,850 | 50,400 | recent |

- History (positions 1–4): last-mention order; chest pain (ED visit 6 months earlier) is the most recent.
- Scaled value = (v − median) / IQR per concept and unit: creatinine (1.4 − 1.0) / 0.6 = 0.67; heart rate (128 − 85) / 20 = 2.15.
- Positions: RoPE, continuous from 0. c = mean of the student's outputs over positions 1–14.
- D2 encoder: the `[VALUE]` rows are dropped. D1: the Q and unit rows too.

**Target, pair 1: Δ = 40, L = 90** (days 24,890 to 24,980)

| pos | token | type | source | value | age_days | seconds_in_day | segment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | creatinine | code | labevents | – | 24,900 | 33,000 | recent |
| 1 | Q9 | quantile | labevents | – | 24,900 | 33,000 | recent |
| 2 | [VALUE] | value | labevents | 1.33 | 24,900 | 33,000 | recent |
| 3 | mg/dL | unit | labevents | – | 24,900 | 33,000 | recent |
| 4 | furosemide | code | prescriptions | – | 24,962 | 28,800 | recent |
| 5 | acute kidney failure (N17.9) | code | diagnoses_icd | – | 24,965 | 46,800 | recent |

y = the teacher's outputs averaged per day (24,900, 24,962, 24,965), then over the 3 days, then layer-normalised: each day counts the same, whatever its number of tokens.

**Z inputs**

| Pair | Δ | L drawn | Patient's data after T + Δ | L_obs | Z input |
| --- | --- | --- | --- | --- | --- |
| 1 | 40 | 90 | cover the whole window | 90 | c, f(40), f(90) |
| 2 | 300 | 200 | stop 50 days in | 50 | c, f(300), f(50) |

## Model

- **Student**: `TrajectoryBert` from the BERT run's `best.pt` (same channels: token, source, age, time of day, value in D3, RoPE positions), plus a segment embedding added to the input and initialised at zero, so at step 0 the student is exactly the BERT encoder. The masked-token heads of BERT are frozen and unused.
- **Teacher**: a copy of the student, updated after every step as an exponential moving average:

```math
\theta_{teacher} \leftarrow m\,\theta_{teacher} + (1 - m)\,\theta_{student}, \qquad m: 0.996 \rightarrow 1 \text{ (cosine)}
```

- **Pooling**: c = mean over the context tokens (gender and padding excluded); y = mean per day, then over days, then layer norm. Variants of c and Z: section Variants below.
- **Predictor Z**: an MLP (dim + 32 → 2·dim → 2·dim → dim, GELU) on [c, f(Δ), f(L_obs)], with

```math
f(x) = \big[\sin(\pi k u),\ \cos(\pi k u)\big]_{k=1..8}, \qquad u = \frac{\log(1+x)}{\log(1+x_{max})}
```

  where x_max is 730 days for Δ and 365 days for L.

## Loss

```math
\mathcal{L} = \underbrace{\frac{1}{P}\sum_{p}\frac{1}{d}\lVert \hat y_p - \mathrm{sg}(y_p)\rVert^2}_{\text{prediction}} + \lambda_{var}\,\frac{1}{d}\sum_j \max\!\big(0,\ 1 - \sigma_j(c)\big) + \lambda_{cov}\,\frac{1}{d}\sum_{i \neq j} \mathrm{Cov}(c)_{ij}^2
```

- p: a (context, target) pair of the batch, P pairs; d: embedding dimension (256)
- sg: stop-gradient (nothing flows into the teacher)
- σ_j(c): standard deviation of dimension j of c over the batch; the variance and covariance terms (VICReg) keep c from collapsing to one point
- λ_var = 1.0, λ_cov = 0.04 by default

**Blind guess**: `mean_target_loss` = the prediction loss of always answering the batch's mean target. `pred_ratio` = `pred_loss` / `mean_target_loss`: below 1 means Z uses the context.

## Optimisation

- AdamW (betas 0.9/0.98, weight decay 0.01), linear warm-up (2,000 steps), cosine decay to 10% of the learning rate.
- Transformer layers at lr × 0.1 (limit forgetting of BERT); embeddings, segment embedding and Z at full lr (3e-4).
- Packing as in BERT: contexts in one set of rows, targets in another (rows of 1,024 positions); each sample attends only to itself; positions restart at 0.
- bf16 on a GPU that supports it; gradient clipping at 1.
- `best.pt` = lowest validation `pred_ratio` (the raw loss is not comparable across steps, because the teacher moves).

## Monitoring (TensorBoard)

| Curve | What it measures | Good sign |
| --- | --- | --- |
| `val/pred_ratio` | prediction loss / blind guess | falls below 1 |
| `val/pred_loss`, `val/mean_target_loss` | the two parts of that ratio | the gap widens |
| `val/pred_loss_delta_<7d`, `_7-30d`, `_30-180d`, `_>=180d` | prediction loss per gap range | lower for short gaps; flat = Z ignores Δ |
| `val/pred_loss_no_history` | same validation pairs, history block removed | higher than `pred_loss` = the history helps |
| `val/std_c`, `val/std_y`, `val/std_y_hat` | mean per-dimension spread of c, y, ŷ | no sharp drop (collapse) |
| `val/rank_c`, `val/rank_y` | effective rank: exp(entropy of the normalised singular values) | no sharp drop |
| `train/var_term`, `train/cov_term` | the anti-collapse terms | `var_term` should not stay the largest part of the loss |
| `train/pairs_per_context`, `train/history_per_context` | pairs found (of `n_pairs`) and history items per context | stable |
| `train/ema_momentum`, `train/lr` | schedules | 0.996 → 1; warm-up then decay |
| `train/skipped_batches` | batches skipped because the loss was not finite | absent or 0 |

A batch whose loss is not finite is skipped and counted in `train/skipped_batches` (should stay 0).

First run (JEPA_D3, step 350): `std_c` 0.25 and `var_term` 0.75, the largest part of the loss. If `std_c` rises without `pred_ratio` improving, try `lambda_var: 0.1`.

## Configuration (`config\jepa_config.yaml`)

| Key | Default | Meaning |
| --- | --- | --- |
| `recent_days` | 30 | recent window before T, in days (also `--recent-days`) |
| `history_max` | 128 | history items kept (the most recent) |
| `history_sources` | diagnoses_icd, diagnosis, procedures_icd, prescriptions, pyxis, medrecon | what counts as history |
| `target_exclude` | vitalsign, triage, chartevents, outputevents | bedside readings left out of the target |
| `n_pairs` | 4 | (Δ, L) pairs per context |
| `delta_days` | 0–730 | gap after T, log scale |
| `length_days` | 1–365 | target window length, log scale |
| `max_pair_tries` | 10 | draws to find a non-empty target |
| `row_len` | 1,024 | positions per packed row |
| `models.D1/D2/D3` | context and target caps 512 / 800 / 1,000 | longest context and target, in tokens |
| `ema` | 0.996 → 1.0 | teacher momentum |
| `lambda_var`, `lambda_cov` | 1.0, 0.04 | anti-collapse terms |
| `time_features` | 8 | sine/cosine frequencies for Δ and L |
| `delta_bins` | 7, 30, 180 | gap ranges for the per-gap loss |

## Arguments (`jepa\train_jepa.py`)

| Argument | Default | Meaning |
| --- | --- | --- |
| `--init-run` | required | BERT run folder; decides D1/D2/D3 and the cut-off |
| `--init-checkpoint` | `best.pt` | or `last.pt` of the BERT run |
| `--config` | `config\jepa_config.yaml` | JEPA settings |
| `--trajectory-config` | `bert\trajectory_columns.yaml` | paths to the data |
| `--recent-days` | from the config | overrides `recent_days` |
| `--out-dir` | `D:\MIMIC\jepa_model` | where runs are saved |
| `--variant` | `base` | `base`, `concat_hist` (option A), `z_transformer` (option B) or `hist_z`, see Variants |
| `--run-name` | `JEPA_<model>_<date>` (base), `JEPA_<model>_<variant>` | run folder; an existing run without `--resume` is refused |
| `--resume` | off | continue the run `--run-name` from `--resume-checkpoint` |
| `--resume-checkpoint` | `last.pt` | or `best.pt`, to go back to the best step (e.g. after a NaN); TensorBoard hides the curve points logged after that step |
| `--prepare-only` | off | build the JEPA arrays and stop |
| `--batch-size` | 32 | contexts per batch (each with `n_pairs` targets) |
| `--lr`, `--lr-layers-scale`, `--min-lr-scale` | 3e-4, 0.1, 0.1 | learning rate, share for the transformer layers, end of the decay |
| `--warmup`, `--max-steps` | 2,000; 60,000 | schedule |
| `--eval-every`, `--eval-batches`, `--log-every` | 1,000; 50; 50 | evaluation and logging |
| `--num-workers` | 0 on Windows | batch preparation processes |
| `--seed`, `--device` | 42, cuda if available | |

## Outputs

Each run folder `D:\MIMIC\jepa_model\<run>` holds:

| File | Content |
| --- | --- |
| `best.pt` | lowest validation `pred_ratio`: student, teacher, predictor Z, optimizer, step, config |
| `last.pt` | saved after every evaluation: what `--resume` reads by default |
| `config.json` | arguments, JEPA settings, BERT run and its config, arrays folder |
| `tensorboard\` | the curves above |

## After training

- **c** (student encoder): the patient as recorded up to T.
- **ŷ = Z(c, Δ = g, L = H)** (student + Z): the expected representation of a disease's label window.
- One resilience score in each latent space, separately. Never mix c and ŷ in one neighbour search; every point, reference patients included, goes through the same Z(·, g, H); the (g, H) used for scoring must lie inside the trained ranges (Δ ≤ 730, L ≤ 365 days).
- The teacher (EMA weights) can be tried as an alternative embedder; Z must still take the student's c.

## Variants: how the context reaches Z (`--variant`, implemented 2026-11-24)

**Problem of the base design.** The student's token outputs are averaged into one vector before Z:

```math
c = \frac{1}{n}\sum_{i=1}^{n} h_i, \qquad \hat y = Z\big(c, f(\Delta), f(L_{obs})\big)
```

- h_i: student output of context token i (256); n: context tokens (gender and padding excluded); f: sin/cos features of `time_features` (16 + 16 numbers)

- The whole context (history + up to 1,000 recent tokens) is squeezed into 256 numbers before any prediction.
- About 15 history items against hundreds of recent tokens: the history weighs about 5 % of c (history gain 0.6–1.2 % in training).
- Dense days dominate c, while the target gives every day the same weight (`pool_target`).
- Signs: ŷ is much narrower than y (`std_y_hat` 0.53 vs `std_y` 0.98); ŷ never beats c in the resilience score.

The target side does not change in any variant: y = LayerNorm(mean over days of the per-day means of the teacher outputs), loss ‖ŷ − sg(y)‖² / d, VICReg on c, same sampling, same schedule.

| `--variant` | Name | c (anti-collapse terms and map) | Z | Z parameters |
| --- | --- | --- | --- | --- |
| `base` | JEPA (runs JEPA_D1–D3) | mean of the context tokens (256) | MLP on [c, f(Δ), f(L_obs)] | 0.54 M |
| `concat_hist` | option A, **JEPA concat hist** | [history mean ; recent per-day mean] (512) | the same MLP, input 512 + 1 + 32 | 0.67 M |
| `z_transformer` | option B, **JEPA Z transformer** | mean of the context tokens (256) | 2 cross-attention layers, one query per (Δ, L) | 1.20 M |

### Option A: JEPA concat hist (history and recent days pooled separately)

```math
c = \Big[\ \frac{1}{|H|}\sum_{i \in H} h_i\ \ ;\ \ \frac{1}{|D|}\sum_{t \in D}\frac{1}{|R_t|}\sum_{i \in R_t} h_i\ \Big] \in \mathbb{R}^{512}, \qquad \hat y = Z\big(c,\ e,\ f(\Delta),\ f(L_{obs})\big)
```

- H: history tokens (segment 1); D: days of the recent window; R_t: recent tokens (segment 2) of day t; gender (segment 0) is left out
- [ ; ]: side by side; e: 1 if the context has no history (its history half is then 0), else 0
- Z: the base MLP (512 + 1 + 32 → 512 → 512 → 256, GELU); output ŷ (256)
- Anti-collapse terms (var, cov) on the 512 vector c
- Resilience map `c`: each half scaled to unit length before the cosine (`map_vector`), so neither half dominates; a point without history keeps a zero history half

```math
\cos(c_i, c_j) \text{ on } \tilde c = \Big[\tfrac{c^{H}}{\lVert c^{H}\rVert}\ ;\ \tfrac{c^{R}}{\lVert c^{R}\rVert}\Big] \;\Rightarrow\; \cos(\tilde c_i,\tilde c_j) = \tfrac{1}{2}\big(\cos(c^H_i,c^H_j) + \cos(c^R_i,c^R_j)\big)
```

- c^H, c^R: history half and recent half of c (256 each); the equality holds when both points have a history

- Example: a context with 15 history items and 600 recent tokens over 9 days → c^H = mean of 15 vectors, c^R = mean of 9 day means. In base, the 15 items would weigh 15 / 615 ≈ 2 % of c.
- What it tests: is the problem the weight of the history and of dense days?
- Code: `pool_context_concat`, `Predictor(..., in_dim=2·dim+1)` with `extra` = e, `map_vector`.

### Option B: JEPA Z transformer (Z reads every context token)

No pooling before Z. For each (Δ, L) pair, a query token reads all context tokens (gender included):

```math
q^{(0)} = q_0 + \text{MLP}\big([\,f(\Delta)\,;\,f(L_{obs})\,]\big)
```

```math
a^{(\ell)}_i = \operatorname{softmax}_i\!\Big(\tfrac{1}{\sqrt{d_h}}\,(\bar q^{(\ell)} W_Q^{(\ell)})\cdot(\bar h_i W_K^{(\ell)})\Big), \qquad u^{(\ell)} = q^{(\ell)} + \Big(\sum_{i=1}^{n} a^{(\ell)}_i\,\bar h_i W_V^{(\ell)}\Big) W_O^{(\ell)}
```

```math
q^{(\ell+1)} = u^{(\ell)} + \text{FFN}^{(\ell)}\big(\text{LN}(u^{(\ell)})\big), \qquad \hat y = W_y\,\text{LN}\big(q^{(2)}\big)
```

- q_0: learned vector (256); MLP: 32 → 256 → 256 (GELU); ℓ = 0, 1 (2 layers); \bar q, \bar h_i: layer-normalised; W_Q, W_K, W_V, W_O, W_y: learned 256 × 256 matrices; 4 heads, d_h = 64; FFN: 256 → 512 → 256 (GELU)
- a_i: weight of context token i for this question (Σ a_i = 1; padding gets 0 through `key_padding_mask`); only q is updated, the h_i stay fixed (cross-attention)
- With every a_i = 1/n and no FFN it is the base design: the attention generalises mean pooling, with weights that depend on (Δ, L) and on each token
- Like `[CLS]` in BERT (a token whose output summarises what it attended to), but conditioned on (Δ, L), placed after the encoder, and trained by the JEPA loss
- 4 pairs per context = 4 queries on the same h; each sees only its own context
- c = mean of h is still computed: for the anti-collapse terms and for the map `c`. ŷ = Z(h, g, H) is the second map worth scoring, since it now comes from the whole sequence
- Example: for Δ = 0, L = 365 the query may put most weight on a few history diagnoses; for Δ = 0, L = 7 on the last days' labs
- What it tests: is the single-vector squeeze the problem?
- Code: `context_tokens` (padded contexts × longest context × 256 + mask), `QueryLayer`, `TransformerPredictor`; settings `z_transformer: {layers, heads, ff_mult}` in the config

Checked on synthetic data: concat pooling = direct computation (max |diff| 2e-7); filling the padding with noise does not change ŷ (0.0); one context alone = the same context in a batch (7e-7); ŷ changes when the history is removed; attention weights sum to 1.

### What each variant changes in the code

| Part | concat_hist | z_transformer |
| --- | --- | --- |
| `train_jepa.py` | `--variant concat_hist`; c from `pool_context_concat`, flag to Z | `--variant z_transformer`; h → `context_tokens` → `TransformerPredictor(H[pair_ctx], pad[pair_ctx], Δ, L)`; c = `pool_context` |
| checkpoint, `config.json` | `"variant"` recorded; `--resume` refuses another variant | same |
| `resilience\embed_points.py` | reads the variant; map `c` = 512 (halves unit length); maps `yhat_*` from its Z | map `c` (mean, 256); maps `yhat_*` from the cross-attention Z |
| `resilience\score.py` | unchanged (any width) | unchanged |
| `resilience\compare_maps.py` | default pairs vs JEPA_Dk and BERT_Dk | same, for `c` and `yhat_g0_h365` |

Runs trained before the variants (JEPA_D1–D3) have no `"variant"` in `config.json`: read as `base`.

### Training and comparison plan

- Start from the same BERT `best.pt` as the base run, all other settings unchanged: `JEPA_D2_concat_hist`, `JEPA_D2_z_transformer` first (about 2.5 h each for concat_hist; z_transformer slower, not timed on the GPU yet), then D1 and D3 if they help.
- Compare with JEPA_D2 on validation: `pred_ratio`, loss per Δ range, history gain (`pred_loss_no_history`), `std_y_hat` vs `std_y`, `rank_c`.
- Then the resilience score: AUROC and PPV top 5 % per disease, neighbour overlap and candidate Jaccard with JEPA_D2 and BERT_D2 (`compare_maps.py`).
- Success: ŷ closer to y in spread, a larger history gain, and a map at least as good at the gate; then the candidate lists decide.

### Results (D2, 2026-11-24)

| Run | best pred_ratio (step) | steps/s | std_c | rank_c | history gain | std ŷ / std y |
| --- | --- | --- | --- | --- | --- | --- |
| JEPA_D2 (base) | 0.672 (57k) | 8.9 | 0.93 | 212 of 256 | 0.93 % | 0.60 / 0.99 |
| JEPA_D2_concat_hist | 0.659 (35k) | 8.2 | 0.70 | 364 of 512 | 0.92 % | 0.60 / 0.98 |
| JEPA_D2_z_transformer | **0.638** (57k) | 6.0 | 0.93 | 214 of 256 | 1.08 % | 0.63 / 0.99 |

- concat hist: better objective from per-day pooling of the recent window; history gain and std ŷ unchanged; std_c lower by construction (zero history half without history).
- Z transformer: best objective, still falling at 60k; history gain and std ŷ up a little; 33 % slower.
- Resilience (val, H = 365, AUROC of c): Z transformer CKD 0.795 → 0.821 (intervals apart), other diseases equal; concat hist equal at H = 365, best at H = 180 for AKI, sepsis, T2D. Each keeps 35–39 of JEPA D2's 100 neighbours. Details: `resilience\README.md`, Results and R5.
- Neither variant reaches "ŷ as spread as y" (0.63 vs 0.99).

### Variant hist_z: history and static information go to Z (implemented 2026-11-26)

The encoder only ever reads windows of events; the history and the age reach the prediction through Z.

| Part | Input | How |
| --- | --- | --- |
| Recent sequence | gender + recent window (labs with bins and values) | student encoder f_θ, as a BERT segment |
| History sequence | history items (diagnoses, procedures, drugs) | the same f_θ, separate pass (own segment number, so the two never attend to each other), history segment embedding |
| Z query | f(Δ), f(L_obs), f(age at T), no-history flag | q = q0 + MLP(...) |
| Z keys | recent tokens + e_recent ; history tokens + e_history ; one learned empty token | 3 cross-attention layers (4 heads, FFN × 2), 1.73 M parameters |
| Target | future window, bedside sources removed | teacher f_θ̄ (EMA), unchanged |

```math
q^{(0)} = q_0 + \mathrm{MLP}\big([\,f(\Delta) ;\ f(L_{obs}) ;\ f(\text{age}_T) ;\ \mathbb{1}[\text{no history}]\,]\big), \qquad \hat y = W_y\,\mathrm{LN}\big(q^{(3)}\big)
```

- q₀: learned starting query; f: sin/cos of log(1 + days) at 8 frequencies; age_T: age at the landmark in days (scaled by `max_age_days` 40,000)
- the empty token gives attention a place to go when nothing fits; the flag tells Z the history is missing
- c = recent window pooled like the target (mean per day, then over days): anti-collapse terms and map `c`
- maps (`embed_points.py`): `c`, `yhat_g0_h365` / `h180`, and `query_g0_h365` / `h180` = Z's final query before its output layer, the one vector that combines recent window, history and age
- history gain = ŷ with vs without the history sequence: exactly what Z takes from the history
- code: `split_context`, `pack_contexts` (`jepa_data.py`); `HistZPredictor`, `pool_days`, `history_tokens` (`jepa_model.py`); settings `hist_z: {layers, heads, ff_mult, max_age_days}` in the config
- checked on synthetic data: recent tokens and c do not change when the history is removed; history tokens = the history encoded alone; ŷ changes with the history only for patients who have one; one context alone = the same context in a batch (5e-7)
- D3 result (2026-11-27): best pred_ratio 0.677 at 60k (Z transformer D3 0.622 at 57k), plateau since 30k, converged; history gain 1.13 % (largest of all runs; Z transformer D3 0.88 %); std ŷ 0.60; 3.1 steps/s (Z transformer D3 5.1). Resilience (val, H = 365): ties Z transformer D3 in AUROC (CKD 0.834 vs 0.833), best CKD PPV top 5 % (0.358 vs 0.339, within the ±0.03 intervals); details in `resilience\README.md`, Results
- D2 result (2026-11-27): best pred_ratio 0.688 at 59k (Z transformer D2 0.638 at 57k), still −0.004 per 10k steps at the end (close to, but not fully, converged); history gain 0.43 → 0.96 % (Z transformer D2 1.08 %); std ŷ 0.59 (std y 0.99); 3.95 steps/s, 4.2 h. Resilience (val, H = 365): ties Z transformer D2 in AUROC (CKD 0.817 vs 0.821; test 0.824 vs 0.830); CKD PPV top 5 % 0.337 vs 0.318 on val but 0.356 vs 0.376 on test, so the top-5 % order is not stable

Other target ideas noted the same day (not decided): drop the time of day from target tokens (y averages days anyway); add `inputevents` to `target_exclude` (dense ICU drug and fluid rates, like the vitals already excluded).

## Code

| File | Content |
| --- | --- |
| `jepa\jepa_data.py` | episodes and landmarks, JEPA arrays, sample building (landmark, history, recent window, targets), packing, fixed validation batches |
| `jepa\jepa_model.py` | `load_bert`, student (`JepaEncoder`), teacher (`make_teacher`, `ema_update`), `Predictor`, `TransformerPredictor`, pooling of c and y (`pool_context`, `pool_context_concat`, `context_tokens`), variant helpers (`make_predictor`, `context_inputs`, `run_predictor`, `map_vector`), anti-collapse terms, effective rank |
| `jepa\train_jepa.py` | arguments, training loop, evaluation, TensorBoard, checkpoints |
| `config\jepa_config.yaml` | the settings above |
| `bert\bert_model.py`, `bert\bert_data.py` | the BERT encoder and arrays it starts from (with the two hooks JEPA uses: `forward(b, extra)` and `to_model_rows(..., keep)`) |

Tested end to end on 300 synthetic patients from D1, D2 and D3 BERT runs, including resume; every sampling rule above checked on every sample.

## Command reference

All commands in PowerShell, from the `jepa` folder.

**Start a run** (the first run of a model also builds its JEPA arrays)

```
cd C:\Users\yy\Desktop\codes_PhD\jepa_mimic\jepa
python train_jepa.py --init-run D:\MIMIC\bert_model\D3_min50 --run-name JEPA_D3
python train_jepa.py --init-run D:\MIMIC\bert_model\D1_min50_20261120_xxxx --run-name JEPA_D1
python train_jepa.py --init-run D:\MIMIC\bert_model\D2_min50 --run-name JEPA_D2
```

**Variants A and B** (same BERT start as JEPA_D2; the run name is set from the variant)

```
python train_jepa.py --init-run D:\MIMIC\bert_model\D2_min50 --variant concat_hist      # -> JEPA_D2_concat_hist
python train_jepa.py --init-run D:\MIMIC\bert_model\D2_min50 --variant z_transformer    # -> JEPA_D2_z_transformer
python train_jepa.py --init-run D:\MIMIC\bert_model\D2_min50 --variant concat_hist --resume    # resume one
```

**Variant hist_z** (D3 and D2 done; D1 optional)

```
python train_jepa.py --init-run D:\MIMIC\bert_model\D3_min50 --variant hist_z                  # -> JEPA_D3_hist_z
python train_jepa.py --init-run D:\MIMIC\bert_model\D1_min50_20261120_xxxx --variant hist_z    # -> JEPA_D1_hist_z
python train_jepa.py --init-run D:\MIMIC\bert_model\D2_min50 --variant hist_z                  # -> JEPA_D2_hist_z
```

If z_transformer runs out of GPU memory, add `--batch-size 16` (a new run name is then advised, e.g. `--run-name JEPA_D2_z_transformer_b16`).

**Change a setting for a new run** (a new `--run-name` each time)

```
python train_jepa.py --init-run D:\MIMIC\bert_model\D3_min50 --run-name JEPA_D3_recent60 --recent-days 60
python train_jepa.py --init-run D:\MIMIC\bert_model\D3_min50 --run-name JEPA_D3_last --init-checkpoint last.pt
```

Other settings (`lambda_var`, `n_pairs`, `target_exclude`, ...) are changed in `config\jepa_config.yaml`, or in a copy passed with `--config`.

**Stop and resume**

- Stop with Ctrl+C right after an `eval step …` line, so `last.pt` is fresh.
- Resume with the same command plus `--resume`, the same `--init-run` and the same config:

```
python train_jepa.py --init-run D:\MIMIC\bert_model\D3_min50 --run-name JEPA_D3 --resume
```

- If the losses became NaN, `last.pt` holds NaN weights: go back to the best step instead (`best.pt` is never saved with a NaN ratio):

```
python train_jepa.py --init-run D:\MIMIC\bert_model\D3_min50 --run-name JEPA_D3 --resume --resume-checkpoint best.pt
```

**Only build the arrays**

```
python train_jepa.py --init-run D:\MIMIC\bert_model\D1_min50_20261120_xxxx --prepare-only
```

**TensorBoard**

```
tensorboard --logdir D:\MIMIC\jepa_model                    # JEPA runs
tensorboard --logdir D:\MIMIC                               # BERT and JEPA runs together
python -m tensorboard.main --logdir D:\MIMIC\jepa_model     # if "tensorboard" is not recognised
```

Then open `http://localhost:6006`; close TensorBoard with Ctrl+C in its window. One training at a time: each run fills the 12 GB GPU.
