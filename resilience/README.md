# Resilience score on the BERT and JEPA maps (MIMIC-IV)

Goal: find points that look like patients who get disease X, but who do not get it. Rules: HERO plan (`claude/hero_resilience_plan.md`, claude.ai project "Mimic Jepa"); disease rules from the MIMIC pilot. Settings: `config\resilience_config.yaml`. Results below: 2026-11-24.

## Pipeline

| Step | Script | What it does | Output (`D:\MIMIC\trajectory_out\resilience`) |
| --- | --- | --- | --- |
| R1 | `build_points.py` | one point per admission or ED visit end | `points.parquet` |
| R2 | `embed_points.py` | one vector per point and map | `embeddings\<name>\<map>.npy` |
| R3 | `build_labels.py` | status of every point per disease and horizon | `labels.parquet`, `label_counts.csv` |
| R4 | `score.py` | neighbours, score r, gate, list | `scores\<name>\<map>\`, `scores\metrics_all.csv` |
| R5 | `compare_maps.py` | neighbour and candidate overlap between maps | `scores\compare_maps.csv` |

R2 and R3 are independent: the vectors never see a label.

## R1 Points

- **Point** = a patient's history up to a **landmark T** = the end of an admission (`H…`, with the ED visit that led to it) or of an ED visit without admission (`E…`). All episodes, the first one included; outpatient days are not points.
- T = the day of the episode's last token (age_days), as in JEPA training.
- **h** = admissions and ED visits ended before this one started; h = 0 = new point.
- 767,911 points; split by patient 60 / 20 / 20 (train / val / test), as for BERT and JEPA.

## R2 Maps

| Map | Encoder and input | Distance |
| --- | --- | --- |
| BERT D1, D2, D3 | BERT before JEPA; gender + recent window (BERT never saw a history block) | cosine |
| JEPA Dk c, no history | JEPA student; gender + recent window (history removed at test time) | cosine |
| JEPA Dk c | JEPA student; gender + history + recent window (as in training) | cosine |
| JEPA Dk ŷ | Z(c, Δ = 0, L = 365): expected embedding of the next year | cosine |
| JEPA Dk concat hist c | option A: [history mean ; recent per-day mean], each half unit length (512) | cosine |
| JEPA Dk concat hist ŷ | MLP Z on [c, no-history flag, f(Δ), f(L)] | cosine |
| JEPA Dk Z transformer c | option B: mean of the context tokens (256), as base | cosine |
| JEPA Dk Z transformer ŷ | cross-attention Z: one query (Δ = 0, L = 365) reads every context token | cosine |
| bag of codes | concepts of the JEPA context (history + recent), log counts × idf, SVD 256 | cosine |
| demographics | age, sex, log(1 + h), standardised | euclidean |
| random | random vectors (sanity check) | cosine |

- Context: built by `jepa_data.context_at`, the training rule. Recent window = the 30 days up to T, capped at 512 / 800 / 1,000 tokens (D1 / D2 / D3); history = past diagnoses, drugs and procedures, each once at its last mention, at most 128.
- The recent window is token for token the same in BERT, JEPA without history and JEPA with history.
- Vector = mean of the encoder's outputs over the context tokens (gender excluded); stored as float16.
- The variant (`base`, `concat_hist`, `z_transformer`) is read from the JEPA run's `config.json` and recorded in `info.json`; the context is pooled and Z is run exactly as in training (`context_inputs`, `run_predictor`, `map_vector` of `jepa_model.py`). Spec of the variants: `jepa\README.md`, section Variants.
- concat hist map: cosine on the two unit halves = ½ (cos of the history halves + cos of the recent halves); a point without history keeps a zero history half (with `--no-history`: only the recent half counts).

## R3 Labels

Time in days since birth. Horizon window = (T + g, T + g + H], with g = 0 and H = 365 days (H = 180 also computed).

| Status (first that applies) | Condition | Own label y |
| --- | --- | --- |
| prevalent | o ≤ T | – |
| excluded | T2D: type 1 code by T + g + H | – |
| confirmed_in_gap | T < o ≤ T + g | – |
| positive | T + g < o ≤ T + g + H | 1 |
| death | d ≤ T + g + H, no X | – |
| no_followup | v < T + g + H, no X | – |
| tested_abnormal | tests in (T, T + g + H], at least one abnormal | 0 |
| tested_negative | tests in the window, all normal | 0 |
| untested | no test in the window | 0 |

- o: first onset of X (earliest of code, lab rule, X-specific drug); d: death day; v: last visit day (admission, ED visit or outpatient day in `omr`; isolated labs do not count); T: landmark day; g: gap; H: horizon.

| Disease | Onset rule | Test; abnormal |
| --- | --- | --- |
| T2D | ICD E11 / 250.x0, 250.x2; HbA1c ≥ 6.5 %; non-insulin antidiabetic (home meds, prescriptions) | HbA1c; ≥ 5.7 % |
| CKD | ICD N18 / 585; eGFR (CKD-EPI 2021) < 60 on consecutive values over ≥ 90 days | eGFR; < 60 |
| AKI | ICD N17 / 584; KDIGO: creatinine +0.3 mg/dL within 48 h, or ≥ 1.5 × the lowest of 7 days | ≥ 2 creatinine values; > 1.2 mg/dL |
| Sepsis | ICD A40–A41 / R65.2 / 038 / 995.91–92 / 785.52; simplified Adult Sepsis Event (blood culture + ≥ 4 antibiotic days + one organ dysfunction within 48 h) | blood culture; organism grown |

- A code takes the start day of its visit; labs by collection time; drugs by start time.
- To confirm with the clinician: the abnormal thresholds, the T2D drug list, the AKI threshold.

## R4 Resilience score

**Neighbours**: for each point, its K = 100 nearest **train** points of **other** patients, each other patient once (its nearest point). Searched once per map, reused for every disease.

```math
r_i = \frac{\#\{j \in N(i) : s_j = \text{positive}\}}{\#\{j \in N(i) : s_j \in \{\text{positive, tested abnormal, tested negative, untested}\}\}}
```

- r_i: resilience (neighbour) score of point i
- N(i): the K = 100 neighbours of point i
- s_j: status of neighbour j for disease X and horizon H
- Neighbours with status death, no follow-up, prevalent, excluded or gap are not counted. Fewer than n_min = 20 counted neighbours: no score, the point gets π.

**Patient weight and prevalence**

```math
w_i = \frac{1}{m_{p(i)}}, \qquad \pi = \frac{\sum_{i \in \text{train}} w_i\, y_i}{\sum_{i \in \text{train}} w_i}
```

- p(i): patient of point i; m_p: number of evaluated points of patient p in the set
- y_i: own label of point i (1 positive; 0 tested abnormal, tested negative, untested)
- π: prevalence among train points with a known outcome, each patient counted once

**List**: candidates = top q of r with own status tested abnormal, tested negative or untested; q = 5 % or 10 %.

## Evaluation

Evaluated points: own label y known (positive, tested abnormal, tested negative, untested). Splits: val chooses, test is reported once, train is the leakage check.

**AUROC** (patient-weighted, ties counted half):

```math
\text{AUROC} = \frac{\sum_{i:\,y_i = 1}\ \sum_{j:\,y_j = 0} w_i\, w_j \left(\mathbb{1}[r_i > r_j] + \tfrac{1}{2}\,\mathbb{1}[r_i = r_j]\right)}{\left(\sum_{i:\,y_i = 1} w_i\right)\left(\sum_{j:\,y_j = 0} w_j\right)}
```

- i: a positive point; j: a negative point; w: patient weights; 𝟙[·]: 1 if true, else 0
- the probability that a random positive is ranked above a random negative; random scores give 0.5
- computed in one pass after sorting by r; equal to scikit-learn `roc_auc_score(..., sample_weight=w)`

**Positive rate in the top q** (the gate for the list):

```math
\text{PPV}_q = \frac{\sum_{i \in \text{top } q} w_i\, y_i}{\sum_{i \in \text{top } q} w_i}, \qquad \text{lift}_q = \frac{\text{PPV}_q}{\pi}
```

- top q: the points with the highest r covering a share q of the total weight (ties at random)

**95 % intervals**: 200 resamples of patients (a patient drawn m times has its weights × m).

## Results (validation, H = 365, 2026-11-24)

Val: AKI 49,555 points (18,176 patients), CKD 55,884 (17,796), sepsis 61,600 (19,293), T2D 49,736 (15,766).

**AUROC (95 % interval)**, best in bold:

| Map | History | AKI | CKD | Sepsis | T2D |
| --- | --- | --- | --- | --- | --- |
| BERT D1 | – | 0.769 (0.76–0.78) | 0.762 (0.75–0.77) | 0.798 (0.79–0.81) | 0.618 (0.60–0.63) |
| BERT D2 | – | 0.779 (0.77–0.79) | 0.793 (0.78–0.80) | **0.808 (0.80–0.82)** | **0.644 (0.63–0.66)** |
| BERT D3 | – | 0.775 (0.77–0.78) | 0.800 (0.79–0.81) | 0.802 (0.79–0.81) | 0.634 (0.62–0.65) |
| JEPA D1 c | no | 0.775 (0.77–0.78) | 0.771 (0.76–0.78) | 0.803 (0.79–0.81) | 0.623 (0.60–0.64) |
| JEPA D2 c | no | 0.778 (0.77–0.79) | 0.796 (0.79–0.81) | 0.804 (0.79–0.81) | 0.630 (0.61–0.65) |
| JEPA D3 c | no | 0.774 (0.77–0.78) | 0.800 (0.79–0.81) | 0.797 (0.79–0.81) | 0.633 (0.61–0.65) |
| JEPA D1 c | yes | **0.779 (0.77–0.79)** | 0.769 (0.76–0.78) | 0.805 (0.79–0.81) | 0.617 (0.60–0.63) |
| JEPA D2 c | yes | 0.779 (0.77–0.79) | 0.795 (0.79–0.80) | 0.805 (0.79–0.81) | 0.632 (0.61–0.65) |
| JEPA D3 c | yes | 0.775 (0.77–0.78) | 0.802 (0.79–0.81) | 0.799 (0.79–0.81) | 0.640 (0.62–0.66) |
| JEPA D1 ŷ | yes | 0.773 (0.77–0.78) | 0.767 (0.75–0.78) | 0.799 (0.79–0.81) | 0.618 (0.60–0.64) |
| JEPA D2 ŷ | yes | 0.776 (0.77–0.78) | 0.799 (0.79–0.81) | 0.800 (0.79–0.81) | 0.637 (0.62–0.66) |
| JEPA D3 ŷ | yes | 0.769 (0.76–0.78) | 0.795 (0.79–0.80) | 0.793 (0.78–0.80) | 0.630 (0.61–0.65) |
| JEPA D2 concat hist c | yes | 0.774 (0.77–0.78) | 0.792 (0.78–0.80) | 0.804 (0.79–0.81) | 0.644 (0.63–0.66) |
| JEPA D2 concat hist ŷ | yes | 0.774 (0.77–0.78) | 0.791 (0.78–0.80) | 0.795 (0.78–0.80) | 0.630 (0.61–0.65) |
| JEPA D2 Z transformer c | yes | 0.779 (0.77–0.79) | **0.821 (0.81–0.83)** | 0.805 (0.80–0.81) | 0.638 (0.62–0.66) |
| JEPA D2 Z transformer ŷ | yes | 0.774 (0.77–0.78) | 0.812 (0.80–0.82) | 0.796 (0.79–0.80) | 0.620 (0.60–0.64) |
| bag of codes | yes | 0.755 (0.75–0.76) | 0.742 (0.73–0.75) | 0.784 (0.78–0.79) | 0.603 (0.59–0.62) |
| demographics | – | 0.648 (0.64–0.66) | 0.708 (0.70–0.72) | 0.627 (0.62–0.64) | 0.588 (0.57–0.61) |
| random | – | 0.492 (0.48–0.50) | 0.500 (0.49–0.51) | 0.498 (0.49–0.51) | 0.477 (0.46–0.50) |

**PPV top 5 %**, best in bold:

| Map | History | AKI | CKD | Sepsis | T2D |
| --- | --- | --- | --- | --- | --- |
| prevalence π | | 0.152 | 0.072 | 0.098 | 0.041 |
| BERT D1 | – | 0.55 | 0.27 | 0.48 | 0.08 |
| BERT D2 | – | 0.51 | 0.31 | 0.48 | 0.09 |
| BERT D3 | – | 0.49 | 0.31 | 0.46 | 0.08 |
| JEPA D1 c | no | 0.55 | 0.27 | **0.50** | 0.09 |
| JEPA D2 c | no | 0.54 | 0.29 | 0.48 | 0.10 |
| JEPA D3 c | no | 0.50 | 0.30 | 0.45 | 0.10 |
| JEPA D1 c | yes | **0.56** | 0.27 | 0.49 | 0.09 |
| JEPA D2 c | yes | 0.54 | 0.31 | 0.48 | **0.11** |
| JEPA D3 c | yes | 0.50 | 0.31 | 0.45 | 0.10 |
| JEPA D1 ŷ | yes | 0.54 | 0.25 | 0.47 | 0.07 |
| JEPA D2 ŷ | yes | 0.53 | 0.28 | 0.47 | 0.08 |
| JEPA D3 ŷ | yes | 0.49 | 0.30 | 0.45 | 0.10 |
| JEPA D2 concat hist c | yes | 0.55 | 0.32 | 0.50 | 0.10 |
| JEPA D2 concat hist ŷ | yes | 0.53 | 0.29 | 0.47 | 0.08 |
| JEPA D2 Z transformer c | yes | 0.52 | 0.32 | 0.48 | 0.08 |
| JEPA D2 Z transformer ŷ | yes | 0.51 | **0.32** | 0.47 | 0.09 |
| bag of codes | yes | 0.56 | 0.28 | 0.49 | 0.09 |
| demographics | – | 0.30 | 0.18 | 0.17 | 0.08 |
| random | – | 0.15 | 0.07 | 0.11 | 0.04 |

- Every learned map is far above demographics and random (≈ 0.5): the maps carry real risk.
- BERT and JEPA beat bag of codes on AUROC for all four diseases (+0.02 to +0.06), but only match it in the top 5 %.
- BERT ≈ JEPA without history ≈ JEPA with history: within 0.015 AUROC everywhere, with overlapping intervals. ŷ never beats c.
- D2 / D3 help the lab-defined diseases (CKD: D1 0.76–0.77, D2 / D3 0.79–0.80); D1 is as good for AKI and sepsis.
- No leakage: train − test AUROC from −0.012 to +0.019 over all maps.

**JEPA variants (D2; spec in `jepa\README.md`, section Variants).** H = 180 in brackets.

| Map c | AKI | CKD | Sepsis | T2D | Coverage |
| --- | --- | --- | --- | --- | --- |
| JEPA D2 | 0.779 (0.786) | 0.795 (0.804) | 0.805 (0.810) | 0.632 (0.625) | 0.87–0.92 |
| concat hist (A) | 0.774 (**0.787**) | 0.792 (0.813) | 0.804 (**0.815**) | 0.644 (**0.645**) | 0.84–0.89 |
| Z transformer (B) | 0.779 (0.781) | **0.821** (**0.826**) | 0.805 (0.812) | 0.638 (0.621) | 0.88–0.92 |

- Z transformer, CKD: 0.795 → 0.821, intervals apart (0.786–0.805 vs 0.813–0.829); test 0.829, train 0.819. Best map of all for CKD at both horizons; PPV top 5 % CKD 0.32 (ŷ 0.323, best of all). Other diseases equal to JEPA D2; T2D PPV lower (0.085 vs 0.105).
- concat hist: within ±0.012 of JEPA D2 at H = 365 (T2D +0.012, ties BERT D2); best of all maps at H = 180 for AKI, sepsis and T2D; best sepsis PPV (0.50); coverage 2–4 points lower (cause not checked).
- ŷ stays at or below c in AUROC for both variants; in PPV it is higher only for the Z transformer, CKD and T2D.
- A better JEPA objective (pred_ratio 0.672 → 0.659 → 0.638) does not carry over evenly: reading every token helps CKD (repeated creatinine and eGFR in the context), even day pooling helps the short horizon, nothing fixes T2D.

## R5 Map comparison

**Although BERT and JEPA barely differ in AUROC, their latent spaces differ a lot: they share only 3–6 of each point's 100 neighbours, and about 30 % of their candidate patients.**

**Neighbour overlap** (val points):

```math
\text{overlap}(A, B) = \frac{\sum_i w_i\, \frac{|N_A(i) \cap N_B(i)|}{K}}{\sum_i w_i}
```

- N_A(i), N_B(i): the K = 100 neighbours of point i in maps A and B
- w_i: patient weight; 0 = unrelated maps (random), 1 = the same neighbourhoods

**Candidate overlap** (per disease, top 5 %, H = 365):

```math
J(A, B) = \frac{|C_A \cap C_B|}{|C_A \cup C_B|}
```

- C_A, C_B: sets of candidate patients of the two lists

| Pair | Neighbour overlap | J AKI | J CKD | J sepsis | J T2D |
| --- | --- | --- | --- | --- | --- |
| BERT D1 vs JEPA D1 (no history) | 0.064 | 0.32 | 0.29 | 0.37 | 0.23 |
| BERT D2 vs JEPA D2 (no history) | 0.040 | 0.28 | 0.30 | 0.34 | 0.24 |
| BERT D3 vs JEPA D3 (no history) | 0.027 | 0.30 | 0.26 | 0.33 | 0.23 |
| JEPA D1, no history vs with | 0.749 | 0.64 | 0.59 | 0.66 | 0.48 |
| JEPA D2, no history vs with | 0.778 | 0.61 | 0.63 | 0.63 | 0.52 |
| JEPA D3, no history vs with | 0.807 | 0.66 | 0.64 | 0.67 | 0.53 |
| BERT D1 vs BERT D2 | 0.202 | 0.34 | 0.35 | 0.40 | 0.28 |
| BERT D2 vs BERT D3 | 0.431 | 0.44 | 0.44 | 0.53 | 0.34 |
| BERT D2 vs bag of codes | 0.104 | 0.31 | 0.33 | 0.39 | 0.24 |
| JEPA D2 vs bag of codes | 0.032 | 0.27 | 0.27 | 0.35 | 0.22 |
| BERT D2 vs random | 0.000 | 0.07 | 0.09 | 0.09 | 0.14 |
| JEPA D2 vs concat hist (c) | 0.389 | 0.44 | 0.44 | 0.53 | 0.34 |
| JEPA D2 vs Z transformer (c) | 0.346 | 0.40 | 0.42 | 0.48 | 0.34 |
| JEPA D2 vs concat hist (ŷ) | 0.456 | 0.49 | 0.47 | 0.54 | 0.39 |
| JEPA D2 vs Z transformer (ŷ) | 0.424 | 0.47 | 0.45 | 0.51 | 0.37 |
| concat hist vs Z transformer (c) | 0.272 | 0.36 | 0.38 | 0.44 | 0.29 |
| BERT D2 vs concat hist (c) | 0.038 | 0.26 | 0.29 | 0.34 | 0.24 |
| BERT D2 vs Z transformer (c) | 0.030 | 0.28 | 0.28 | 0.32 | 0.23 |

- JEPA training moved the neighbourhoods further from BERT than bag of codes is from BERT.
- The two variants keep only 35–39 of JEPA D2's 100 neighbours and 27 of each other's (encoder and Z learn together, so a new Z also gives a new encoder); both stay as far from BERT as JEPA D2 (3–4).
- The history block changes about a fifth of the neighbours and about 40 % of the candidates.
- Equal AUROC, different patients: the choice between maps rests on the lists (tested negatives, "not yet", expert review), not on AUROC.

## Tests (synthetic data, 300 patients)

- Points: the landmark token's day = T in D1, D2, D3; patient lengths match the master table.
- Vectors = a direct computation one context at a time (difference = float16 rounding); `context_at` leaves training samples unchanged. Same check for the concat hist and Z transformer runs (c and ŷ, max |diff| 1e-4 and 5e-4).
- Labels: 12 hand-made cases (one per status, boundaries included); every label = a plain-loop recomputation; KDIGO, CKD run, antibiotic runs and ASE on hand cases.
- Neighbours = brute-force search; r = a direct count; AUROC = scikit-learn; overlap = a brute-force count.

## Commands

From `C:\Users\yy\Desktop\codes_PhD\jepa_mimic\resilience`; the bag baseline needs `pip install scikit-learn scipy`.

```
python build_points.py
python build_labels.py
python embed_points.py --run D:\MIMIC\jepa_model\JEPA_D1                  # also JEPA_D2, JEPA_D3
python embed_points.py --run D:\MIMIC\jepa_model\JEPA_D1 --no-history     # saved as JEPA_D1_nohist
python embed_points.py --bert-run D:\MIMIC\bert_model\D1_min50_20261120_xxxx   # saved as BERT_D1
python embed_points.py --bert-run D:\MIMIC\bert_model\D2_min50
python embed_points.py --bert-run D:\MIMIC\bert_model\D3_min50
python embed_points.py --baseline bag                                     # also demo, random
python score.py --embedding JEPA_D1 JEPA_D2 JEPA_D3 JEPA_D1_nohist JEPA_D2_nohist JEPA_D3_nohist BERT_D1 BERT_D2 BERT_D3 bag demo random
python compare_maps.py
```

Variants A and B (after `train_jepa.py --variant concat_hist` / `--variant z_transformer`, see `jepa\README.md`):

```
python embed_points.py --run D:\MIMIC\jepa_model\JEPA_D2_concat_hist                # maps c (512), yhat_g0_h365, yhat_g0_h180
python embed_points.py --run D:\MIMIC\jepa_model\JEPA_D2_z_transformer              # maps c (256), yhat_g0_h365, yhat_g0_h180
python embed_points.py --run D:\MIMIC\jepa_model\JEPA_D2_concat_hist --no-history   # optional: JEPA_D2_concat_hist_nohist
python score.py --embedding JEPA_D2_concat_hist JEPA_D2_z_transformer
python compare_maps.py                     # default pairs now include the variants (pairs not built yet are skipped)
```

Options:

```
python embed_points.py --run D:\MIMIC\jepa_model\JEPA_D3 --checkpoint last.pt    # saved as JEPA_D3_last
python embed_points.py --run D:\MIMIC\jepa_model\JEPA_D3 --batch 64              # if the GPU runs out of memory
python build_labels.py --diseases ckd aki                                        # some diseases only
python score.py --embedding JEPA_D3 --map c                                      # one map
python score.py --collect                                                        # rebuild metrics_all.csv only
python compare_maps.py --pairs BERT_D2/c:JEPA_D2/c --q 0.10
```
