# MIMIC resilience: patient trajectories and trajectory encoders

Code for a PhD project on disease resilience: patient trajectories are built from MIMIC-IV, encoded by a
transformer pre-trained first with masked tokens (BERT) and then with a joint-embedding predictive objective
(JEPA), and used to find patients who look at risk for a disease but do not develop it. MIMIC-IV is the pilot;
the same code is meant to be adapted to the HERO cohort.

## Part 1. Repository

| Folder | Content |
| --- | --- |
| `config/` | MIMIC column inventory (`mimic_columns.csv`), checks against the files, masking settings (`masking_configs.yaml`) |
| `trajectory/` | events, episodes and points from the MIMIC tables |
| `vocabulary/` | token vocabulary and value statistics |
| `quantiles/` | quantile cut points and bins Q1–Q10 |
| `thinning/` | lowest and highest value per 4-hour block |
| `bert/` | master token table, BERT data, model and training (D1, D2, D3); its own `README.md` is reproduced as Part 2 below |

**Data.** No MIMIC data is stored in this repository. MIMIC-IV 3.1 and MIMIC-IV-ED 2.2 are credentialed datasets
from PhysioNet: access requires the PhysioNet credentialing and data use agreement. Paths to the data
(`D:\MIMIC\...`) are set in the scripts and in `bert/trajectory_columns.yaml`. Notebook outputs are stripped
before each commit (`nbstripout`), and `.gitignore` excludes data files, checkpoints and logs.

**Requirements.** Python 3.11, Polars, NumPy, PyYAML, PyTorch 2.1 or later, TensorBoard.

## Part 2. BERT pre-training: data, trajectories, masking and model

What has been built so far, how it fits together, and the commands to run it.
Last updated 2026-11-23. Online version: [MIMIC BERT pre-training (claude.ai document)](https://claude.ai/artifact/61YbsyYkvjQY2u3otTuWHA).

### Overview

Three BERT encoders are pre-trained on the same MIMIC-IV patient trajectories. They differ only in what each event carries, and therefore in what can be masked and predicted. The trained encoder is the starting point of the JEPA stage.

| Model | Each event carries | Predicted when masked |
| --- | --- | --- |
| D1 | concept | concept |
| D2 | concept, quantile bin (Q1–Q10), unit | concept, quantile, unit |
| D3 | concept, quantile bin, `[VALUE]` with the scaled number, unit | concept, quantile and value, unit |

The pipeline has three stages: MIMIC tables to thinned and binned events (steps 1–7), events to one master token table (step 8), and the master table to arrays and training (steps 9–10). Code lives in `C:\Users\yy\Desktop\codes_PhD\jepa_mimic`, data in `D:\MIMIC\trajectory_out`, trained models in `D:\MIMIC\bert_model`.

### Data preparation (steps 1–7)

Each step reads the output of the step before it. Paths to code are relative to `jepa_mimic`; outputs are in `D:\MIMIC\trajectory_out` unless stated.

| Step | Code | What it does | Output |
| --- | --- | --- | --- |
| 1 | `config\mimic_columns.csv`, `config\check_headers.py` | Lists every MIMIC column with its type and a use switch; checks the headers against the files | the column inventory |
| 2 | `trajectory\build_trajectories_lowmem.py` | One event per row: age-based time (`days_since_birth`, `time_of_day`), `time_basis`, ICD version, drug name, episodes | `events.parquet`, `episodes.parquet` |
| 3 | `trajectory\build_points.py` | One point per episode, undated events dropped | `points.parquet`, `events_clean.parquet` |
| 4 | `vocabulary\build_vocabulary.py` | Tokens = source, ontology_version, label, code; special tokens, Q1–Q10 + unknown, gender, units; median and IQR per concept and unit (all patients) | `vocabulary\vocab.parquet`, `value_stats.parquet` |
| 5 | `quantiles\build_cutpoints.py` | 9 cut points per concept and unit with at least 100 values, whole cohort | `quantiles\quantile_cutpoints.parquet` |
| 6 | `quantiles\assign_bins.py` | Bin Q1–Q10 (or unknown) for every numeric event, plus `label`, `ontology_version`, `event_row` | `events_binned.parquet` |
| 7 | `thinning\thin_events_4h.py` | Keeps the lowest and highest value per patient, 4-hour block and concept-unit pair | `events_thinned_4h.parquet` (624,252,246 events) |

Units are kept as written in MIMIC, with no conversion. Quantile bins and value scaling are computed per concept and unit, so Q8 means the same rank in any unit.

### Trajectory construction (step 8)

One master table holds every token of every patient, at the D3 level; D1 and D2 are obtained by dropping rows. Code: `bert\build_bert_master.ipynb` with `bert\trajectory_columns.yaml`. Output: `bert\master\master_n128_bXXX.parquet`, `split.parquet` (60/20/20 by patient, seed 42), `patient_index_n128.parquet`.

Rules:

- A sequence is the gender token, then the events in time order. There is no `[SEP]`: visits are shown by the time channels, and later the JEPA parts by a segment channel. `[SEP]` stays a reserved id.
- An event gives up to four tokens, always in this order: concept, quantile, `[VALUE]`, unit. Quantile and `[VALUE]` exist only for numeric events; the unit only for numeric events that have one.
- Order: `age_days`, then `seconds_in_day`, then `event_row`. Date-only events (ICD procedures, OMR, death) go last in their day.
- Undated events and patients without events are removed.

| Column | Role | Content |
| --- | --- | --- |
| `token_id` | input | full vocabulary id |
| `source` | input | event table; `patients` for gender |
| `value` | input (D3) | raw number, on the `[VALUE]` row only |
| `age_days` | input | days from birth to the event; null for gender |
| `seconds_in_day` | input | 0–86,399; null for gender and date-only events |
| `type` | masking | gender, code, quantile, value, unit |
| `piece` | masking | event number per patient, 0 = gender; shared by all tokens of an event |
| `label` | reading | readable name of the event |
| `unit`, `episode_id`, `setting`, `time_basis`, `event_row`, `split`, `pos` | loader and checks | value scaling, window ends, balance check, link back to the event |

Example, a creatinine of 1.4 mg/dL: `creatinine | Q8 | [VALUE] (1.4) | mg/dL`, four rows sharing one `piece`.

### Vocabulary cut-off

The master table stores full vocabulary ids, so the vocabulary size is chosen later without a rebuild. A concept is kept for the model if at least `min_patients` patients have it (counted on all patients); rarer concepts become `[UNK]`. Special tokens, quantile bins, gender, death and units are always kept.

Kept tokens are renumbered without gaps in the order of the full vocabulary, so special tokens keep their ids (`[UNK]` = 1). The result is a remap `token_id → model_id`, written as `vocab_remap_min<N>.parquet`. Choose N with section 2 of the master notebook, which prints the vocabulary size for several cut-offs, then pass it to training with `--min-patients N`.

### Data for the model (step 9)

The master table is converted once per (model, cut-off) into flat memory-mapped arrays, built automatically by the training script the first time. Code: `bert\bert_data.py` (`prepare_arrays`). Output: `D:\MIMIC\trajectory_out\bert\arrays\<model>_min<N>\{train,val,test}`.

| Array | Content |
| --- | --- |
| `tokens` | model id of each token |
| `source`, `age`, `sec` | input channels; -1 = no time |
| `value` | scaled value on `[VALUE]` tokens: clip((v − median) / IQR, −5, 5), per concept and unit |
| `ttype`, `piece`, `kind` | token type, event number, kind of event (no number / unknown bin / numeric) |
| `patient_ptr`, `day_starts`, `day_ptr` | where each patient and each day with data starts |

Sampling follows the balanced recipe:

- **One segment per patient per epoch.** A patient that fits in the segment cap is shown whole. A longer one starts at the first event of a day drawn at random, so every day counts the same, and is cut at the cap without splitting an event.
- **Segment caps:** D1 512, D2 800, D3 1,000 tokens, gender included. Every segment starts with the gender token.
- **Packing:** segments are placed one after another in rows of 1,024 positions, the same for all models, so the compute per batch is equal.
- **Isolation:** each segment has its own number; a token attends only to its own segment, and positions restart at 0. A packed patient is computed exactly as if alone, so the encoder transfers unchanged to JEPA.
- **Validation:** a fixed set of validation batches (seed 12345) is used at every evaluation.

### Masking strategy

Masking works on events (pieces), and every masked part is predicted. Settings: `config\masking_configs.yaml`; code: `Masker` in `bert\bert_data.py`.

1. **Choose events.** 15% of the events of each segment, in runs of consecutive events (geometric length, mean 3, at most 10). The gender is never masked; `source` and the time channels are never masked.
2. **Choose parts inside each event.**
   - Event without a number, or with an `unknown` bin: the concept (and its unit) is masked.
   - Numeric event (D2, D3): three parts — concept, magnitude, unit. Each is masked independently with probability 0.5, at least one. Magnitude = the quantile, plus `[VALUE]` and its number in D3: they are masked together, because the bin can be read off the value.
   - D1: an event has one token, so its concept is masked.
3. **Replace (BERT 80/10/10).** 80% `[MASK]`, 10% a random token of the same type, 10% unchanged. The loss is computed on all three cases. A masked `[VALUE]` also hides its number.

| Masked part | Random replacement drawn from |
| --- | --- |
| concept | other medical concepts of the model vocabulary |
| quantile | Q1–Q10 |
| unit | other units |
| value (D3) | the scaled value of another numeric event in the same segment |

Example, D2, creatinine Q8 mg/dL with concept and unit drawn: `[MASK] Q8 [MASK]` → predict creatinine and mg/dL. With only the magnitude drawn: `creatinine [MASK] mg/dL` → predict Q8.

### Loss

The loss has up to five parts. Each is averaged within each patient segment first, then over segments, so a sparse patient counts as much as a dense ICU one. Each is divided by its baseline and weighted:

```math
L = \sum_k w_k \, \frac{\bar\ell_k}{b_k}
```

- $\bar\ell_k$: mean loss of part $k$; $w_k$: weight in `loss_weights` (1 by default)
- $b_k$: baseline = loss of the frequency guess, which predicts each answer with its frequency and ignores the context. It is measured before training on 50 validation batches, with the same averaging.
- A ratio $\bar\ell_k / b_k$ of 1 means nothing was learned from the context; 0 means perfect.

| Part | Targets | Loss | Metrics logged |
| --- | --- | --- | --- |
| `concept_event` | concepts of events without a number | cross-entropy | accuracy, top-10 |
| `concept_measure` | concepts of numeric events | cross-entropy | accuracy, top-10 |
| `quantile` (D2, D3) | masked bins | soft cross-entropy: 0.8 on the true bin, 0.1 on each neighbour | accuracy, within ±1 bin |
| `unit` (D2, D3) | masked units | cross-entropy | accuracy |
| `value` (D3) | masked scaled values | Huber (delta 1) | mean absolute error |

Splitting the concept loss in two gives diagnoses and drugs the same share of the concept signal as labs and vitals, whatever their proportion in the data (about 48% of events are numeric after thinning). Bins are deciles per concept and unit, so the concept alone says nothing about the bin: quantile accuracy above 10% comes from the context.

### Model

A ModernBERT-style encoder: 256 dimensions, 4 layers, 4 heads, pre-norm blocks without biases, GeGLU feed-forward (width 704), rotary position embeddings, and no learned position table. Code: `bert\bert_model.py` (`TrajectoryBert`).

Input of each position, summed then normalised:

```math
x_i = E_{token}(id_i) + E_{source}(s_i) + A(age\_days_i) + T(seconds\_in\_day_i) + \mathbb{1}[value\ visible]\, V(\tilde v_i)
```

- A: sine and cosine of the age at 16 periods from 1 day to 100 years, then a linear layer. T: sine and cosine of the time of day at harmonics 1–8 of 24 h, then a linear layer. Missing time uses a learned "no time" vector.
- V (D3 only): a small network on the scaled value, at `[VALUE]` positions whose number is visible.
- RoPE: positions restart at 0 in each segment, and handle the longer JEPA inputs later without a fixed table.
- Attention mask: block-diagonal per segment, so packed patients never see each other.

Heads: a shared transform (linear, GELU, layer norm), then one output per target: concept (model vocabulary), quantile (10), unit (number of units), value (1). Each head is evaluated only where its target was masked. Training: AdamW (lr 3e-4, weight decay 0.01, betas 0.9/0.98), 2,000 warm-up steps, cosine decay to 3e-5, gradient clipping at 1, bf16 on a GPU that supports it.

### How to run (step 10)

Run from the `bert` folder in PowerShell. The first run of a (model, cut-off) builds its arrays, then trains.

| Argument | Default | Meaning |
| --- | --- | --- |
| `--model` | required | D1, D2 or D3 |
| `--masking-config` | `config\masking_configs.yaml` | masking, loss, packing settings |
| `--min-patients` | `min_patients` in `trajectory_columns.yaml` (50) | vocabulary cut-off |
| `--out-dir` | `D:\MIMIC\bert_model` | where runs are saved |
| `--run-name` | `<model>_min<N>_<date>` | run folder |
| `--resume` | off | continue the run `--run-name` from `last.pt` |
| `--prepare-only` | off | build the arrays and stop |
| `--batch-size` | 32 | packed rows per batch |
| `--max-steps`, `--eval-every`, `--patience` | 150,000; 1,000; 8 | stop at max steps or after 8 evaluations without a 0.2% gain |
| `--num-workers` | 0 on Windows | processes preparing batches; 0 = main process. Default 0 on Windows (2 elsewhere): workers exhausted the paging file there (error 1455) |

Each run folder holds `best.pt` (lowest validation loss), `last.pt`, `config.json` (arguments, masking settings, baselines) and `tensorboard\`. TensorBoard shows `train/` and `val/` curves: total loss, each part, the ratio to its baseline, accuracies, and the share of masked events that are numeric.

Requirements: PyTorch 2.1 or later, `pip install tensorboard`.

#### Code

| File | Content |
| --- | --- |
| `bert\build_bert_master.ipynb`, `bert\trajectory_columns.yaml` | master table, vocabulary cut-off, model views |
| `bert\bert_data.py` | remap, arrays, segments, masking, packing |
| `bert\bert_model.py` | encoder with RoPE and heads |
| `bert\train_bert.py` | arguments, baselines, training loop, TensorBoard, checkpoints |
| `config\masking_configs.yaml` | masking and loss settings |

### Results: D1, D2 and D3 (MIMIC-IV, November 2026)

All three runs use 256 dimensions, 4 layers, 4 heads, batch 32, `min_patients` 50 (21,440 tokens in the model vocabulary, 114 units). Values below are on the validation set, at the best checkpoint of each run. Runs: `D:\MIMIC\bert_model\D1_min50_20261120_xxxx`, `D2_min50`, `D3_min50`.

| Metric | What it measures | D1 | D2 | D3 | Blind guess |
| --- | --- | --- | --- | --- | --- |
| Context length (segment cap) | Longest stretch of one patient shown at once, in tokens, gender included | 512 | 800 | 1,000 | |
| Events per full window (estimate) | How many events fit in that stretch: 1 token per event in D1, about 1.9 in D2, about 2.4 in D3 | about 511 | about 420 | about 415 | |
| Row length (packing) | Positions per packed input row, shared by several patients; sets the compute per step | 1,024 | 1,024 | 1,024 | |
| Steps trained | Optimisation steps up to the best checkpoint | 136k | 128k | 104k | |
| Training time | Wall-clock hours of training | 13.1 h | 9.6 h | 7.5 h | |
| concept_event accuracy | Share of masked diagnoses, drugs, procedures and deaths whose concept was guessed exactly (top-1). The fair comparison: these events are masked the same way in all three models | **65.2%** | 64.8% | 62.3% | |
| concept_event top-10 | Share of those masked events whose true concept is among the model's 10 most likely answers | **83.7%** | 81.6% | 79.5% | |
| concept_event ratio | Loss divided by the loss of the frequency guess (always answering by how common each concept is). 1 = nothing learned from the context, 0 = perfect | **0.27** | 0.32 | 0.35 | 1 |
| concept_measure accuracy | Share of masked labs and vital signs whose concept was guessed exactly. Not comparable across models: in D2 and D3 the bin or unit is sometimes still visible, which narrows the answer | 93.1% | **97.2%** | 96.7% | |
| concept_measure ratio | Same ratio for labs and vital signs | 0.04 | 0.02 | 0.02 | 1 |
| quantile accuracy | Share of masked bins (Q1–Q10) guessed exactly, with the concept visible | – | 56.4% | **61.8%** | 10% |
| quantile within ±1 bin | Share of masked bins guessed exactly or one bin off (Q7 or Q9 for a true Q8) | – | 78.9% | **80.7%** | about 28% |
| unit accuracy | Share of masked units guessed exactly; nearly fixed once the concept is known | – | 98.7% | 98.6% | |
| value MAE | Mean absolute error of the predicted number, in IQR units of that concept and unit (0.24 = about a quarter of the typical spread) | – | – | 0.24 | ratio 0.32 |

What each model is:

| Model | Specificity | Strength | Cost |
| --- | --- | --- | --- |
| D1 | concept only, 1 token per event | best map of which diagnoses, drugs and procedures go together | blind to values |
| D2 | concept + quantile bin + unit | knows how abnormal a measurement is (bin) | about 18% fewer events per window; capacity shared by 4 tasks |
| D3 | concept + quantile bin + `[VALUE]` + unit | also sees exact values, so it places other values best | about 18% fewer events per window; capacity shared by 5 tasks; about 3 points less on diagnoses than D1 |

**Observations**

- All three learn far beyond the blind guess, and none overfits: validation loss stays slightly below training loss (dropout is on in training only).
- Diagnoses and drugs: D1 > D2 > D3. Values of labs and vital signs: D3 > D2.
- Units (about 98.6%) and lab and vital-sign concepts (93–97%) are nearly solved.
- All three were still improving slowly when they ended (about 0.5 accuracy points per 20,000 steps); none triggered early stopping.

**Interpretation**

- Adding values costs context (about 18% fewer events per window in D2 and D3) and splits a small model's capacity across 4–5 tasks instead of 2. This explains D1's small lead on diagnoses.
- Exact numbers (D3) help predict other values. Part of this may be persistence (the same lab a few hours earlier), not physiology: to be checked against a "same bin as the previous measurement" baseline.
- The unit task teaches almost nothing.

**Against the blind guess**

- The context removes 65–73% of the uncertainty about diagnoses and drugs (ratio 0.27 in D1, 0.35 in D3).
- Labs and vitals are almost fully predictable from the context (ratio 0.02–0.04).
- The quantile is placed 6 times better than chance (62% against 10%), and two thirds of the value error is removed (ratio 0.32).

**Ready for JEPA**

- As a starting point, yes: stable training, far above the blind guess, no overfitting.
- To choose which encoder, not yet: masked-token accuracy does not show which gives the best neighbourhoods for resilience. That is the gate of the training plan (neighbour AUROC against bag-of-codes + SVD).
- Practical path: start JEPA from D3 (most information, small gap on diagnoses), keep D1 as the comparison, and run the gate on both.

**Worth trying**

1. A persistence baseline for the quantile (bin of the previous measurement of the same concept in the window).
2. Give `concept_event` back its share in D2 and D3: drop the unit task, or `unit: 0.25` and `concept_event: 2` in `loss_weights`.
3. Equal history per window: caps of about 970 (D2) and 1,200 (D3) tokens, so all models see about 500 events.
4. A larger model (512 dimensions, 6–8 layers): no overfitting so far, and D3 shares its capacity across 5 tasks.
5. Speed: better packing (rows only 75–83% filled) and faster masking (the data pipeline, not the GPU, limits the speed).

### Command reference

All commands in PowerShell, from `C:\Users\yy\Desktop\codes_PhD\jepa_mimic`, in pipeline order. Each step needs the output of the step before it.

**1. Check the configuration**

```
cd C:\Users\yy\Desktop\codes_PhD\jepa_mimic\config
python check_config.py
python check_headers.py
```

**2. Events, episodes and points**

```
cd ..\trajectory
python build_trajectories_lowmem.py                       # everything
python build_trajectories_lowmem.py --limit 2000          # quick test on 2,000 patients
python build_trajectories_lowmem.py --tables labevents    # one table only
python build_points.py
```

**3. Vocabulary and value statistics**

```
cd ..\vocabulary
python build_vocabulary.py        # close vocab.csv in Excel first
```

**4. Cut points and bins**

```
cd ..\quantiles
python build_cutpoints.py
python assign_bins.py
```

**5. Thinning**

```
cd ..\thinning
python thin_events_4h.py
```

**6. Master table (notebook)**

Open `bert\build_bert_master.ipynb` and run sections 1 to 9. Settings are in `bert\trajectory_columns.yaml` (`n_buckets: 128`, `min_patients: 50`). Section 2 prints the vocabulary size for several cut-offs. If section 8 crashes, rerun the cell: buckets already written are skipped.

**7. BERT training**

```
cd ..\bert

# start a run (the first run of each model also builds its arrays)
python train_bert.py --model D1 --min-patients 50 --run-name D1_min50_20261120_xxxx
python train_bert.py --model D2 --min-patients 50 --run-name D2_min50
python train_bert.py --model D3 --min-patients 50 --run-name D3_min50

# resume a stopped run: same command + --resume
python train_bert.py --model D1 --min-patients 50 --run-name D1_min50_20261120_xxxx --resume
python train_bert.py --model D2 --min-patients 50 --run-name D2_min50 --resume

# only build the arrays, without training
python train_bert.py --model D3 --min-patients 50 --prepare-only
```

- Stop a run with Ctrl+C right after an `eval step …` line, so `last.pt` is fresh.
- One training at a time: each run fills the 12 GB GPU.
- Useful options: `--batch-size 24` (less GPU memory), `--num-workers 0` (default on Windows), `--max-steps`, `--eval-every`, `--patience`.

**8. TensorBoard**

```
pip install tensorboard                                     # once
tensorboard --logdir D:\MIMIC\bert_model                    # in a second window
python -m tensorboard.main --logdir D:\MIMIC\bert_model     # if "tensorboard" is not recognised
```

Then open `http://localhost:6006` in a browser; close TensorBoard with Ctrl+C in its window.
