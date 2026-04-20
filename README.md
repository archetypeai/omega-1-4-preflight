# omega-1-4-preflight

Predict whether a dataset is likely to reach **~70% accuracy** on Archetype AI's `omega_1_4_base` model via the `machine-state-job-pipeline` — **before** committing to a full batch run.

Lessons baked in from six sibling batch-example repos: [higgs](https://github.com/archetypeai/archetypeai-batch-examples-higgs), [tep](https://github.com/archetypeai/archetypeai-batch-examples-tep), [swat](https://github.com/archetypeai/archetypeai-batch-examples-swat), [pump-sensor](https://github.com/archetypeai/archetypeai-batch-examples-pump-sensor), [3w](https://github.com/archetypeai/archetypeai-batch-examples-3w), [nasa-bearing](https://github.com/archetypeai/archetypeai-batch-examples-nasa-bearing).

> **Note:** `omega_1_4_base` is currently only available in dev. Point `ATAI_API_ENDPOINT` at the dev endpoint (see `.env.example`).

## What it does

- **Static checks** (fast, local, no API): schema, timestamps, contiguity, class balance, majority-baseline, constant columns, missing values, feature-scale heterogeneity, window-vs-sampling, n-shot support.
- **Pilot run** (optional, requires API access): holds out ~20% of each shot file as a labeled pilot slice, uploads the remaining shots + pilot as a tiny `machine-state-job-pipeline` job, scores predictions vs. known labels, and reports a verdict against the 70% bar and the always-predict-majority baseline.

v1 scope: **binary time-series** (normal vs fault). Multi-class is a later flag.

## Setup

```bash
# Clone
git clone https://github.com/archetypeai/omega-1-4-preflight.git
cd omega-1-4-preflight

# Configure credentials
cp .env.example .env
# Edit .env with your ATAI_API_KEY and ATAI_API_ENDPOINT

# Create a virtual environment
python3 -m venv myenv

# Activate it
source myenv/bin/activate

# Install Python dependencies
pip install requests

# Deactivate when done
deactivate
```

## Example data

Two ready-to-use shot files ship under `data/`:

| File | Rows | Class | Contents |
|---|---|---|---|
| `data/normal_shots.csv` | 2,000 | normal | Contiguous fault-free block (Tennessee Eastman Process) |
| `data/fault_shots.csv`  | 2,000 | fault  | Contiguous faulted block (Tennessee Eastman Process) |

Both are 53-column CSVs (timestamp + 52 process variables). You can run every example below against them out of the box.

### Data attribution

These shot files are contiguous 2,000-row blocks extracted from the **Tennessee Eastman Process (TEP)** simulation dataset via the preparation pipeline in the [TEP sibling repo](https://github.com/archetypeai/archetypeai-batch-examples-tep). Provenance:

- **Original process benchmark:** Downs, J. J., & Vogel, E. F. (1993). *A plant-wide industrial process control problem.* Computers & Chemical Engineering, 17(3), 245–255. Eastman Chemical Company.
- **Extended simulation dataset:** Rieth, C. A., Amsel, B. D., Tran, R., & Cook, M. B. (2017). *Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation.* Harvard Dataverse, V1. [doi:10.7910/DVN/6C3JR1](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1)
- **CSV version:** [kaggle.com/datasets/afrniomelo/tep-csv](https://www.kaggle.com/datasets/afrniomelo/tep-csv) (afrniomelo, Kaggle).
- **Shot extraction:** `1_prepare_data/` in the [TEP sibling repo](https://github.com/archetypeai/archetypeai-batch-examples-tep#2-dataset) — see that README for the full 15.3M-row dataset description, fault taxonomy (21 types), and process-variable glossary.

Licensing follows the Harvard Dataverse terms of use for the source dataset.

## Usage

### Static checks only (no API calls)

```bash
python preflight.py \
  --shots-normal data/normal_shots.csv \
  --shots-fault  data/fault_shots.csv
```

### With pilot run

```bash
python preflight.py \
  --shots-normal data/normal_shots.csv \
  --shots-fault  data/fault_shots.csv \
  --pilot
```

### All flags

| Flag | Default | Meaning |
|---|---|---|
| `--shots-normal PATH` | required | CSV of contiguous normal-class rows |
| `--shots-fault PATH`  | required | CSV of contiguous fault-class rows |
| `--timestamp-column`  | `timestamp` | Timestamp column name |
| `--window-size N`     | 64 | Rows per inference window |
| `--n-neighbors K`     | 5  | kNN neighbors |
| `--metric M`          | `euclidean` | Distance metric (`euclidean`/`cosine`/`manhattan`) |
| `--weights W`         | `uniform` | kNN weighting (`uniform`/`distance`) |
| `--step-size N`       | 1 | Sliding step |
| `--nshot-floor N`     | 500 | Minimum usable rows/class |
| `--pilot`             | off | Run held-out pilot via batch API |
| `--pilot-size N`      | auto | Override auto split — rows/class for pilot |
| `--force`             | off | Run pilot even below floor (underpowered) |
| `--env PATH`          | `.env` | Env file path |

## Pilot split rule

From the default (held-out) mode:

| Rows per class | Split | Label |
|---|---|---|
| ≥ 1,000 | 80% shots / 20% pilot | high-confidence |
| 500–999 | 70% shots / 30% pilot (pilot ≥ 300) | low-confidence |
| < 500   | refused (unless `--force`) | — |

When refused, preflight prints three options:
1. `--pilot-size N --force` — override floors, run underpowered.
2. Supply your own labeled slice via `--pilot-set FILE` (planned).
3. Expand shot files with more contiguous data from other simulations / shifts / batches.

## Static checks (what's run)

| # | Check | Type | Fails when |
|---|---|---|---|
| 1 | `schema` | universal | timestamp column missing or non-numeric cells present |
| 2 | `timestamp` | time-series | non-monotonic or large gaps (>5× median delta) |
| 3 | `missing_values` | universal | any column has missing values (warn; strongly warn if >5%) |
| 4 | `constant_columns` | universal | any feature column has zero variance |
| 5 | `feature_scale` | universal | feature ranges span >3 orders of magnitude |
| 6 | `nshot_support` | labeled | fewer than `--nshot-floor` rows/class |
| 7 | `schema_match` | universal | shot files have different columns |
| 8 | `class_balance` | labeled | majority class > 70% (warn) or > 85% (strong warn) |
| 9 | `window_vs_sampling` | time-series | info-only — translates `window_size × median_delta` into physical time so you can sanity-check it covers a relevant event timescale |
| 10 | `accuracy_prior` | labeled | info-only — expected accuracy band and family-specific guidance |

## Pilot verdicts

| Verdict | Meaning |
|---|---|
| `PASS`  | Pilot accuracy ≥ 70% AND beats majority baseline by >5pp |
| `WARN`  | Pilot below 70% but above majority baseline — tuning may help |
| `FAIL`  | Pilot below majority baseline — base model not useful; consider fine-tuning |

## Pilot caveats

The held-out pilot measures **within-distribution** accuracy — the last ~20% of each shot file, drawn from the same simulations / shifts / runs / equipment as the shots themselves. This is an easier task than full-inference generalization, so a passing pilot is necessary but **not sufficient** evidence that the full run will clear 70%.

- **Observed on TEP:** pilot 0.784 vs. full-inference 0.506–0.537 on 15.3M rows from different simulations. A `PASS` verdict overestimated real-world accuracy by ~25pp.
- **Be skeptical when:** your production inference data comes from simulations, shifts, equipment, time periods, or environmental conditions not represented in the shot files. The pilot cannot detect distribution shift it never sees.
- **Escape hatch (planned `--pilot-set FILE`):** supply your own labeled slice drawn from the inference distribution; preflight scores against it directly and bypasses the held-out split.
- **Boundary artifact:** the pilot concatenates `normal` then `fault` into a single inference file, so ~`window_size` rows straddle the class boundary. The pipeline drops windows it cannot fully form, so the effective pilot size is slightly smaller than the requested split. Negligible for the overall signal, but explains why per-class `support` sums to less than the requested pilot size in the report.

Rule of thumb: treat a `PASS` at pilot as "worth running the full job," not as "the full job will clear 70%." Treat a `WARN` or `FAIL` at pilot as strong evidence the full run will not clear 70% without tuning or fine-tuning.

## Example output

```
================================================================================
 omega-1-4-preflight
================================================================================
  shots-normal: data/normal_shots.csv
  shots-fault:  data/fault_shots.csv
  window_size=64 n_neighbors=5 metric=euclidean weights=uniform

[A] Static checks
--------------------------------------------------------------------------------
  [PASS] schema               data/normal_shots.csv: 52 numeric feature columns + timestamp
  [PASS] timestamp            data/normal_shots.csv: monotonic, median delta 1.0, no large gaps
  [PASS] missing_values       data/normal_shots.csv: no missing values
  [PASS] constant_columns     data/normal_shots.csv: all feature columns have variance
  [WARN] feature_scale        data/normal_shots.csv: feature ranges span 3.7 orders of magnitude …
  [PASS] nshot_support        data/normal_shots.csv: 2000 rows, 1937 usable embeddings at window=64
  ...
  [PASS] class_balance        normal=50.0% (2000), fault=50.0% (2000)
  [INFO] window_vs_sampling   window_size=64 × 1.0-unit sample period = 64 timestamp-units …
  [INFO] accuracy_prior       Dataset family: binary time-series + contiguous shots …

  totals: PASS=12  WARN=2  FAIL=0  INFO=2
```

With `--pilot`, an additional pilot section reports:

```
================================================================================
 Pilot Result
================================================================================
  Verdict:          WARN (high-confidence)
  Reason:           Accuracy 0.612 below 70% bar (but above majority baseline 0.500). Tuning may close the gap.
  Accuracy:         0.612
  Macro F1:         0.581
  Majority baseline:0.500
  Job id:           job_...

  Class        Precision   Recall       F1    Support
  ----------------------------------------------------
  normal           0.598    0.634    0.615        400
  fault            0.627    0.591    0.608        400
```

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Static checks clean OR pilot verdict `PASS`/`WARN` |
| 1 | Static checks failed OR missing credentials |
| 2 | Pilot verdict `FAIL` |

## License

Apache 2.0
