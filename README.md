# omega-1-4-preflight

Predict whether a dataset is likely to reach **~70% accuracy** on Archetype AI's `omega_1_4_base` model via the `machine-state-job-pipeline` — **before** committing to a full batch run.

Lessons baked in from six sibling batch-example repos: [higgs](https://github.com/archetypeai/archetypeai-batch-examples-higgs), [tep](https://github.com/archetypeai/archetypeai-batch-examples-tep), [swat](https://github.com/archetypeai/archetypeai-batch-examples-swat), [pump-sensor](https://github.com/archetypeai/archetypeai-batch-examples-pump-sensor), [3w](https://github.com/archetypeai/archetypeai-batch-examples-3w), [nasa-bearing](https://github.com/archetypeai/archetypeai-batch-examples-nasa-bearing).

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
