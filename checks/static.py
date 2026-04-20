"""Static checks — operate on local CSV shot files, no API calls."""

from __future__ import annotations

import csv
import math
import os
import statistics
from dataclasses import dataclass, field
from typing import Optional


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
INFO = "INFO"


@dataclass
class CheckResult:
    name: str
    level: str
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class ShotFileSummary:
    path: str
    n_rows: int
    header: list
    timestamp_col: str
    feature_cols: list
    timestamp_deltas_median: Optional[float]
    timestamp_deltas_max: Optional[float]
    monotonic: bool
    missing_per_col: dict
    variance_per_col: dict
    min_per_col: dict
    max_per_col: dict
    non_numeric_cols: list


def _parse_row_numeric(row: dict, feature_cols: list):
    """Return dict of floats; missing/non-numeric become None."""
    out = {}
    for c in feature_cols:
        v = row.get(c, "")
        if v is None or v == "":
            out[c] = None
            continue
        try:
            out[c] = float(v)
        except (ValueError, TypeError):
            out[c] = None
    return out


def summarize_shot_file(path: str, timestamp_col: str = "timestamp") -> ShotFileSummary:
    """Single pass over CSV collecting stats needed for all static checks."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if not header:
            raise ValueError(f"{path}: empty or missing header")

        has_ts = timestamp_col in header
        feature_cols = [c for c in header if c != timestamp_col]

        n_rows = 0
        prev_ts: Optional[float] = None
        monotonic = True
        deltas: list = []

        sums = {c: 0.0 for c in feature_cols}
        sumsq = {c: 0.0 for c in feature_cols}
        counts = {c: 0 for c in feature_cols}
        missing = {c: 0 for c in feature_cols}
        mins = {c: math.inf for c in feature_cols}
        maxs = {c: -math.inf for c in feature_cols}
        non_numeric = set()

        for row in reader:
            n_rows += 1

            if has_ts:
                try:
                    ts = float(row[timestamp_col])
                    if prev_ts is not None:
                        d = ts - prev_ts
                        if d < 0:
                            monotonic = False
                        deltas.append(d)
                    prev_ts = ts
                except (ValueError, TypeError):
                    monotonic = False

            vals = _parse_row_numeric(row, feature_cols)
            for c, v in vals.items():
                if v is None:
                    missing[c] += 1
                    if row.get(c) not in ("", None):
                        non_numeric.add(c)
                else:
                    sums[c] += v
                    sumsq[c] += v * v
                    counts[c] += 1
                    if v < mins[c]:
                        mins[c] = v
                    if v > maxs[c]:
                        maxs[c] = v

    variance = {}
    for c in feature_cols:
        n = counts[c]
        if n < 2:
            variance[c] = 0.0
            continue
        mean = sums[c] / n
        variance[c] = max(0.0, sumsq[c] / n - mean * mean)

    ts_med = statistics.median(deltas) if deltas else None
    ts_max = max(deltas) if deltas else None

    return ShotFileSummary(
        path=path,
        n_rows=n_rows,
        header=header,
        timestamp_col=timestamp_col if has_ts else "",
        feature_cols=feature_cols,
        timestamp_deltas_median=ts_med,
        timestamp_deltas_max=ts_max,
        monotonic=monotonic,
        missing_per_col=missing,
        variance_per_col=variance,
        min_per_col={c: (mins[c] if mins[c] != math.inf else None) for c in feature_cols},
        max_per_col={c: (maxs[c] if maxs[c] != -math.inf else None) for c in feature_cols},
        non_numeric_cols=sorted(non_numeric),
    )


def check_schema(s: ShotFileSummary) -> CheckResult:
    if not s.timestamp_col:
        return CheckResult(
            "schema", FAIL,
            f"{s.path}: timestamp column not found. Header={s.header[:6]}...",
        )
    if s.non_numeric_cols:
        return CheckResult(
            "schema", FAIL,
            f"{s.path}: non-numeric values in {len(s.non_numeric_cols)} column(s): "
            f"{s.non_numeric_cols[:5]}",
        )
    return CheckResult("schema", PASS, f"{s.path}: {len(s.feature_cols)} numeric feature columns + timestamp")


def check_timestamp(s: ShotFileSummary) -> CheckResult:
    if not s.timestamp_col:
        return CheckResult("timestamp", FAIL, f"{s.path}: no timestamp column")
    if not s.monotonic:
        return CheckResult("timestamp", FAIL, f"{s.path}: timestamp not monotonically increasing")
    if s.timestamp_deltas_median is None:
        return CheckResult("timestamp", WARN, f"{s.path}: could not compute timestamp deltas")
    med = s.timestamp_deltas_median
    mx = s.timestamp_deltas_max or 0
    if mx > 5 * med and med > 0:
        return CheckResult(
            "timestamp", WARN,
            f"{s.path}: large timestamp gap — median delta {med}, max {mx}. "
            f"Windows may cross segment boundaries.",
            details={"median_delta": med, "max_delta": mx},
        )
    return CheckResult(
        "timestamp", PASS,
        f"{s.path}: monotonic, median delta {med}, no large gaps",
        details={"median_delta": med, "max_delta": mx},
    )


def check_missing(s: ShotFileSummary) -> CheckResult:
    offenders = {c: n for c, n in s.missing_per_col.items() if n > 0}
    if not offenders:
        return CheckResult("missing_values", PASS, f"{s.path}: no missing values")
    worst = max(offenders.values())
    frac = worst / s.n_rows if s.n_rows else 0
    if frac > 0.05:
        return CheckResult(
            "missing_values", WARN,
            f"{s.path}: {len(offenders)} col(s) with missing values, worst {frac:.1%} "
            f"— forward-fill or drop column; do NOT drop rows",
        )
    return CheckResult(
        "missing_values", WARN,
        f"{s.path}: {len(offenders)} col(s) with missing values, worst {frac:.2%}",
    )


def check_constant_cols(s: ShotFileSummary) -> CheckResult:
    dead = [c for c, v in s.variance_per_col.items() if v < 1e-12]
    if not dead:
        return CheckResult("constant_columns", PASS, f"{s.path}: all feature columns have variance")
    return CheckResult(
        "constant_columns", WARN,
        f"{s.path}: {len(dead)} constant/near-constant column(s) — drop before upload: {dead[:5]}",
        details={"columns": dead},
    )


def check_feature_scale(s: ShotFileSummary) -> CheckResult:
    ranges = []
    for c in s.feature_cols:
        lo, hi = s.min_per_col[c], s.max_per_col[c]
        if lo is None or hi is None:
            continue
        r = abs(hi - lo)
        if r > 0:
            ranges.append((c, r))
    if len(ranges) < 2:
        return CheckResult("feature_scale", INFO, f"{s.path}: insufficient data to compare scales")
    ranges.sort(key=lambda x: x[1])
    smallest_col, smallest_range = ranges[0]
    largest_col, largest_range = ranges[-1]
    span = math.log10(largest_range) - math.log10(smallest_range)
    if span > 3:
        return CheckResult(
            "feature_scale", WARN,
            f"{s.path}: largest-range column '{largest_col}' spans {largest_range:,.3g}, "
            f"smallest '{smallest_col}' spans {smallest_range:,.3g} — a {span:.1f}-decade gap. "
            f"With metric=euclidean, '{largest_col}' will dominate and small-scale columns "
            f"will be ignored, often dropping accuracy 10-20pp. "
            f"Fix: z-score each column ((x - mean) / std) before uploading, "
            f"or try --metric cosine (scale-invariant).",
            details={
                "orders_of_magnitude": span,
                "largest": {"column": largest_col, "range": largest_range},
                "smallest": {"column": smallest_col, "range": smallest_range},
            },
        )
    return CheckResult(
        "feature_scale", PASS,
        f"{s.path}: feature scales within {span:.1f} orders of magnitude",
    )


def check_nshot_support(s: ShotFileSummary, window_size: int, floor_rows: int = 500) -> CheckResult:
    usable = max(0, s.n_rows - window_size + 1)
    if s.n_rows < floor_rows:
        return CheckResult(
            "nshot_support", FAIL,
            f"{s.path}: {s.n_rows} rows < {floor_rows} floor — too few contiguous shots. "
            f"Add more contiguous data from other simulations/shifts/batches.",
            details={"rows": s.n_rows, "usable_embeddings": usable},
        )
    if s.n_rows < 2000:
        return CheckResult(
            "nshot_support", WARN,
            f"{s.path}: {s.n_rows} rows (below recommended 2000) — "
            f"{usable} usable embeddings at window={window_size}",
            details={"rows": s.n_rows, "usable_embeddings": usable},
        )
    return CheckResult(
        "nshot_support", PASS,
        f"{s.path}: {s.n_rows} rows, {usable} usable embeddings at window={window_size}",
        details={"rows": s.n_rows, "usable_embeddings": usable},
    )


def check_schema_match(normal: ShotFileSummary, fault: ShotFileSummary) -> CheckResult:
    if normal.header != fault.header:
        only_n = set(normal.header) - set(fault.header)
        only_f = set(fault.header) - set(normal.header)
        return CheckResult(
            "schema_match", FAIL,
            f"Shot files have different columns. Only-in-normal={sorted(only_n)}, "
            f"only-in-fault={sorted(only_f)}",
        )
    return CheckResult("schema_match", PASS, f"Both shot files share {len(normal.header)} columns")


def check_class_balance(normal: ShotFileSummary, fault: ShotFileSummary) -> CheckResult:
    total = normal.n_rows + fault.n_rows
    if total == 0:
        return CheckResult("class_balance", FAIL, "Both shot files are empty")
    n_frac = normal.n_rows / total
    f_frac = fault.n_rows / total
    majority = max(n_frac, f_frac)
    msg = f"normal={n_frac:.1%} ({normal.n_rows}), fault={f_frac:.1%} ({fault.n_rows})"
    if majority > 0.85:
        return CheckResult(
            "class_balance", WARN,
            f"{msg}. Severe imbalance (>85%) — model will tend to predict the majority "
            f"class; any accuracy below {majority:.1%} is worse than always-predict-majority.",
            details={"majority_baseline": majority},
        )
    if majority > 0.70:
        return CheckResult(
            "class_balance", WARN,
            f"{msg}. Moderate imbalance — compare pilot accuracy against "
            f"always-predict-majority baseline {majority:.1%}.",
            details={"majority_baseline": majority},
        )
    return CheckResult("class_balance", PASS, msg, details={"majority_baseline": majority})


UNIT_TO_SECONDS = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0}


def _humanize_seconds(total_s: float) -> str:
    if total_s < 60:
        return f"{total_s:.1f} seconds"
    if total_s < 3600:
        return f"{total_s / 60:.1f} minutes"
    if total_s < 86400:
        return f"{total_s / 3600:.1f} hours"
    return f"{total_s / 86400:.1f} days"


def check_window_vs_sampling(
    normal: ShotFileSummary,
    fault: ShotFileSummary,
    window_size: int,
    timestamp_unit: str = "auto",
) -> CheckResult:
    deltas = [
        x for x in (normal.timestamp_deltas_median, fault.timestamp_deltas_median)
        if x is not None
    ]
    if not deltas:
        return CheckResult("window_vs_sampling", INFO, "No timestamp deltas available")
    delta = statistics.median(deltas)
    raw_span = window_size * delta

    if timestamp_unit == "auto":
        msg = (
            f"Each window spans {window_size} rows. "
            f"If rows are 1-minute samples, that's {window_size} min; "
            f"if 3-minute, that's {window_size * 3 / 60:.1f} hours; "
            f"if 1-second, that's {window_size} s. "
            f"Pass --timestamp-unit {{seconds,minutes,hours}} for a concrete number. "
            f"Sanity-check this covers at least one typical fault/event duration for your process."
        )
        details = {"window_size": window_size, "median_sample_delta": delta, "timestamp_unit": "auto"}
    else:
        total_seconds = raw_span * UNIT_TO_SECONDS[timestamp_unit]
        human = _humanize_seconds(total_seconds)
        msg = (
            f"Each window spans {window_size} rows × {delta} {timestamp_unit}/row "
            f"= {human} of process data. "
            f"Sanity-check this covers at least one typical fault/event duration for your process."
        )
        details = {
            "window_size": window_size,
            "median_sample_delta": delta,
            "timestamp_unit": timestamp_unit,
            "window_span_seconds": total_seconds,
        }
    return CheckResult("window_vs_sampling", INFO, msg, details=details)


def check_expected_accuracy_prior(
    normal: ShotFileSummary, fault: ShotFileSummary
) -> CheckResult:
    return CheckResult(
        "accuracy_prior", INFO,
        "Base-model accuracy on time-series data varies widely (coin-flip to ~0.80) — "
        "it depends on how distinct your classes look in the learned embedding space. "
        "Run with --pilot for a real number specific to your data.",
    )


def run_all_static_checks(
    normal_path: str,
    fault_path: str,
    window_size: int,
    timestamp_col: str = "timestamp",
    nshot_floor: int = 500,
    timestamp_unit: str = "auto",
) -> tuple[list[CheckResult], ShotFileSummary, ShotFileSummary]:
    normal = summarize_shot_file(normal_path, timestamp_col)
    fault = summarize_shot_file(fault_path, timestamp_col)

    results: list[CheckResult] = []
    for summary in (normal, fault):
        results.append(check_schema(summary))
        results.append(check_timestamp(summary))
        results.append(check_missing(summary))
        results.append(check_constant_cols(summary))
        results.append(check_feature_scale(summary))
        results.append(check_nshot_support(summary, window_size, nshot_floor))

    results.append(check_schema_match(normal, fault))
    results.append(check_class_balance(normal, fault))
    results.append(check_window_vs_sampling(normal, fault, window_size, timestamp_unit))
    results.append(check_expected_accuracy_prior(normal, fault))

    return results, normal, fault
