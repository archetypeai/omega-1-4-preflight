#!/usr/bin/env python3
"""omega-1-4-preflight — predict whether a dataset is likely to reach ~70%
accuracy on Archetype AI's omega_1_4_base via the machine-state-job-pipeline,
before committing to a full batch run.

Usage:
    python preflight.py --shots-normal PATH --shots-fault PATH [--pilot] [options]
"""

import argparse
import os
import sys
import warnings

# Suppress urllib3 LibreSSL noise that appears on default macOS Python 3.9.
# Must run before `requests` (indirectly imported below) pulls in urllib3.
warnings.filterwarnings("ignore", module="urllib3")

from checks.pilot import ArchetypeClient, PilotConfig, run_pilot
from checks.static import FAIL, PASS, WARN, CheckResult, run_all_static_checks


COLOR = {
    PASS: "\033[32m",
    WARN: "\033[33m",
    FAIL: "\033[31m",
    "INFO": "\033[36m",
    "reset": "\033[0m",
}


def _load_env(env_path: str):
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)


def _parse_args():
    p = argparse.ArgumentParser(
        prog="preflight",
        description="Preflight a dataset against omega_1_4_base + machine-state-job-pipeline.",
    )
    p.add_argument("--shots-normal", required=True, help="CSV of contiguous normal-class rows")
    p.add_argument("--shots-fault", required=True, help="CSV of contiguous fault-class rows")
    p.add_argument("--timestamp-column", default="timestamp")
    p.add_argument(
        "--timestamp-unit",
        default="auto",
        choices=["auto", "seconds", "minutes", "hours"],
        help="Physical unit of the timestamp column (e.g. 'minutes' if each integer step = 1 minute). "
             "Default 'auto' shows row-count + common-rate examples.",
    )
    p.add_argument("--window-size", type=int, default=64)
    p.add_argument("--n-neighbors", type=int, default=5)
    p.add_argument("--metric", default="euclidean", choices=["euclidean", "cosine", "manhattan"])
    p.add_argument("--weights", default="uniform", choices=["uniform", "distance"])
    p.add_argument("--step-size", type=int, default=1)
    p.add_argument("--nshot-floor", type=int, default=500)

    p.add_argument("--pilot", action="store_true", help="Run held-out pilot via batch API")
    p.add_argument("--pilot-size", type=int, help="Override auto split — rows per class to use as pilot")
    p.add_argument("--force", action="store_true", help="Run pilot even below floor (underpowered)")
    p.add_argument("--env", default=".env", help="Path to .env file")
    return p.parse_args()


def _print_check(r: CheckResult):
    c = COLOR.get(r.level, COLOR["INFO"])
    print(f"  {c}[{r.level:<4}]{COLOR['reset']} {r.name:<20} {r.message}")


def _summary(results: list) -> dict:
    out = {PASS: 0, WARN: 0, FAIL: 0, "INFO": 0}
    for r in results:
        out[r.level] = out.get(r.level, 0) + 1
    return out


def main() -> int:
    args = _parse_args()
    _load_env(args.env)

    print("=" * 80)
    print(" omega-1-4-preflight")
    print("=" * 80)
    print(f"  shots-normal: {args.shots_normal}")
    print(f"  shots-fault:  {args.shots_fault}")
    print(f"  window_size={args.window_size} n_neighbors={args.n_neighbors} "
          f"metric={args.metric} weights={args.weights}")
    print()

    print("[A] Static checks")
    print("-" * 80)
    results, normal_summary, fault_summary = run_all_static_checks(
        args.shots_normal, args.shots_fault,
        window_size=args.window_size,
        timestamp_col=args.timestamp_column,
        nshot_floor=args.nshot_floor,
        timestamp_unit=args.timestamp_unit,
    )
    for r in results:
        _print_check(r)
    s = _summary(results)
    print()
    print(f"  totals: PASS={s[PASS]}  WARN={s[WARN]}  FAIL={s[FAIL]}  INFO={s.get('INFO', 0)}")
    print()

    if s[FAIL] > 0 and not args.pilot:
        print(f"{COLOR[FAIL]}[FAIL]{COLOR['reset']} Static checks failed. Fix these before running --pilot.")
        return 1

    if not args.pilot:
        print("Skipping pilot (no --pilot flag). Run again with --pilot to get a predicted-accuracy number.")
        return 0 if s[FAIL] == 0 else 1

    if s[FAIL] > 0 and not args.force:
        print(f"{COLOR[FAIL]}[FAIL]{COLOR['reset']} Static checks failed. Pass --force to run pilot anyway.")
        return 1

    api_key = os.environ.get("ATAI_API_KEY")
    api_endpoint = os.environ.get("ATAI_API_ENDPOINT")
    if not api_key or not api_endpoint:
        print(f"{COLOR[FAIL]}[FAIL]{COLOR['reset']} ATAI_API_KEY / ATAI_API_ENDPOINT not set (check {args.env}).")
        return 1

    print()
    print("[B] Pilot run")
    print("-" * 80)

    client = ArchetypeClient(api_key, api_endpoint)
    cfg = PilotConfig(
        window_size=args.window_size,
        n_neighbors=args.n_neighbors,
        metric=args.metric,
        weights=args.weights,
        step_size=args.step_size,
        timestamp_col=args.timestamp_column,
    )

    if normal_summary.feature_cols != fault_summary.feature_cols:
        print(f"{COLOR[FAIL]}[FAIL]{COLOR['reset']} Shot files have different feature columns; cannot build pilot job.")
        return 1

    result = run_pilot(
        normal_path=args.shots_normal,
        fault_path=args.shots_fault,
        normal_rows=normal_summary.n_rows,
        fault_rows=fault_summary.n_rows,
        data_columns=normal_summary.feature_cols,
        cfg=cfg,
        client=client,
        force=args.force,
        pilot_size=args.pilot_size,
    )

    print()
    print("=" * 80)
    print(" Pilot Result")
    print("=" * 80)
    c = COLOR.get(result.verdict, COLOR["INFO"])
    print(f"  Verdict:          {c}{result.verdict}{COLOR['reset']} ({result.confidence}-confidence)")
    print(f"  Reason:           {result.reason}")
    print(f"  Accuracy:         {result.accuracy:.3f}")
    print(f"  Macro F1:         {result.macro_f1:.3f}")
    print(f"  Majority baseline:{result.majority_baseline:.3f}")
    if result.job_id:
        print(f"  Job id:           {result.job_id}")
    print()
    if result.per_class:
        print(f"  {'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>10}")
        print(f"  {'-' * 52}")
        for cname, pc in result.per_class.items():
            print(f"  {cname:<12} {pc['precision']:>10.3f} {pc['recall']:>8.3f} "
                  f"{pc['f1']:>8.3f} {pc['support']:>10,}")
        print()

    if result.verdict == "PASS":
        return 0
    if result.verdict == "WARN":
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
