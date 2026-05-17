"""Pilot runner — splits shots into shots+labeled-pilot, uploads, runs a batch
job against the machine-state-classification pipeline, scores predictions vs. known labels.

The pipeline_key defaults to `machine-state-classification` (the active deployment on both
stage and prod as of 2026-05). Override with the `ATAI_PIPELINE_KEY` environment variable
if you're targeting a different deployment."""

from __future__ import annotations

import csv
import io
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import requests


TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}


@dataclass
class PilotConfig:
    window_size: int = 64
    n_neighbors: int = 5
    metric: str = "euclidean"
    weights: str = "uniform"
    step_size: int = 1
    timestamp_col: str = "timestamp"
    batch_size: int = 32
    flush_every_n_iteration: int = 150
    model_type: str = "omega_1_4_base"
    parallelism: int = 1


@dataclass
class SplitPlan:
    shot_rows: int
    pilot_rows: int
    confidence: str  # "high" | "low" | "refused"
    reason: str


@dataclass
class PilotResult:
    verdict: str  # "PASS" | "WARN" | "FAIL"
    accuracy: float
    macro_f1: float
    per_class: dict
    confusion: dict
    majority_baseline: float
    confidence: str
    reason: str
    job_id: Optional[str] = None


# Minimum non-overlap guards for the --force and --pilot-size paths. Files at or
# above the 500-row "low confidence" threshold use the larger 100-row floors
# baked into those branches; these guards only kick in for smaller files where
# the old 100-row floors would have produced overlapping shots/pilot slices.
_MIN_SHOT_ROWS = 10
_MIN_PILOT_ROWS = 10


def plan_split(n_rows: int, force: bool = False, size_override: Optional[int] = None) -> SplitPlan:
    """Default 80/20 rule with floors, from the design spec."""
    if size_override is not None:
        pilot = min(size_override, max(0, n_rows - _MIN_SHOT_ROWS))
        shot = n_rows - pilot
        if (shot < _MIN_SHOT_ROWS or pilot < _MIN_PILOT_ROWS) and not force:
            return SplitPlan(0, 0, "refused",
                f"Override leaves shots={shot} pilot={pilot} "
                f"(need ≥{_MIN_SHOT_ROWS} each — pass --force to bypass).")
        return SplitPlan(shot, pilot, "overridden",
            f"User-specified pilot size {pilot} (shots {shot}).")

    if n_rows >= 1000:
        pilot = max(300, n_rows // 5)
        return SplitPlan(n_rows - pilot, pilot, "high",
            f"Standard 80/20 split ({n_rows - pilot} shots, {pilot} pilot).")
    if n_rows >= 500:
        pilot = max(300, n_rows * 3 // 10)
        shot = n_rows - pilot
        return SplitPlan(shot, pilot, "low",
            f"70/30 split near the floor ({shot} shots, {pilot} pilot). Low-confidence.")
    if force:
        pilot = max(_MIN_PILOT_ROWS, n_rows // 3)
        shot = n_rows - pilot
        if shot < _MIN_SHOT_ROWS:
            return SplitPlan(0, 0, "refused",
                f"Only {n_rows} rows — even with --force, a 1/3 pilot split "
                f"leaves shots={shot} (<{_MIN_SHOT_ROWS}). Expand shots or lower --pilot-size.")
        return SplitPlan(shot, pilot, "low",
            f"--force override below floor ({shot} shots, {pilot} pilot). Underpowered.")
    return SplitPlan(0, 0, "refused",
        f"Only {n_rows} rows (<500 floor). Use --pilot-set FILE, --force, or expand shots.")


def _split_csv(path: str, shot_n: int, pilot_n: int, stamp: str = "") -> tuple[str, str]:
    """Write first shot_n rows to temp shots file, last pilot_n rows to temp pilot file.
    Returns (shot_tmp_path, pilot_tmp_path). Both preserve the header.
    `stamp` is appended to filenames so repeated pilot runs don't collide with previously
    uploaded files on the platform."""
    tmpdir = tempfile.mkdtemp(prefix="preflight_")
    base = os.path.splitext(os.path.basename(path))[0]
    suffix = f"_{stamp}" if stamp else ""
    shot_out = os.path.join(tmpdir, f"{base}_shots{suffix}.csv")
    pilot_out = os.path.join(tmpdir, f"{base}_pilot{suffix}.csv")

    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    with open(shot_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows[:shot_n])

    with open(pilot_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows[-pilot_n:] if pilot_n else [])

    return shot_out, pilot_out


def _concat_csvs(normal_pilot: str, fault_pilot: str, out_path: str) -> int:
    """Combine two pilot CSVs into one inference CSV. Returns total data rows.
    Preserves header from first file.

    Smaller class goes first: the pipeline tags each window with its first row's
    timestamp, so a trailing class with fewer rows than window_size loses all
    its tagged windows."""
    with open(normal_pilot, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        normal_rows = list(reader)
    with open(fault_pilot, newline="") as f:
        reader = csv.reader(f)
        fheader = next(reader)
        fault_rows = list(reader)

    if header != fheader:
        raise ValueError("Pilot files have different headers")

    if len(fault_rows) <= len(normal_rows):
        first_rows, second_rows = fault_rows, normal_rows
    else:
        first_rows, second_rows = normal_rows, fault_rows

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(first_rows)
        w.writerows(second_rows)

    return len(normal_rows) + len(fault_rows)


def _pilot_label_map(
    inference_path: str,
    normal_pilot: str,
    fault_pilot: str,
    ts_col: str,
    window_size: int,
) -> dict:
    """Window-first-row-index -> true label, content-aware.

    The pilot's combined inference file places one class's pilot rows
    immediately after the other's (see `_concat_csvs`). For each prediction
    Newton emits, we score it against the true class of its window's
    *content* (all `window_size` rows). Windows whose rows mix classes are
    dropped — labelling them by their first row would penalize the model
    for correctly classifying the window's actual content.

    Predictions are keyed by row index (0..N-window_size) rather than by
    timestamp because shot files can share timestamp values (e.g. when two
    contiguous 5-second slices both start at t=0) and high-rate sources
    such as 100 kHz vibration would otherwise collapse to a handful of
    integer-truncated keys, losing all alignment. `ts_col` is kept for
    backward compatibility but no longer used."""
    del ts_col  # row-index keying makes this irrelevant

    def _count_rows(path: str) -> int:
        with open(path, newline="") as f:
            return sum(1 for _ in f) - 1  # exclude header

    normal_n = _count_rows(normal_pilot)
    fault_n = _count_rows(fault_pilot)

    # `_concat_csvs` puts the smaller class first; for ties, fault goes first.
    if fault_n <= normal_n:
        first_class, first_n = "fault", fault_n
        second_class = "normal"
    else:
        first_class, first_n = "normal", normal_n
        second_class = "fault"

    total = normal_n + fault_n
    labels: dict = {}
    for i in range(total - window_size + 1):
        end = i + window_size
        if end <= first_n:
            labels[i] = first_class
        elif i >= first_n:
            labels[i] = second_class
        # else: window straddles class boundary → drop
    return labels


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
class ArchetypeClient:
    def __init__(self, api_key: str, endpoint: str):
        self.base = f"{endpoint.rstrip('/')}/v0.5"
        self.auth = {"Authorization": f"Bearer {api_key}"}

    def upload(self, path: str) -> str:
        """Multipart upload. Returns the filename (used as file_id downstream)."""
        size = os.path.getsize(path)
        filename = os.path.basename(path)
        init = requests.post(
            f"{self.base}/files/uploads/initiate",
            headers={**self.auth, "Content-Type": "application/json"},
            json={"filename": filename, "file_type": "text/csv", "num_bytes": size},
        )
        init.raise_for_status()
        info = init.json()
        upload_id = info["upload_id"]
        parts_meta = info["parts"]

        completed = []
        try:
            with open(path, "rb") as f:
                for p in parts_meta:
                    f.seek(p["offset"])
                    data = f.read(p["length"])
                    put = requests.put(p["url"], data=data,
                                       headers={"Content-Length": str(p["length"])})
                    put.raise_for_status()
                    etag = put.headers.get("ETag", "").strip('"')
                    completed.append({"part_number": p["part_number"], "part_token": etag})
        except Exception:
            requests.post(f"{self.base}/files/uploads/{upload_id}/abort", headers=self.auth)
            raise

        done = requests.post(
            f"{self.base}/files/uploads/{upload_id}/complete",
            headers={**self.auth, "Content-Type": "application/json"},
            json={"parts": completed},
        )
        done.raise_for_status()
        return filename

    def create_job(
        self,
        inference_file: str,
        normal_shot_file: str,
        fault_shot_file: str,
        data_columns: list,
        cfg: PilotConfig,
        name: str = "preflight-pilot",
    ) -> str:
        payload = {
            "name": name,
            "pipeline_type": "batch",
            "pipeline_key": os.environ.get("ATAI_PIPELINE_KEY", "machine-state-classification"),
            "inputs": {
                "worker.inference": [{"file_id": inference_file}],
                "worker.n_shots": [
                    {"file_id": normal_shot_file, "metadata": {"class": "normal"}},
                    {"file_id": fault_shot_file, "metadata": {"class": "fault"}},
                ],
            },
            "parameters": {
                "worker": {
                    "parallelism": cfg.parallelism,
                    "config": {
                        "batch_size": cfg.batch_size,
                        "classifier_config": {
                            "metric": cfg.metric,
                            "n_neighbors": cfg.n_neighbors,
                            "normalize_embeddings": False,
                            "weights": cfg.weights,
                        },
                        "flush_every_n_iteration": cfg.flush_every_n_iteration,
                        "model_type": cfg.model_type,
                        "reader_config": {
                            "data_columns": data_columns,
                            "step_size": cfg.step_size,
                            "timestamp_column": cfg.timestamp_col,
                            "window_size": cfg.window_size,
                        },
                    },
                }
            },
        }
        resp = requests.post(f"{self.base}/batch/jobs",
                             headers={**self.auth, "Content-Type": "application/json"}, json=payload)
        resp.raise_for_status()
        return resp.json()["id"]

    def wait_for_job(self, job_id: str, poll_s: int = 5, timeout_s: int = 1800) -> str:
        t0 = time.time()
        last_status = None
        while True:
            resp = requests.get(f"{self.base}/batch/jobs/{job_id}", headers=self.auth)
            resp.raise_for_status()
            job = resp.json()
            status = job["status"]
            if status != last_status:
                print(f"      [{time.strftime('%H:%M:%S')}] {status}")
                last_status = status
            if status in TERMINAL_STATUSES:
                return status
            if time.time() - t0 > timeout_s:
                return "TIMEOUT"
            time.sleep(poll_s)

    def get_predictions(self, job_id: str) -> dict:
        """row_index -> predicted label.

        Newton emits one prediction per window in input row order, so the
        i-th prediction in the output CSV corresponds to the window starting
        at input row i. Keying by sequence position avoids the pitfalls of
        matching by timestamp (column-name drift across pipeline versions,
        duplicate timestamps when shot files share a time origin, and loss
        of resolution for high-rate sources like 100 kHz vibration)."""
        outputs = []
        offset = 0
        while True:
            resp = requests.get(f"{self.base}/batch/jobs/{job_id}/outputs",
                                headers=self.auth, params={"limit": 50, "offset": offset})
            resp.raise_for_status()
            data = resp.json()
            outputs.extend(data["outputs"])
            total = data["total"]
            if offset + 50 >= total:
                break
            offset += 50

        preds: dict = {}
        for out in outputs:
            url = out["data"]["ref"]
            r = requests.get(url)
            r.raise_for_status()
            for i, row in enumerate(csv.DictReader(io.StringIO(r.text))):
                preds[i] = row["Prediction"]
        return preds


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score(predictions: dict, labels: dict, classes: list = None) -> dict:
    classes = classes or ["normal", "fault"]
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cm = [[0] * n for _ in range(n)]
    correct = matched = 0
    for ts, pred in predictions.items():
        if ts not in labels:
            continue
        matched += 1
        actual = labels[ts]
        ai, pi = idx.get(actual), idx.get(pred)
        if ai is not None and pi is not None:
            cm[ai][pi] += 1
            if actual == pred:
                correct += 1

    accuracy = correct / matched if matched else 0.0
    per_class = {}
    for i, c in enumerate(classes):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(n)) - tp
        fn = sum(cm[i][j] for j in range(n)) - tp
        sup = sum(cm[i])
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_class[c] = {"precision": p, "recall": r, "f1": f1, "support": sup}
    macro_f1 = sum(pc["f1"] for pc in per_class.values()) / n
    total_support = sum(pc["support"] for pc in per_class.values())
    majority_support = max(pc["support"] for pc in per_class.values())
    majority_baseline = majority_support / total_support if total_support else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion": {classes[i]: {classes[j]: cm[i][j] for j in range(n)} for i in range(n)},
        "matched": matched,
        "majority_baseline": majority_baseline,
    }


def run_pilot(
    normal_path: str,
    fault_path: str,
    normal_rows: int,
    fault_rows: int,
    data_columns: list,
    cfg: PilotConfig,
    client: ArchetypeClient,
    force: bool = False,
    pilot_size: Optional[int] = None,
) -> PilotResult:
    n_plan = plan_split(normal_rows, force=force, size_override=pilot_size)
    f_plan = plan_split(fault_rows, force=force, size_override=pilot_size)
    if n_plan.confidence == "refused" or f_plan.confidence == "refused":
        return PilotResult(
            verdict="FAIL", accuracy=0.0, macro_f1=0.0, per_class={}, confusion={},
            majority_baseline=0.0,
            confidence="refused",
            reason=f"Split refused. normal: {n_plan.reason} | fault: {f_plan.reason}",
        )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    print(f"[pilot] run id: {stamp}")
    print(f"[pilot] split plan -> normal: {n_plan.reason}")
    print(f"[pilot] split plan -> fault:  {f_plan.reason}")

    print("[pilot] writing temp shot + pilot files...")
    normal_shot, normal_pilot = _split_csv(normal_path, n_plan.shot_rows, n_plan.pilot_rows, stamp)
    fault_shot, fault_pilot = _split_csv(fault_path, f_plan.shot_rows, f_plan.pilot_rows, stamp)

    pilot_combined = os.path.join(os.path.dirname(normal_pilot),
                                  f"preflight_pilot_inference_{stamp}.csv")
    n_rows = _concat_csvs(normal_pilot, fault_pilot, pilot_combined)
    print(f"[pilot] combined pilot inference: {n_rows} rows at {pilot_combined}")

    print("[pilot] uploading 3 files (shots + pilot inference)...")
    normal_id = client.upload(normal_shot)
    fault_id = client.upload(fault_shot)
    pilot_id = client.upload(pilot_combined)

    print("[pilot] creating batch job...")
    job_id = client.create_job(pilot_id, normal_id, fault_id, data_columns, cfg,
                               name=f"preflight-pilot-{stamp}")
    print(f"[pilot] job_id={job_id}")

    print("[pilot] waiting for completion...")
    status = client.wait_for_job(job_id)
    if status != "COMPLETED":
        return PilotResult(
            verdict="FAIL", accuracy=0.0, macro_f1=0.0, per_class={}, confusion={},
            majority_baseline=0.0,
            confidence=n_plan.confidence,
            reason=f"Job ended with status={status}",
            job_id=job_id,
        )

    print("[pilot] downloading predictions...")
    preds = client.get_predictions(job_id)
    labels = _pilot_label_map(pilot_combined, normal_pilot, fault_pilot,
                              cfg.timestamp_col, cfg.window_size)
    scorable = sum(1 for ts in preds if ts in labels)
    dropped = len(preds) - scorable
    print(f"[pilot] scoring {scorable} pure-class windows "
          f"({dropped} mixed-content windows dropped)")
    m = score(preds, labels)

    # verdict
    thresh = 0.70
    if m["accuracy"] >= thresh and m["accuracy"] > m["majority_baseline"] + 0.05:
        verdict = "PASS"
        reason = f"Accuracy {m['accuracy']:.3f} clears 70% bar and beats majority baseline {m['majority_baseline']:.3f}."
    elif m["accuracy"] >= m["majority_baseline"]:
        verdict = "WARN"
        reason = (f"Accuracy {m['accuracy']:.3f} below 70% bar (but {'ties' if abs(m['accuracy'] - m['majority_baseline']) < 0.01 else 'above'} "
                  f"majority baseline {m['majority_baseline']:.3f}). Tuning may close the gap.")
    else:
        verdict = "FAIL"
        reason = (f"Accuracy {m['accuracy']:.3f} < majority baseline {m['majority_baseline']:.3f}. "
                  f"Base model not useful for this dataset; consider fine-tuning.")

    confidence = "low" if (n_plan.confidence == "low" or f_plan.confidence == "low") else n_plan.confidence

    return PilotResult(
        verdict=verdict,
        accuracy=m["accuracy"],
        macro_f1=m["macro_f1"],
        per_class=m["per_class"],
        confusion=m["confusion"],
        majority_baseline=m["majority_baseline"],
        confidence=confidence,
        reason=reason,
        job_id=job_id,
    )
