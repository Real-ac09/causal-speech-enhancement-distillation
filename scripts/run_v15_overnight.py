#!/usr/bin/env python3
"""Guarded overnight controller for V15 candidates B and conditional C."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = Path("results/v15/overnight")
RECORD = RESULT_ROOT / "execution_record.json"
OVERNIGHT_PROTOCOL = Path("configs/v15/frozen_overnight_protocol.yaml")
B_UNIT = "cnvqg-v15-quiet-level-identity-seed1200.service"
B_CHECKPOINT = Path(
    "checkpoints/v15/preservation/quiet_level_identity_seed1200/epoch_003.pt"
)
B_OUTPUT = Path(
    "results/v15/preservation/quiet_level_identity_seed1200"
)
C_CONFIG = Path(
    "configs/v15/quiet_identity_mild_under_seed1200.yaml"
)
C_CHECKPOINT = Path(
    "checkpoints/v15/preservation/"
    "quiet_identity_mild_under_seed1200/epoch_003.pt"
)
C_OUTPUT = Path(
    "results/v15/preservation/quiet_identity_mild_under_seed1200"
)
C_TRAIN_DIRECTORY = C_CHECKPOINT.parent
A_STOI_GAIN = -0.030359359834984445
ALLOWED_ABSOLUTE_STOI_FAILURES = {
    "cross_candidate_minus_noisy_stoi_mean",
    "cross_stoi_harm_rate",
    "cross_quietest_stoi_gain_mean",
    "cross_quietest_stoi_harm_rate",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src:scripts"
    environment["LD_LIBRARY_PATH"] = (
        "/home/mohamedb/miniconda3/envs/cnvqg/lib"
    )
    return environment


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=_environment(), check=True)


def _write_record(record: dict[str, object]) -> None:
    path = ROOT / RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def _verify_protocol() -> str:
    import yaml

    protocol_path = ROOT / OVERNIGHT_PROTOCOL
    protocol = yaml.safe_load(protocol_path.read_text())
    if protocol.get("status") != "frozen_before_overnight_execution":
        raise ValueError("Overnight protocol is not frozen")
    for path, expected in {
        **protocol["frozen_inputs"],
        **protocol["source_files"],
    }.items():
        actual = _sha256(ROOT / path)
        if actual != expected:
            raise ValueError(
                f"Frozen overnight input changed: {path}\n"
                f"expected {expected}\nactual   {actual}"
            )
    return _sha256(protocol_path)


def _unit_active(unit: str) -> bool:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", unit],
        check=False,
    )
    return result.returncode == 0


def _wait_for_b(record: dict[str, object]) -> None:
    print(f"Waiting for {B_UNIT}", flush=True)
    while _unit_active(B_UNIT):
        time.sleep(30)
    if not (ROOT / B_CHECKPOINT).is_file():
        raise RuntimeError("Candidate B stopped without epoch_003.pt")
    metrics = B_CHECKPOINT.parent / "metrics.csv"
    if not metrics.is_file() or len(metrics.read_text().splitlines()) != 4:
        raise RuntimeError("Candidate B metrics are incomplete")
    record["candidate_b"]["training"] = {
        "status": "complete",
        "checkpoint": str(B_CHECKPOINT),
        "checkpoint_sha256": _sha256(ROOT / B_CHECKPOINT),
        "completed_at": _now(),
    }
    _write_record(record)


def _evaluate(
    *,
    checkpoint: Path,
    candidate_name: str,
    output_root: Path,
    seed: int,
) -> dict[str, object]:
    _run(
        [
            sys.executable,
            "scripts/run_v15_candidate_evaluation.py",
            "--checkpoint",
            str(checkpoint),
            "--candidate-name",
            candidate_name,
            "--output-root",
            str(output_root),
            "--bootstrap-seed",
            str(seed),
            "--device",
            "cuda",
        ]
    )
    return json.loads((ROOT / output_root / "gate/gate_report.json").read_text())


def _candidate_c_permitted(report: dict[str, object]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    failures = set(report["gate_summary"]["failed_names"])
    stoi_gain = float(
        report["cross_domain"]["metrics"]["stoi"]["candidate_gain_mean"]
    )
    if report["decision"] == "promote_to_three_seed_replication":
        reasons.append("candidate_b_already_passed")
    if stoi_gain <= A_STOI_GAIN:
        reasons.append("candidate_b_did_not_improve_absolute_stoi_over_a")
    if not failures.issubset(ALLOWED_ABSOLUTE_STOI_FAILURES):
        reasons.append("candidate_b_failed_non_stoi_or_relative_safeguard")
    return not reasons, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("Dry run: B evaluation followed by guarded conditional C.")
        return
    protocol_digest = _verify_protocol()
    record: dict[str, object] = {
        "status": "running",
        "started_at": _now(),
        "protocol": str(OVERNIGHT_PROTOCOL),
        "protocol_sha256": protocol_digest,
        "external_test_used": False,
        "candidate_b": {},
        "candidate_c": {"status": "not_considered"},
    }
    _write_record(record)
    try:
        _wait_for_b(record)
        b_report = _evaluate(
            checkpoint=B_CHECKPOINT,
            candidate_name="v15_quiet_level_identity_seed1200_epoch3",
            output_root=B_OUTPUT,
            seed=15_020,
        )
        record["candidate_b"]["evaluation"] = {
            "status": "complete",
            "gate_report": str(B_OUTPUT / "gate/gate_report.json"),
            "decision": b_report["decision"],
            "gate_summary": b_report["gate_summary"],
        }
        permitted, reasons = _candidate_c_permitted(b_report)
        record["conditional_c"] = {
            "permitted": permitted,
            "reasons_not_permitted": reasons,
            "candidate_b_absolute_stoi_gain": b_report["cross_domain"][
                "metrics"
            ]["stoi"]["candidate_gain_mean"],
            "candidate_a_absolute_stoi_gain": A_STOI_GAIN,
        }
        _write_record(record)
        if not permitted:
            record["status"] = "complete"
            record["final_decision"] = (
                "candidate_b_ready_for_replication"
                if b_report["decision"]
                == "promote_to_three_seed_replication"
                else "stop_after_candidate_b"
            )
            record["completed_at"] = _now()
            _write_record(record)
            return

        if C_TRAIN_DIRECTORY.exists():
            raise RuntimeError(
                f"Candidate C output already exists: {C_TRAIN_DIRECTORY}"
            )
        record["candidate_c"] = {
            "status": "training",
            "started_at": _now(),
            "config": str(C_CONFIG),
        }
        _write_record(record)
        _run(
            [
                sys.executable,
                "scripts/train_distill_v15_identity.py",
                "--config",
                str(C_CONFIG),
                "--device",
                "cuda",
            ]
        )
        if not (ROOT / C_CHECKPOINT).is_file():
            raise RuntimeError("Candidate C training did not produce epoch 3")
        record["candidate_c"]["training"] = {
            "status": "complete",
            "checkpoint": str(C_CHECKPOINT),
            "checkpoint_sha256": _sha256(ROOT / C_CHECKPOINT),
            "completed_at": _now(),
        }
        _write_record(record)
        c_report = _evaluate(
            checkpoint=C_CHECKPOINT,
            candidate_name="v15_quiet_identity_mild_under_seed1200_epoch3",
            output_root=C_OUTPUT,
            seed=15_030,
        )
        record["candidate_c"]["status"] = "complete"
        record["candidate_c"]["evaluation"] = {
            "gate_report": str(C_OUTPUT / "gate/gate_report.json"),
            "decision": c_report["decision"],
            "gate_summary": c_report["gate_summary"],
        }
        record["status"] = "complete"
        record["final_decision"] = (
            "candidate_c_ready_for_replication"
            if c_report["decision"] == "promote_to_three_seed_replication"
            else "stop_after_candidate_c"
        )
        record["completed_at"] = _now()
        _write_record(record)
    except Exception as error:
        record["status"] = "failed"
        record["error"] = repr(error)
        record["failed_at"] = _now()
        _write_record(record)
        raise


if __name__ == "__main__":
    main()
