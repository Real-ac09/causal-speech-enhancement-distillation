#!/usr/bin/env python3
"""Execute fixed-epoch V15 candidate D and its frozen evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path("configs/v15/frozen_candidate_d_protocol.yaml")
CONFIG = Path("configs/v15/causal_preservation_gate_seed1200.yaml")
CHECKPOINT_DIRECTORY = Path(
    "checkpoints/v15/preservation/causal_preservation_gate_seed1200"
)
CHECKPOINT = CHECKPOINT_DIRECTORY / "epoch_003.pt"
OUTPUT_ROOT = Path(
    "results/v15/preservation/causal_preservation_gate_seed1200"
)
RECORD = OUTPUT_ROOT / "execution_record.json"


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
    subprocess.run(
        command,
        cwd=ROOT,
        env=_environment(),
        check=True,
    )


def _write_record(record: dict[str, object]) -> None:
    path = ROOT / RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def _verify_protocol() -> str:
    import yaml

    protocol_path = ROOT / PROTOCOL
    protocol = yaml.safe_load(protocol_path.read_text())
    if protocol.get("status") != "frozen_before_execution":
        raise ValueError("Candidate D protocol is not frozen")
    for path, expected in {
        **protocol["frozen_inputs"],
        **protocol["source_files"],
    }.items():
        actual = _sha256(ROOT / path)
        if actual != expected:
            raise ValueError(
                f"Frozen candidate D input changed: {path}\n"
                f"expected {expected}\nactual   {actual}"
            )
    return _sha256(protocol_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("Dry run: fixed epoch-3 candidate D training and evaluation.")
        return
    protocol_digest = _verify_protocol()
    if (ROOT / CHECKPOINT_DIRECTORY).exists():
        raise RuntimeError(
            f"Candidate D checkpoint directory exists: {CHECKPOINT_DIRECTORY}"
        )
    if (ROOT / OUTPUT_ROOT).exists():
        raise RuntimeError(
            f"Candidate D result directory exists: {OUTPUT_ROOT}"
        )
    record: dict[str, object] = {
        "status": "running",
        "started_at": _now(),
        "protocol": str(PROTOCOL),
        "protocol_sha256": protocol_digest,
        "external_test_used": False,
        "training": {"status": "running"},
        "evaluation": {"status": "pending"},
    }
    _write_record(record)
    try:
        _run(
            [
                sys.executable,
                "scripts/train_v15_gate.py",
                "--config",
                str(CONFIG),
                "--device",
                "cuda",
            ]
        )
        metrics = ROOT / CHECKPOINT_DIRECTORY / "metrics.csv"
        if not (ROOT / CHECKPOINT).is_file():
            raise RuntimeError("Candidate D did not produce epoch_003.pt")
        if not metrics.is_file() or len(metrics.read_text().splitlines()) != 4:
            raise RuntimeError("Candidate D training metrics are incomplete")
        record["training"] = {
            "status": "complete",
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": _sha256(ROOT / CHECKPOINT),
            "completed_at": _now(),
        }
        record["evaluation"] = {"status": "running"}
        _write_record(record)
        _run(
            [
                sys.executable,
                "scripts/run_v15_candidate_d_evaluation.py",
                "--checkpoint",
                str(CHECKPOINT),
                "--candidate-name",
                "v15_causal_preservation_gate_seed1200_epoch3",
                "--output-root",
                str(OUTPUT_ROOT),
                "--bootstrap-seed",
                "15040",
                "--device",
                "cuda",
            ]
        )
        gate_path = ROOT / OUTPUT_ROOT / "gate/gate_report.json"
        gate = json.loads(gate_path.read_text())
        record["evaluation"] = {
            "status": "complete",
            "gate_report": str(
                OUTPUT_ROOT / "gate/gate_report.json"
            ),
            "decision": gate["decision"],
            "gate_summary": gate["gate_summary"],
            "deployment": gate["deployment_measurement"],
        }
        record["status"] = "complete"
        record["final_decision"] = gate["decision"]
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
