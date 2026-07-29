#!/usr/bin/env python3
"""Run the frozen V17 recipe-1 training and learnability protocol."""

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
CONFIG = Path("configs/v17/ordinal_gate_recipe1_seed17040.yaml")
PROTOCOL = Path("configs/v17/frozen_recipe1_protocol.yaml")
METADATA = Path("data/processed/v17_balanced_ordinal/calibration.csv")
CHECKPOINT_DIR = Path("checkpoints/v17/ordinal_gate_recipe1_seed17040")
RESULT_DIR = Path("results/v17/recipe1_learnability")
RECORD = Path("results/v17/recipe1_execution_record.json")
REQUIRED_EPOCHS = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with (ROOT / path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_record(record: dict) -> None:
    target = ROOT / RECORD
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(target)


def _run(command: list[str], record: dict, stage: str) -> None:
    record["current_stage"] = stage
    record["stages"][stage] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
    }
    _write_record(record)
    print(f"\n=== V17 {stage} ===", flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={
            **os.environ,
            "LD_LIBRARY_PATH": (
                "/home/mohamedb/miniconda3/envs/cnvqg/lib"
            ),
            "PYTHONPATH": "src:scripts",
        },
        check=False,
    )
    stage_record = record["stages"][stage]
    stage_record["finished_at"] = datetime.now(timezone.utc).isoformat()
    stage_record["return_code"] = completed.returncode
    stage_record["status"] = (
        "complete" if completed.returncode == 0 else "failed"
    )
    _write_record(record)
    if completed.returncode != 0:
        raise RuntimeError(
            f"V17 stage {stage!r} failed with code "
            f"{completed.returncode}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    python = "/home/mohamedb/miniconda3/envs/cnvqg/bin/python"

    frozen_files = [
        CONFIG,
        PROTOCOL,
        Path("data/processed/v17_balanced_ordinal/train.csv"),
        METADATA,
        Path("src/cnvqg/models/causal_ordinal_residual_gate_v17.py"),
        Path("src/cnvqg/losses/losses.py"),
        Path("scripts/train_v17_gate.py"),
        Path("scripts/evaluate_v17_learnability.py"),
    ]
    record = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "current_stage": "initialising",
        "device": args.device,
        "frozen_assets": {
            str(path): _sha256(path) for path in frozen_files
        },
        "stages": {},
        "error": None,
    }
    _write_record(record)

    try:
        epochs = sorted((ROOT / CHECKPOINT_DIR).glob("epoch_*.pt"))
        if len(epochs) < REQUIRED_EPOCHS:
            training_command = [
                python,
                "scripts/train_v17_gate.py",
                "--config",
                str(CONFIG),
                "--device",
                args.device,
            ]
            latest = ROOT / CHECKPOINT_DIR / "latest.pt"
            if latest.is_file():
                training_command.extend(["--resume", str(latest)])
            _run(training_command, record, "training")
        else:
            record["stages"]["training"] = {
                "status": "already_complete",
                "epoch_checkpoints": len(epochs),
            }
            _write_record(record)

        _run(
            [
                python,
                "scripts/evaluate_v17_learnability.py",
                "--checkpoint-dir",
                str(CHECKPOINT_DIR),
                "--metadata",
                str(METADATA),
                "--output-dir",
                str(RESULT_DIR),
                "--device",
                args.device,
            ],
            record,
            "learnability_evaluation",
        )
        decision_path = ROOT / RESULT_DIR / "summary.json"
        decision = json.loads(decision_path.read_text())
        record["status"] = "complete"
        record["current_stage"] = None
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        record["selected_checkpoint"] = decision["selected_checkpoint"]
        record["decision"] = decision["decision"]
        _write_record(record)
        print(
            f"\nV17 recipe 1 complete: {record['decision']}\n"
            f"Selected: {record['selected_checkpoint']}",
            flush=True,
        )
    except Exception as error:
        record["status"] = "failed"
        record["error"] = repr(error)
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_record(record)
        raise


if __name__ == "__main__":
    sys.exit(main())
