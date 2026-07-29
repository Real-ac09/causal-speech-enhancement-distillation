#!/usr/bin/env python3
"""Run frozen V17 Recipe-5 training and selection evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/mohamedb/miniconda3/envs/cnvqg/bin/python"
CONFIG = Path("configs/v17/utility_safety_recipe5_seed17044.yaml")
CHECKPOINTS = Path("checkpoints/v17/utility_safety_recipe5_seed17044")
RESULTS = Path("results/v17/recipe5_selection")
RECORD = Path("results/v17/recipe5_execution_record.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with (ROOT / path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(record: dict) -> None:
    path = ROOT / RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def _run(command: list[str], record: dict, stage: str) -> None:
    record["current_stage"] = stage
    record["stages"][stage] = {
        "status": "running",
        "started_at": _now(),
        "command": command,
    }
    _write(record)
    print(f"\n=== V17 Recipe 5: {stage} ===", flush=True)
    result = subprocess.run(
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
    record["stages"][stage].update(
        status="complete" if result.returncode == 0 else "failed",
        finished_at=_now(),
        return_code=result.returncode,
    )
    _write(record)
    if result.returncode:
        raise RuntimeError(
            f"Recipe-5 stage {stage} failed with code {result.returncode}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    frozen = [
        CONFIG,
        Path("configs/v17/frozen_recipe5_protocol.yaml"),
        Path("data/processed/v17_recipe5/train.csv"),
        Path("data/processed/v17_recipe5/calibration.csv"),
        Path("data/processed/v17_recipe5/summary.json"),
        Path("src/cnvqg/models/causal_utility_safety_gate_v17.py"),
        Path("src/cnvqg/losses/losses.py"),
        Path("scripts/train.py"),
        Path("scripts/train_v17_recipe5.py"),
        Path("scripts/evaluate_v17_recipe5.py"),
    ]
    record = {
        "status": "running",
        "started_at": _now(),
        "current_stage": "initialising",
        "device": args.device,
        "frozen_assets": {
            str(path): _sha256(path) for path in frozen
        },
        "stages": {
            "target_preparation": {
                "status": "complete",
                "train_items": 3437,
                "calibration_items": 638,
            },
            "causality_gradient_native_smoke": {
                "status": "complete",
                "gate_parameters": 3603,
            },
            "cuda_training_evaluator_smoke": {
                "status": "complete",
                "checkpoint": (
                    "checkpoints/v17/utility_safety_recipe5_smoke/"
                    "epoch_001.pt"
                ),
            },
        },
        "error": None,
    }
    _write(record)
    try:
        epochs = sorted((ROOT / CHECKPOINTS).glob("epoch_*.pt"))
        if len(epochs) < 8:
            command = [
                PYTHON,
                "scripts/train_v17_recipe5.py",
                "--config",
                str(CONFIG),
                "--device",
                args.device,
            ]
            latest = ROOT / CHECKPOINTS / "latest.pt"
            if latest.is_file():
                command.extend(["--resume", str(latest)])
            _run(command, record, "training")
        else:
            record["stages"]["training"] = {
                "status": "already_complete",
                "epoch_checkpoints": len(epochs),
            }
            _write(record)
        _run(
            [
                PYTHON,
                "scripts/evaluate_v17_recipe5.py",
                "--checkpoint-dir",
                str(CHECKPOINTS),
                "--output-dir",
                str(RESULTS),
                "--device",
                args.device,
            ],
            record,
            "selection_evaluation",
        )
        decision = json.loads(
            (ROOT / RESULTS / "summary.json").read_text()
        )
        record.update(
            status="complete",
            current_stage=None,
            finished_at=_now(),
            selected_checkpoint=decision["selected_checkpoint"],
            decision=decision["decision"],
        )
        _write(record)
        print(
            f"\nV17 Recipe 5 complete: {record['decision']}\n"
            f"Selected: {record['selected_checkpoint']}",
            flush=True,
        )
    except Exception as error:
        record.update(
            status="failed",
            error=repr(error),
            finished_at=_now(),
        )
        _write(record)
        raise


if __name__ == "__main__":
    main()
