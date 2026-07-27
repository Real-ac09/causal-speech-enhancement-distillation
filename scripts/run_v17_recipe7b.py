#!/usr/bin/env python3
"""Run frozen Recipe-7b prefix-supervised training and evaluation."""

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
CONFIG = Path("configs/v17/utility_safety_recipe7b_prefix_seed17048.yaml")
PROTOCOL = Path("configs/v17/frozen_recipe7b_training_protocol.yaml")
CHECKPOINTS = Path("checkpoints/v17/utility_safety_recipe7b_prefix_seed17048")
RESULTS = Path("results/v17/recipe7b_selection")
RECORD = Path("results/v17/recipe7b_execution_record.json")


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
    print(f"\n=== V17 Recipe 7b: {stage} ===", flush=True)
    result = subprocess.run(
        command,
        cwd=ROOT,
        env={
            **os.environ,
            "LD_LIBRARY_PATH": "/home/mohamedb/miniconda3/envs/cnvqg/lib",
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
        raise RuntimeError(f"{stage} failed with code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = ROOT / PROTOCOL
    if not protocol.is_file():
        raise RuntimeError("Recipe 7b is not frozen after target audit")
    assets = [
        CONFIG,
        PROTOCOL,
        Path("data/processed/v17_recipe7/train.csv"),
        Path("data/processed/v17_recipe7/calibration.csv"),
        Path("src/cnvqg/models/causal_statistics_utility_gate_v17.py"),
        Path("src/cnvqg/losses/losses.py"),
        Path("scripts/train_v17_recipe7b.py"),
        Path("scripts/evaluate_v17_recipe5.py"),
    ]
    record = {
        "status": "running",
        "started_at": _now(),
        "current_stage": "initialising",
        "device": args.device,
        "frozen_assets": {str(path): _sha256(path) for path in assets},
        "stages": {},
        "error": None,
    }
    _write(record)
    try:
        epochs = sorted((ROOT / CHECKPOINTS).glob("epoch_*.pt"))
        if len(epochs) < 8:
            command = [
                PYTHON,
                "scripts/train_v17_recipe7b.py",
                "--config",
                str(CONFIG),
                "--device",
                args.device,
            ]
            latest = ROOT / CHECKPOINTS / "latest.pt"
            if latest.is_file():
                command.extend(["--resume", str(latest)])
            _run(command, record, "training")
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
                "--failure-decision",
                "stop_recipe7b_and_report_prefix_supervision_limit",
            ],
            record,
            "selection_evaluation",
        )
        summary = json.loads((ROOT / RESULTS / "summary.json").read_text())
        decision = (
            "advance_recipe7b_to_confirmation"
            if summary["selected_summary"]["passes_recipe5_gate"]
            else summary["decision"]
        )
        record.update(
            status="complete",
            current_stage=None,
            completed_at=_now(),
            selected_checkpoint=summary["selected_checkpoint"],
            decision=decision,
        )
        _write(record)
        print(f"\nRecipe 7b complete: {decision}", flush=True)
    except Exception as error:
        record.update(
            status="failed",
            error=repr(error),
            completed_at=_now(),
        )
        _write(record)
        raise


if __name__ == "__main__":
    main()
