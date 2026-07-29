#!/usr/bin/env python3
"""Run the frozen Recipe-8 training-domain programme."""

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
CONFIG = Path(
    "configs/v18/utility_safety_recipe8_two_stage_seed18000.yaml"
)
PROTOCOL = Path("configs/v18/frozen_recipe8_protocol.yaml")
CHECKPOINTS = Path(
    "checkpoints/v18/utility_safety_recipe8_two_stage_seed18000"
)
RESULTS = Path("results/v18/recipe8_selection")
RECORD = Path("results/v18/recipe8_execution_record.json")


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
    assets = [
        CONFIG,
        PROTOCOL,
        Path("data/processed/v17_recipe7/train.csv"),
        Path("data/processed/v17_recipe7/calibration.csv"),
        Path("src/cnvqg/models/causal_two_stage_utility_gate_v18.py"),
        Path("src/cnvqg/losses/losses.py"),
        Path("scripts/train_v18_recipe8.py"),
        Path("scripts/evaluate_v18_two_stage.py"),
    ]
    record = {
        "status": "running",
        "started_at": _now(),
        "current_stage": "initialising",
        "device": args.device,
        "development_set_used": False,
        "external_test_used": False,
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
                "scripts/train_v18_recipe8.py",
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
                "scripts/evaluate_v18_two_stage.py",
                "--checkpoint-dir",
                str(CHECKPOINTS),
                "--output-dir",
                str(RESULTS),
                "--device",
                args.device,
            ],
            record,
            "training_domain_selection",
        )
        summary = json.loads((ROOT / RESULTS / "summary.json").read_text())
        record.update(
            status="complete",
            current_stage=None,
            completed_at=_now(),
            selected_checkpoint=summary["selected_checkpoint"],
            decision=summary["decision"],
        )
        _write(record)
        print(f"Recipe 8 complete: {summary['decision']}", flush=True)
    except Exception as error:
        record.update(
            status="failed",
            current_stage=None,
            error=repr(error),
            completed_at=_now(),
        )
        _write(record)
        raise


if __name__ == "__main__":
    main()
