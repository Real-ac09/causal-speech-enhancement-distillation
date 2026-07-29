#!/usr/bin/env python3
"""Acquire and prepare frozen Recipe-7b one-second prefix targets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/mohamedb/miniconda3/envs/cnvqg/bin/python"
RECORD = Path("results/v17/recipe7_prefix_oracles_execution_record.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    print(f"\n=== V17 Recipe 7b assets: {stage} ===", flush=True)
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
    record = {
        "status": "running",
        "started_at": _now(),
        "current_stage": "initialising",
        "device": args.device,
        "protocol": (
            "configs/v17/frozen_recipe7b_prefix_oracle_protocol.yaml"
        ),
        "stages": {},
        "error": None,
    }
    _write(record)
    try:
        for split in ("train", "calibration"):
            output = Path("results/v17/recipe7_prefix_oracles") / split
            command = [
                PYTHON,
                "scripts/generate_v17_prefix_oracles.py",
                "--metadata",
                (
                    f"results/v17/recipe2_local_oracles/{split}/"
                    "metadata.csv"
                ),
                "--output-dir",
                str(output),
                "--device",
                args.device,
            ]
            if (ROOT / output / "progress.json").is_file():
                command.append("--resume")
            _run(command, record, f"{split}_prefix_oracles")
        _run(
            [PYTHON, "scripts/prepare_v17_recipe7_targets.py"],
            record,
            "prepare_metadata",
        )
        record.update(
            status="complete",
            current_stage=None,
            completed_at=_now(),
            decision="freeze_and_train_recipe7b",
        )
        _write(record)
        print("\nRecipe 7b prefix targets are ready.", flush=True)
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
