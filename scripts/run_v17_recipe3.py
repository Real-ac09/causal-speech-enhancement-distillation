#!/usr/bin/env python3
"""Run V17 recipe 3 training and frozen learnability evaluation."""

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
CONFIG = Path("configs/v17/cumulative_recipe3_seed17042.yaml")
CHECKPOINTS = Path("checkpoints/v17/cumulative_recipe3_seed17042")
RESULTS = Path("results/v17/recipe3_learnability")
RECORD = Path("results/v17/recipe3_execution_record.json")


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
    print(f"\n=== V17 recipe 3: {stage} ===", flush=True)
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
            f"Recipe-3 stage {stage} failed with code {result.returncode}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    frozen = [
        CONFIG,
        Path("configs/v17/frozen_recipe3_protocol.yaml"),
        Path("results/v17/recipe2_local_oracles/train/metadata.csv"),
        Path(
            "results/v17/recipe2_local_oracles/calibration/metadata.csv"
        ),
        Path("results/v17/recipe2_label_audit.json"),
        Path("src/cnvqg/models/causal_cumulative_ordinal_gate_v17.py"),
        Path("src/cnvqg/losses/losses.py"),
        Path("scripts/train_v17_local_gate.py"),
        Path("scripts/evaluate_v17_learnability.py"),
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
            "cpu_gradient_and_native_streaming_smoke": {
                "status": "complete"
            },
            "cuda_training_smoke": {
                "status": "complete",
                "checkpoint": (
                    "checkpoints/v17/cumulative_recipe3_smoke/"
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
                "scripts/train_v17_local_gate.py",
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
                "scripts/evaluate_v17_learnability.py",
                "--checkpoint-dir",
                str(CHECKPOINTS),
                "--metadata",
                (
                    "results/v17/recipe2_local_oracles/"
                    "calibration/metadata.csv"
                ),
                "--output-dir",
                str(RESULTS),
                "--device",
                args.device,
                "--failure-decision",
                "stop_recipe3_and_expand_causal_controller_features",
            ],
            record,
            "learnability_evaluation",
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
            f"\nV17 recipe 3 complete: {record['decision']}\n"
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
