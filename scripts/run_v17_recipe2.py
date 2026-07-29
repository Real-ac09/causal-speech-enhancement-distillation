#!/usr/bin/env python3
"""Run V17 recipe 2 from local oracle generation through learnability."""

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
RECORD = Path("results/v17/recipe2_execution_record.json")
TRAIN_ORACLES = Path("results/v17/recipe2_local_oracles/train")
CAL_ORACLES = Path("results/v17/recipe2_local_oracles/calibration")
CHECKPOINTS = Path("checkpoints/v17/ordinal_gate_recipe2_seed17041")
LEARNABILITY = Path("results/v17/recipe2_learnability")


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
    print(f"\n=== V17 recipe 2: {stage} ===", flush=True)
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
    stage_record = record["stages"][stage]
    stage_record.update(
        status="complete" if result.returncode == 0 else "failed",
        finished_at=_now(),
        return_code=result.returncode,
    )
    _write(record)
    if result.returncode:
        raise RuntimeError(
            f"Recipe-2 stage {stage} failed with code {result.returncode}"
        )


def _oracle_command(
    *,
    metadata: str,
    output: Path,
    device: str,
) -> list[str]:
    command = [
        PYTHON,
        "scripts/generate_v17_local_oracle_labels.py",
        "--metadata",
        metadata,
        "--output-dir",
        str(output),
        "--device",
        device,
        "--checkpoint-every",
        "50",
    ]
    if (ROOT / output / "progress.json").is_file():
        command.append("--resume")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    frozen = [
        Path("configs/v17/local_oracle_policy.yaml"),
        Path("configs/v17/frozen_recipe2_protocol.yaml"),
        Path("configs/v17/ordinal_gate_recipe2_seed17041.yaml"),
        Path("data/processed/v16_oracle_corpus/train_base.csv"),
        Path("data/processed/v16_oracle_corpus/calibration_base.csv"),
        Path("configs/v16/oracle_corpus_selection.json"),
        Path("checkpoints/v15/preservation/quiet_level_seed1200/epoch_003.pt"),
        Path("src/cnvqg/models/causal_ordinal_residual_gate_v17.py"),
        Path("scripts/generate_v17_local_oracle_labels.py"),
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
        "stages": {},
        "error": None,
    }
    _write(record)
    try:
        if not (ROOT / TRAIN_ORACLES / "summary.json").is_file():
            _run(
                _oracle_command(
                    metadata=(
                        "data/processed/v16_oracle_corpus/train_base.csv"
                    ),
                    output=TRAIN_ORACLES,
                    device=args.device,
                ),
                record,
                "train_local_oracles",
            )
        else:
            record["stages"]["train_local_oracles"] = {
                "status": "already_complete"
            }
        if not (ROOT / CAL_ORACLES / "summary.json").is_file():
            _run(
                _oracle_command(
                    metadata=(
                        "data/processed/v16_oracle_corpus/"
                        "calibration_base.csv"
                    ),
                    output=CAL_ORACLES,
                    device=args.device,
                ),
                record,
                "calibration_local_oracles",
            )
        else:
            record["stages"]["calibration_local_oracles"] = {
                "status": "already_complete"
            }
        _write(record)

        _run(
            [
                PYTHON,
                "scripts/audit_v17_recipe2_labels.py",
                "--train",
                str(TRAIN_ORACLES / "metadata.csv"),
                "--calibration",
                str(CAL_ORACLES / "metadata.csv"),
                "--output",
                "results/v17/recipe2_label_audit.json",
            ],
            record,
            "label_audit",
        )

        smoke_checkpoint = (
            ROOT
            / "checkpoints/v17/ordinal_gate_recipe2_smoke/epoch_001.pt"
        )
        if not smoke_checkpoint.is_file():
            _run(
                [
                    PYTHON,
                    "scripts/train_v17_local_gate.py",
                    "--config",
                    "configs/v17/ordinal_gate_recipe2_seed17041.yaml",
                    "--device",
                    args.device,
                    "--max-train-batches",
                    "2",
                    "--max-val-batches",
                    "1",
                    "--epochs",
                    "1",
                    "--disable-perceptual-validation",
                    "--experiment-name",
                    "ordinal_gate_recipe2_smoke",
                ],
                record,
                "training_smoke",
            )
        else:
            record["stages"]["training_smoke"] = {
                "status": "already_complete"
            }

        epochs = sorted((ROOT / CHECKPOINTS).glob("epoch_*.pt"))
        if len(epochs) < 8:
            command = [
                PYTHON,
                "scripts/train_v17_local_gate.py",
                "--config",
                "configs/v17/ordinal_gate_recipe2_seed17041.yaml",
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
                str(CAL_ORACLES / "metadata.csv"),
                "--output-dir",
                str(LEARNABILITY),
                "--device",
                args.device,
                "--failure-decision",
                (
                    "stop_recipe2_and_reassess_controller_features_or_"
                    "target_policy"
                ),
            ],
            record,
            "learnability_evaluation",
        )
        decision = json.loads(
            (ROOT / LEARNABILITY / "summary.json").read_text()
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
            f"\nV17 recipe 2 complete: {record['decision']}\n"
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
