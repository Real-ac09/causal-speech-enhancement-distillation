#!/usr/bin/env python3
"""Run resumable stages of the frozen V16 oracle-gate experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path("configs/v16/frozen_oracle_gate_protocol.yaml")
SELECTION = Path("configs/v16/oracle_corpus_selection.json")
CORPUS = Path("data/processed/v16_oracle_corpus")
TRAIN_BASE = CORPUS / "train_base.csv"
CALIBRATION_BASE = CORPUS / "calibration_base.csv"
TRAIN_LABELS = Path("results/v16/oracle_labels/train")
CALIBRATION_LABELS = Path("results/v16/oracle_labels/calibration")
TRAIN_CONFIG = Path("configs/v16/oracle_gate_seed16040.yaml")
CHECKPOINT_DIR = Path("checkpoints/v16/oracle_gate_seed16040")
CHECKPOINT = CHECKPOINT_DIR / "best.pt"
OUTPUT_ROOT = Path("results/v16/oracle_gate_seed16040")
RECORD = OUTPUT_ROOT / "execution_record.json"
STAGES = ("prepare", "labels", "train", "evaluate")


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


def _write_record(record: dict[str, Any]) -> None:
    path = ROOT / RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def _verify_protocol() -> str:
    protocol_path = ROOT / PROTOCOL
    protocol = yaml.safe_load(protocol_path.read_text())
    if protocol.get("status") != "frozen_before_execution":
        raise ValueError("V16 protocol is not frozen before execution")
    for path, expected in {
        **protocol["frozen_inputs"],
        **protocol["source_files"],
    }.items():
        actual = _sha256(ROOT / path)
        if actual != expected:
            raise ValueError(
                f"Frozen V16 input changed: {path}\n"
                f"expected {expected}\nactual   {actual}"
            )
    return _sha256(protocol_path)


def _label_complete(path: Path, expected: int) -> bool:
    summary_path = ROOT / path / "summary.json"
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text())
    return summary.get("status") == "complete" and int(
        summary.get("items", -1)
    ) == expected


def _training_complete() -> bool:
    metrics_path = ROOT / CHECKPOINT_DIR / "metrics.csv"
    if not (ROOT / CHECKPOINT).is_file() or not metrics_path.is_file():
        return False
    with metrics_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    return len(rows) == 5 and int(rows[-1]["epoch"]) == 5


def _record_for_stage(
    record: dict[str, Any],
    stage: str,
    status: str,
    **values: Any,
) -> None:
    stages = record.setdefault("stages", {})
    stages[stage] = {
        "status": status,
        "updated_at": _now(),
        **values,
    }
    record["updated_at"] = _now()
    _write_record(record)


def _prepare(record: dict[str, Any]) -> None:
    manifest_path = ROOT / CORPUS / "manifest.json"
    if manifest_path.is_file():
        _record_for_stage(
            record,
            "prepare",
            "complete",
            manifest=str(CORPUS / "manifest.json"),
        )
        return
    _record_for_stage(record, "prepare", "running")
    _run(
        [
            sys.executable,
            "scripts/prepare_v16_oracle_corpus.py",
            "--materialize",
            "--selection",
            str(SELECTION),
            "--output-dir",
            str(CORPUS),
        ]
    )
    _record_for_stage(
        record,
        "prepare",
        "complete",
        manifest=str(CORPUS / "manifest.json"),
    )


def _generate_one_label_set(
    *,
    metadata: Path,
    output: Path,
    device: str,
) -> None:
    command = [
        sys.executable,
        "scripts/generate_v16_oracle_labels.py",
        "--metadata",
        str(metadata),
        "--output-dir",
        str(output),
        "--device",
        device,
    ]
    if (ROOT / output / "progress.json").is_file():
        command.append("--resume")
    _run(command)


def _labels(record: dict[str, Any], device: str) -> None:
    if not (ROOT / TRAIN_BASE).is_file() or not (
        ROOT / CALIBRATION_BASE
    ).is_file():
        raise FileNotFoundError(
            "V16 corpus is not materialized; run the prepare stage first"
        )
    if _label_complete(TRAIN_LABELS, 1_440) and _label_complete(
        CALIBRATION_LABELS,
        260,
    ):
        _record_for_stage(record, "labels", "complete")
        return
    _record_for_stage(record, "labels", "running", device=device)
    if not _label_complete(TRAIN_LABELS, 1_440):
        _generate_one_label_set(
            metadata=TRAIN_BASE,
            output=TRAIN_LABELS,
            device=device,
        )
    if not _label_complete(CALIBRATION_LABELS, 260):
        _generate_one_label_set(
            metadata=CALIBRATION_BASE,
            output=CALIBRATION_LABELS,
            device=device,
        )
    _record_for_stage(
        record,
        "labels",
        "complete",
        training_items=1_440,
        calibration_items=260,
    )


def _train(record: dict[str, Any], device: str) -> None:
    if not _label_complete(TRAIN_LABELS, 1_440) or not _label_complete(
        CALIBRATION_LABELS,
        260,
    ):
        raise RuntimeError("V16 oracle labels are incomplete")
    if _training_complete():
        _record_for_stage(
            record,
            "train",
            "complete",
            checkpoint=str(CHECKPOINT),
            checkpoint_sha256=_sha256(ROOT / CHECKPOINT),
        )
        return
    _record_for_stage(record, "train", "running", device=device)
    command = [
        sys.executable,
        "scripts/train_v16_gate.py",
        "--config",
        str(TRAIN_CONFIG),
        "--device",
        device,
    ]
    latest = ROOT / CHECKPOINT_DIR / "latest.pt"
    if latest.is_file():
        command.extend(["--resume", str(CHECKPOINT_DIR / "latest.pt")])
    _run(command)
    if not _training_complete():
        raise RuntimeError("V16 training did not complete all five epochs")
    _record_for_stage(
        record,
        "train",
        "complete",
        checkpoint=str(CHECKPOINT),
        checkpoint_sha256=_sha256(ROOT / CHECKPOINT),
    )


def _evaluate(record: dict[str, Any], device: str) -> None:
    if not _training_complete():
        raise RuntimeError("V16 training is incomplete")
    gate_path = ROOT / OUTPUT_ROOT / "gate/gate_report.json"
    if gate_path.is_file():
        gate = json.loads(gate_path.read_text())
        _record_for_stage(
            record,
            "evaluate",
            "complete",
            decision=gate["decision"],
            gate_summary=gate["gate_summary"],
        )
        return
    _record_for_stage(record, "evaluate", "running", device=device)
    _run(
        [
            sys.executable,
            "scripts/run_v16_evaluation.py",
            "--checkpoint",
            str(CHECKPOINT),
            "--output-root",
            str(OUTPUT_ROOT),
            "--device",
            device,
        ]
    )
    gate = json.loads(gate_path.read_text())
    _record_for_stage(
        record,
        "evaluate",
        "complete",
        decision=gate["decision"],
        gate_summary=gate["gate_summary"],
    )
    record["status"] = "complete"
    record["final_decision"] = gate["decision"]
    record["completed_at"] = _now()
    _write_record(record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(*STAGES, "all"),
        default="all",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    protocol_digest = _verify_protocol()
    selected = STAGES if args.stage == "all" else (args.stage,)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "protocol": str(PROTOCOL),
                    "protocol_sha256": protocol_digest,
                    "selected_stages": selected,
                    "device": args.device,
                },
                indent=2,
            )
        )
        return

    record_path = ROOT / RECORD
    if record_path.is_file():
        record = json.loads(record_path.read_text())
        if record.get("protocol_sha256") != protocol_digest:
            protocol = yaml.safe_load((ROOT / PROTOCOL).read_text())
            amendment = protocol.get("amendment", {})
            previous_digest = record.get("protocol_sha256")
            if (
                previous_digest
                != amendment.get("supersedes_protocol_sha256")
            ):
                raise ValueError(
                    "Existing execution record uses another protocol"
                )
            record.setdefault("protocol_history", []).append(
                {
                    "protocol_sha256": previous_digest,
                    "superseded_at": _now(),
                    "reason": amendment.get("reason"),
                }
            )
            record["protocol_sha256"] = protocol_digest
            record["protocol_amendment_id"] = amendment.get("id")
    else:
        record = {
            "status": "running",
            "started_at": _now(),
            "protocol": str(PROTOCOL),
            "protocol_sha256": protocol_digest,
            "external_test_used": False,
            "stages": {},
        }
        _write_record(record)
    record["status"] = "running"
    for key in ("failed_stage", "error", "failed_at"):
        record.pop(key, None)
    _write_record(record)

    actions = {
        "prepare": lambda: _prepare(record),
        "labels": lambda: _labels(record, args.device),
        "train": lambda: _train(record, args.device),
        "evaluate": lambda: _evaluate(record, args.device),
    }
    try:
        for stage in selected:
            actions[stage]()
    except Exception as error:
        record["status"] = "failed"
        record["failed_stage"] = stage
        record["error"] = repr(error)
        record["failed_at"] = _now()
        _write_record(record)
        raise


if __name__ == "__main__":
    main()
