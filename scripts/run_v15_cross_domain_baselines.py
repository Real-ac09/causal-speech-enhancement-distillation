#!/usr/bin/env python3
"""Verify and run the frozen CPU-only V15 cross-domain baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = Path("configs/v15/frozen_cross_domain_dev_protocol.yaml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_digest(relative_path: str, expected: str) -> None:
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"Frozen input changed: {relative_path}\n"
            f"expected {expected}\nactual   {actual}"
        )


def _csv_rows(path: Path) -> int:
    with path.open(newline="") as file:
        reader = csv.reader(file)
        next(reader)
        return sum(1 for _ in reader)


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    python = Path(sys.executable)
    libraries = [str(python.parent.parent / "lib"), "/opt/cuda/lib64"]
    if environment.get("LD_LIBRARY_PATH"):
        libraries.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = ":".join(libraries)
    environment["PYTHONPATH"] = "src"
    return environment


def _complete(directory: Path) -> bool:
    return (
        (directory / "summary.json").is_file()
        and (directory / "per_file_metrics.csv").is_file()
    )


def _run(command: list[str], *, execute: bool, complete: bool) -> None:
    label = "= existing output:" if complete else "+"
    print(label, " ".join(command), flush=True)
    if execute and not complete:
        subprocess.run(command, cwd=ROOT, env=_environment(), check=True)


def _load_protocol() -> tuple[dict[str, Any], str]:
    path = ROOT / PROTOCOL_PATH
    protocol = yaml.safe_load(path.read_text())
    if protocol.get("status") != "frozen":
        raise ValueError("Cross-domain development protocol is not frozen")
    for section in ("dataset",):
        for key in ("metadata", "manifest", "asset_selection"):
            _require_digest(
                protocol[section][key],
                protocol[section][f"{key}_sha256"],
            )
    for model in protocol["models"].values():
        _require_digest(model["checkpoint"], model["checkpoint_sha256"])
    _require_digest(
        protocol["promotion_gates"]["path"],
        protocol["promotion_gates"]["sha256"],
    )
    for source, digest in protocol["source_files"].items():
        _require_digest(source, digest)
    expected = int(protocol["dataset"]["items"])
    if _csv_rows(ROOT / protocol["dataset"]["metadata"]) != expected:
        raise ValueError("Frozen development metadata row count changed")
    manifest = json.loads(
        (ROOT / protocol["dataset"]["manifest"]).read_text()
    )
    if int(manifest["items"]) != expected:
        raise ValueError("Frozen development manifest item count changed")
    for pair in manifest["pair_hashes"]:
        file_id = pair["file_id"]
        for kind in ("clean", "noisy"):
            audio = (
                ROOT
                / protocol["dataset"]["audio_root"]
                / kind
                / f"{file_id}.wav"
            )
            if _sha256(audio) != pair[f"{kind}_sha256"]:
                raise ValueError(f"Frozen development audio changed: {audio}")
    return protocol, _sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    protocol, protocol_digest = _load_protocol()
    metadata = protocol["dataset"]["metadata"]
    output_root = ROOT / protocol["evaluation"]["output_root"]
    directories: dict[str, Path] = {}

    for name, model in protocol["models"].items():
        directory = output_root / name
        directories[name] = directory
        command = [
            sys.executable,
            "scripts/evaluate.py",
            "--checkpoint",
            model["checkpoint"],
            "--metadata",
            metadata,
            "--output-dir",
            str(directory.relative_to(ROOT)),
            "--device",
            "cpu",
            "--weights",
            "model",
        ]
        _run(command, execute=args.execute, complete=_complete(directory))

    comparison = output_root / "v14_2_vs_v13"
    comparison_command = [
        sys.executable,
        "scripts/compare_evaluations.py",
        "--reference",
        f"v13={directories['v13'].relative_to(ROOT)}",
        "--candidate",
        f"v14_2={directories['v14_2'].relative_to(ROOT)}",
        "--output-dir",
        str(comparison.relative_to(ROOT)),
        "--bootstrap-samples",
        str(protocol["evaluation"]["bootstrap_samples"]),
        "--seed",
        str(protocol["evaluation"]["comparison_seed"]),
    ]
    comparison_complete = (comparison / "comparison.json").is_file()
    _run(
        comparison_command,
        execute=args.execute,
        complete=comparison_complete,
    )

    analysis = output_root / "condition_analysis"
    analysis_command = [
        sys.executable,
        "scripts/analyze_v15_cross_domain_dev.py",
        "--metadata",
        metadata,
        "--reference",
        str(directories["v13"].relative_to(ROOT)),
        "--candidate",
        str(directories["v14_2"].relative_to(ROOT)),
        "--output-dir",
        str(analysis.relative_to(ROOT)),
        "--bootstrap-samples",
        str(protocol["evaluation"]["bootstrap_samples"]),
        "--bootstrap-seed",
        str(protocol["evaluation"]["analysis_seed"]),
    ]
    analysis_complete = (analysis / "summary.json").is_file()
    _run(analysis_command, execute=args.execute, complete=analysis_complete)

    if args.execute:
        for directory in directories.values():
            summary = json.loads((directory / "summary.json").read_text())
            if int(summary["num_items"]) != int(protocol["dataset"]["items"]):
                raise ValueError("Baseline evaluation item count changed")
            if summary["weights"] != "model":
                raise ValueError("Baseline evaluation did not use raw weights")
        record = {
            "status": "complete",
            "protocol": str(PROTOCOL_PATH),
            "protocol_sha256": protocol_digest,
            "device": "cpu",
            "outputs": {
                "v13": str(directories["v13"].relative_to(ROOT)),
                "v14_2": str(directories["v14_2"].relative_to(ROOT)),
                "comparison": str(comparison.relative_to(ROOT)),
                "condition_analysis": str(analysis.relative_to(ROOT)),
            },
        }
        (output_root / "execution_record.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
