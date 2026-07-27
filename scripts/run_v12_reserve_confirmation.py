#!/usr/bin/env python3
"""Prepare or execute the untouched V12 internal-reserve confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESERVE_METADATA = Path(
    "data/processed/voicebank_demand/metadata/v12_internal_reserve.csv"
)
VALIDATION_MANIFEST = Path(
    "data/processed/voicebank_demand/metadata/v12_validation_manifest.json"
)
VARIANTS = ("mamba_control", "gru_matched_time1")
SEEDS = (1200, 1201, 1202)


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    python = Path(sys.executable)
    libraries = [str(python.parent.parent / "lib"), "/opt/cuda/lib64"]
    if environment.get("LD_LIBRARY_PATH"):
        libraries.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = ":".join(libraries)
    environment["PYTHONPATH"] = "src"
    return environment


def _digest(frame: pd.DataFrame) -> str:
    identifiers = "\n".join(sorted(frame["file_id"].astype(str))) + "\n"
    return hashlib.sha256(identifiers.encode()).hexdigest()


def _verify_reserve() -> None:
    manifest = json.loads((ROOT / VALIDATION_MANIFEST).read_text())
    expected = manifest["splits"]["v12_internal_reserve"]
    reserve = pd.read_csv(ROOT / RESERVE_METADATA)
    if not reserve["file_id"].is_unique:
        raise ValueError("Reserve file IDs must be unique")
    if len(reserve) != int(expected["rows"]):
        raise ValueError(
            f"Reserve row count changed: expected {expected['rows']}, got {len(reserve)}"
        )
    actual_digest = _digest(reserve)
    if actual_digest != expected["file_id_sha256"]:
        raise ValueError(
            "Reserve membership digest changed: "
            f"expected {expected['file_id_sha256']}, got {actual_digest}"
        )


def _run(command: list[str], execute: bool) -> None:
    print("+", " ".join(command), flush=True)
    if execute:
        subprocess.run(command, cwd=ROOT, env=_environment(), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confirm V12 models on the untouched 437-file reserve."
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="Defaults to CPU so VRAM use must be explicitly requested.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Without this flag, verify inputs and print commands only.",
    )
    args = parser.parse_args()

    _verify_reserve()
    evaluations: dict[tuple[str, int], Path] = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            experiment = f"{variant}_seed{seed}"
            checkpoint = ROOT / f"checkpoints/v12/full/{experiment}/best.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            evaluation = (
                ROOT
                / f"results/v12/reserve/{experiment}/internal_reserve_437"
            )
            evaluations[(variant, seed)] = evaluation
            _run(
                [
                    sys.executable,
                    "scripts/evaluate.py",
                    "--checkpoint",
                    str(checkpoint.relative_to(ROOT)),
                    "--metadata",
                    str(RESERVE_METADATA),
                    "--output-dir",
                    str(evaluation.relative_to(ROOT)),
                    "--device",
                    args.device,
                ],
                args.execute and not (evaluation / "summary.json").exists(),
            )

        comparison = ROOT / f"results/v12/reserve/comparison_seed{seed}"
        _run(
            [
                sys.executable,
                "scripts/compare_evaluations.py",
                "--reference",
                f"mamba_control={evaluations[('mamba_control', seed)]}",
                "--candidate",
                f"gru_matched_time1={evaluations[('gru_matched_time1', seed)]}",
                "--output-dir",
                str(comparison),
            ],
            args.execute and not (comparison / "comparison.json").exists(),
        )

    aggregate = ROOT / "results/v12/reserve/aggregate_three_seed/comparison.json"
    command = [
        sys.executable,
        "scripts/aggregate_v12_seeds.py",
    ]
    for seed in SEEDS:
        command.extend(
            [
                "--reference",
                str(evaluations[("mamba_control", seed)].relative_to(ROOT)),
            ]
        )
    for seed in SEEDS:
        command.extend(
            [
                "--candidate",
                str(evaluations[("gru_matched_time1", seed)].relative_to(ROOT)),
            ]
        )
    command.extend(
        [
            "--output",
            str(aggregate.relative_to(ROOT)),
        ]
    )
    _run(command, args.execute and not aggregate.exists())


if __name__ == "__main__":
    main()
