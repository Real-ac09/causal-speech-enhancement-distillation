#!/usr/bin/env python3
"""Evaluate one fixed V15 checkpoint and apply the frozen development gates."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CROSS_REFERENCE = Path(
    "results/v15/cross_domain_dev/baselines_seed1200/v14_2"
)
VOICE_REFERENCE = Path("results/v14/distillation/winner_dev400")


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    environment["LD_LIBRARY_PATH"] = (
        "/home/mohamedb/miniconda3/envs/cnvqg/lib"
    )
    return environment


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=_environment(), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    cross_output = args.output_root / "cross_domain_dev"
    voice_output = args.output_root / "voicebank_dev400"
    if not (cross_output / "summary.json").is_file():
        _run(
            [
                sys.executable,
                "scripts/evaluate.py",
                "--checkpoint",
                str(args.checkpoint),
                "--metadata",
                "data/processed/dns_cross_domain_dev/metadata.csv",
                "--output-dir",
                str(cross_output),
                "--device",
                args.device,
                "--weights",
                "model",
            ]
        )
    if not (voice_output / "summary.json").is_file():
        _run(
            [
                sys.executable,
                "scripts/evaluate.py",
                "--checkpoint",
                str(args.checkpoint),
                "--metadata",
                (
                    "data/processed/voicebank_demand/metadata/"
                    "v12_architecture_selection_400.csv"
                ),
                "--output-dir",
                str(voice_output),
                "--device",
                args.device,
                "--weights",
                "model",
            ]
        )

    comparison_seed = args.bootstrap_seed
    _run(
        [
            sys.executable,
            "scripts/compare_evaluations.py",
            "--reference",
            f"v14_2={CROSS_REFERENCE}",
            "--candidate",
            f"{args.candidate_name}={cross_output}",
            "--output-dir",
            str(args.output_root / "cross_domain_comparison"),
            "--bootstrap-samples",
            "20000",
            "--seed",
            str(comparison_seed),
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/compare_evaluations.py",
            "--reference",
            f"v14_2={VOICE_REFERENCE}",
            "--candidate",
            f"{args.candidate_name}={voice_output}",
            "--output-dir",
            str(args.output_root / "voicebank_comparison"),
            "--bootstrap-samples",
            "20000",
            "--seed",
            str(comparison_seed + 1),
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/gate_v15_candidate.py",
            "--candidate-name",
            args.candidate_name,
            "--gates",
            "configs/v15/promotion_gates.yaml",
            "--cross-metadata",
            "data/processed/dns_cross_domain_dev/metadata.csv",
            "--cross-reference",
            str(CROSS_REFERENCE),
            "--cross-candidate",
            str(cross_output),
            "--voice-reference",
            str(VOICE_REFERENCE),
            "--voice-candidate",
            str(voice_output),
            "--output-dir",
            str(args.output_root / "gate"),
            "--bootstrap-samples",
            "20000",
            "--bootstrap-seed",
            str(comparison_seed + 2),
        ]
    )


if __name__ == "__main__":
    main()
