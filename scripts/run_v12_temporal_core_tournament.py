#!/usr/bin/env python3
"""Prepare or execute the controlled V12 temporal-core tournament."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/v8/train_v8_direct_scalar_full.yaml"
EPOCH_METADATA = Path(
    "data/processed/voicebank_demand/metadata/v12_epoch_selection_100.csv"
)
ARCHITECTURE_METADATA = Path(
    "data/processed/voicebank_demand/metadata/v12_architecture_selection_400.csv"
)
VARIANTS: dict[str, dict[str, object]] = {
    "mamba_control": {
        "use_mamba": True,
        "temporal_core": "mamba",
        "time_kernel_size": 3,
    },
    "gru_matched": {
        "use_mamba": False,
        "temporal_core": "gru",
        "temporal_hidden_dim": 232,
        "time_kernel_size": 3,
    },
    "gru_matched_time1": {
        "use_mamba": False,
        "temporal_core": "gru",
        "temporal_hidden_dim": 232,
        "time_kernel_size": 1,
    },
    "gru128_time1": {
        "use_mamba": False,
        "temporal_core": "gru",
        "temporal_hidden_dim": 128,
        "time_kernel_size": 1,
    },
}


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    python = Path(sys.executable)
    libraries = [str(python.parent.parent / "lib"), "/opt/cuda/lib64"]
    if environment.get("LD_LIBRARY_PATH"):
        libraries.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = ":".join(libraries)
    environment["PYTHONPATH"] = "src"
    return environment


def _configuration(
    base: dict[str, object],
    *,
    stage: str,
    variant: str,
    seed: int,
) -> dict[str, object]:
    config = deepcopy(base)
    epochs = 12 if stage == "screen" else 40
    experiment = f"{variant}_seed{seed}"
    config["project"]["seed"] = seed
    config["data"]["val_metadata"] = str(EPOCH_METADATA)
    config["model"].update(
        {
            "architecture": "causal_temporal_core_v12",
            **VARIANTS[variant],
        }
    )
    config["training"].update(
        {
            "epochs": epochs,
            "val_every": 1,
            "checkpoint_metric": "perceptual_enhanced_pesq",
            "checkpoint_mode": "max",
        }
    )
    config["training"]["perceptual_validation"] = {
        "enabled": True,
        "max_items": 100,
        "whole_utterance": True,
    }
    if stage == "screen":
        config["training"]["early_stopping"] = {"enabled": False}
        config["training"]["lr_scheduler"] = {
            "name": "warmup_cosine",
            "warmup_epochs": 2,
            "minimum_factor": 0.05,
        }
    config["paths"] = {
        "checkpoint_dir": f"checkpoints/v12/{stage}",
        "experiment_name": experiment,
    }
    return config


def _write_config(config: dict[str, object], stage: str, experiment: str) -> Path:
    path = ROOT / f"configs/v12/generated/{stage}/{experiment}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _run(command: list[str], execute: bool) -> None:
    print("+", " ".join(command), flush=True)
    if execute:
        subprocess.run(command, cwd=ROOT, env=_environment(), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["screen", "full"], default="screen")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=list(VARIANTS),
        help="For full training, pass only candidates promoted by the screen.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1200])
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Without this flag, only write configs and print the commands.",
    )
    args = parser.parse_args()
    if args.stage == "screen" and len(args.seeds) != 1:
        parser.error("The screen uses one common seed; use multiple seeds only for full runs")

    for metadata in (ROOT / EPOCH_METADATA, ROOT / ARCHITECTURE_METADATA):
        if not metadata.exists():
            raise FileNotFoundError(
                f"{metadata} is missing; run scripts/create_v12_validation_splits.py"
            )

    base = yaml.safe_load(BASE_CONFIG.read_text())
    evaluations: dict[tuple[str, int], Path] = {}
    for seed in args.seeds:
        for variant in args.variants:
            experiment = f"{variant}_seed{seed}"
            config = _configuration(
                base, stage=args.stage, variant=variant, seed=seed
            )
            config_path = _write_config(config, args.stage, experiment)
            checkpoint = (
                ROOT / config["paths"]["checkpoint_dir"] / experiment / "best.pt"
            )
            evaluation = (
                ROOT
                / f"results/v12/{args.stage}/{experiment}/architecture_selection_400"
            )
            evaluations[(variant, seed)] = evaluation
            _run(
                [
                    sys.executable,
                    "scripts/train.py",
                    "--config",
                    str(config_path.relative_to(ROOT)),
                ],
                args.execute and not checkpoint.exists(),
            )
            _run(
                [
                    sys.executable,
                    "scripts/evaluate.py",
                    "--checkpoint",
                    str(checkpoint.relative_to(ROOT)),
                    "--metadata",
                    str(ARCHITECTURE_METADATA),
                    "--output-dir",
                    str(evaluation.relative_to(ROOT)),
                    "--device",
                    "auto",
                ],
                args.execute and not (evaluation / "summary.json").exists(),
            )

    if "mamba_control" in args.variants:
        for seed in args.seeds:
            candidates = [
                variant for variant in args.variants if variant != "mamba_control"
            ]
            if not candidates:
                continue
            command = [
                sys.executable,
                "scripts/compare_evaluations.py",
                "--reference",
                f"mamba_control={evaluations[('mamba_control', seed)]}",
            ]
            for variant in candidates:
                command.extend(
                    [
                        "--candidate",
                        f"{variant}={evaluations[(variant, seed)]}",
                    ]
                )
            command.extend(
                [
                    "--output-dir",
                    str(
                        ROOT
                        / f"results/v12/{args.stage}/comparison_seed{seed}"
                    ),
                ]
            )
            _run(command, args.execute)


if __name__ == "__main__":
    main()
