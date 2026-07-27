#!/usr/bin/env python3
"""Run the controlled V9.2 topology tournament and conditionally combine winners."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("CNVQG_PYTHON", Path.home() / "miniconda3/envs/cnvqg/bin/python"))
CONFIG_DIR = ROOT / "configs/v9/generated/v94_topology_ablation"
CHECKPOINT_DIR = ROOT / "checkpoints/v94_topology_ablation"
RESULT_DIR = ROOT / "results/v9/v94_topology_ablation"
BASELINE_PESQ = 2.143557
MINIMUM_GAIN = 0.02


def configure(name: str, architecture: str, **model_overrides: object) -> dict:
    config = yaml.safe_load((ROOT / "configs/v9/train_v92_one_file_capacity.yaml").read_text())
    config["project"]["seed"] = 9501
    config["model"].update(
        {
            "architecture": architecture,
            "variant": "student",
            "use_mamba": True,
            "use_frequency_mamba": False,
            "use_noise_conditioning": False,
            "use_auxiliary_vq": False,
            "use_phase_detail": False,
            "phase_residual_limit": 0.0,
            **model_overrides,
        }
    )
    config["training"].update(
        {
            "epochs": 500,
            "learning_rate": 0.0001,
            "precision": "bf16",
            "log_every": 25,
            "val_every": 25,
            "checkpoint_metric": "perceptual_enhanced_pesq",
            "checkpoint_mode": "max",
            "optimizer": {"name": "adamw"},
            "lr_scheduler": {"name": "none"},
            "early_stopping": {"enabled": False},
            "perceptual_validation": {"enabled": True, "max_items": 1},
            "ema": {"enabled": False},
            "metricgan_lite": {"enabled": False},
            "save_every_epoch": False,
        }
    )
    for key in ("init_checkpoint", "init_strict", "gradient_accumulation_steps"):
        config["training"].pop(key, None)
    config["paths"] = {
        "checkpoint_dir": str(CHECKPOINT_DIR.relative_to(ROOT)),
        "experiment_name": name,
    }
    return config


def write_config(name: str, config: dict) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def run(name: str, config: dict) -> None:
    if (CHECKPOINT_DIR / name / "best.pt").exists():
        print(f"Skipping completed {name}", flush=True)
        return
    path = write_config(name, config)
    environment = os.environ.copy()
    libraries = [str(PYTHON.parent.parent / "lib"), "/opt/cuda/lib64"]
    if environment.get("LD_LIBRARY_PATH"):
        libraries.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = ":".join(libraries)
    environment["PYTHONPATH"] = "src"
    print(f"Starting {name}", flush=True)
    subprocess.run(
        [str(PYTHON), "scripts/train.py", "--config", str(path.relative_to(ROOT))],
        cwd=ROOT,
        env=environment,
        check=True,
    )


def best_metrics(name: str) -> dict[str, float | int]:
    with (CHECKPOINT_DIR / name / "metrics.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    validated = [row for row in rows if row.get("val_perceptual_enhanced_pesq")]
    best = max(validated, key=lambda row: float(row["val_perceptual_enhanced_pesq"]))
    return {
        "epoch": int(best["epoch"]),
        "pesq": float(best["val_perceptual_enhanced_pesq"]),
        "si_sdr": float(best["val_perceptual_enhanced_si_sdr"]),
        "stoi": float(best["val_perceptual_enhanced_stoi"]),
        "estoi": float(best["val_perceptual_enhanced_estoi"]),
        "magnitude_ratio_l1": float(best["val_loss_magnitude_ratio"]),
    }


def save_summary(summary: dict) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    candidates = {
        "v92_wide_shallow_bf16": configure(
            "v92_wide_shallow_bf16",
            "causal_single_scale_mamba_v92",
            core_channels=232,
            blocks=1,
        ),
        "v94_temporal_detail_bf16": configure(
            "v94_temporal_detail_bf16", "causal_temporal_detail_mamba_v94"
        ),
    }
    summary: dict = {
        "control": {"name": "v92_one_file_capacity_bf16", "pesq": BASELINE_PESQ},
        "promotion_threshold": BASELINE_PESQ + MINIMUM_GAIN,
        "candidates": {},
    }
    for name, config in candidates.items():
        run(name, config)
        metrics = best_metrics(name)
        metrics["promoted"] = metrics["pesq"] >= BASELINE_PESQ + MINIMUM_GAIN
        summary["candidates"][name] = metrics
        save_summary(summary)

    independent_winners = [
        name for name, metrics in summary["candidates"].items() if metrics["promoted"]
    ]
    summary["independent_winners"] = independent_winners
    if len(independent_winners) == 2:
        name = "v94_temporal_detail_wide_shallow_bf16"
        combined = configure(
            name,
            "causal_temporal_detail_mamba_v94",
            core_channels=224,
            blocks=1,
        )
        run(name, combined)
        summary["combined"] = best_metrics(name)
    else:
        summary["combined"] = {
            "skipped": True,
            "reason": "Both isolated topology changes must beat the control by 0.02 PESQ.",
        }
    save_summary(summary)


if __name__ == "__main__":
    main()
