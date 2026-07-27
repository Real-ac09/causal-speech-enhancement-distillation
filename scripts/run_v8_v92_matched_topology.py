#!/usr/bin/env python3
"""Matched one-file precision/topology comparison for V8 and V9.2."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("CNVQG_PYTHON", str(Path.home() / "miniconda3/envs/cnvqg/bin/python")))
CONFIG_DIR = ROOT / "configs/v9/generated/v8_v92_matched"
CHECKPOINT_DIR = ROOT / "checkpoints/v9_topology"
RESULT_DIR = ROOT / "results/v9/v8_v92_matched"


def common_training(config: dict, name: str, precision: str) -> dict:
    config["project"]["seed"] = 9401
    config["data"].update(
        {
            "train_metadata": "data/processed/voicebank_demand/metadata/v89_overfit_1.csv",
            "val_metadata": "data/processed/voicebank_demand/metadata/v89_overfit_1.csv",
            "chunk_seconds": 4.0,
            "batch_size": 1,
            "num_workers": 0,
            "train_random_crop": False,
        }
    )
    config["loss"].update(
        {
            "waveform_l1_weight": 0.0,
            "si_sdr_weight": 0.0,
            "stft_weight": 0.0,
            "vq_weight": 0.0,
            "mel_weight": 0.0,
            "complex_stft_weight": 0.0,
            "noise_prediction_weight": 0.0,
            "noise_spectrum_weight": 0.0,
            "magnitude_weight": 0.0,
            "magnitude_log_weight": 0.0,
            "magnitude_ratio_weight": 1.0,
            "magnitude_ratio_cap": 1.0,
            "magnitude_ratio_loss": "l1",
            "phase_weight": 0.0,
            "group_delay_weight": 0.0,
            "instantaneous_frequency_weight": 0.0,
            "phase_confidence_weight": 0.0,
            "compute_weight": 0.0,
            "tf_detail_n_fft": 512,
            "tf_detail_hop_length": 160,
            "tf_detail_win_length": 320,
            "tf_detail_center": False,
            "tf_detail_magnitude_power": 0.3,
        }
    )
    training = config["training"]
    training.update(
        {
            "epochs": 500,
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
            "grad_clip_norm": 1.0,
            "precision": precision,
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
        training.pop(key, None)
    config["paths"] = {
        "checkpoint_dir": "checkpoints/v9_topology",
        "experiment_name": name,
    }
    return config


def make_configs() -> list[tuple[str, Path]]:
    v92_base = yaml.safe_load((ROOT / "configs/v9/train_v92_one_file_capacity.yaml").read_text())
    v8_base = yaml.safe_load(
        (ROOT / "configs/v8/generated/v89_direct_scalar/direct_scalar_one_fp32.yaml").read_text()
    )
    specifications = []

    v92 = common_training(v92_base, "v92_student_fp32", "fp32")
    specifications.append(("v92_student_fp32", v92))

    for precision in ("fp32", "bf16"):
        config = yaml.safe_load(yaml.safe_dump(v8_base))
        name = f"v8_student_{precision}"
        config = common_training(config, name, precision)
        model = config["model"]
        model.update(
            {
                "architecture": "predictive_noise_vq_mamba_v8",
                "variant": "student",
                "auxiliary_vq": False,
                "reconstruction_mode": "direct_scalar_mask",
                "scale_preserving_detail": False,
                "phase_residual_scale": 0.0,
                "use_mamba": True,
            }
        )
        for key in ("channels", "noise_dim", "temporal_layers"):
            model.pop(key, None)
        specifications.append((name, config))

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, config in specifications:
        path = CONFIG_DIR / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        paths.append((name, path))
    return paths


def run_config(name: str, path: Path) -> None:
    checkpoint = CHECKPOINT_DIR / name / "best.pt"
    if checkpoint.exists():
        print(f"Skipping completed {name}", flush=True)
        return
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


def summarize(names: list[str]) -> dict[str, dict[str, float | int]]:
    summary = {}
    for name in names:
        metrics_path = CHECKPOINT_DIR / name / "metrics.csv"
        if not metrics_path.exists():
            continue
        with metrics_path.open(newline="") as file:
            rows = list(csv.DictReader(file))
        validated = [row for row in rows if row.get("val_perceptual_enhanced_pesq")]
        if not validated:
            continue
        best = max(validated, key=lambda row: float(row["val_perceptual_enhanced_pesq"]))
        summary[name] = {
            "epoch": int(best["epoch"]),
            "pesq": float(best["val_perceptual_enhanced_pesq"]),
            "si_sdr": float(best["val_perceptual_enhanced_si_sdr"]),
            "stoi": float(best["val_perceptual_enhanced_stoi"]),
            "estoi": float(best["val_perceptual_enhanced_estoi"]),
            "magnitude_ratio_l1": float(best["val_loss_magnitude_ratio"]),
        }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    configs = make_configs()
    for name, path in configs:
        run_config(name, path)
        summarize([item[0] for item in configs])


if __name__ == "__main__":
    main()
