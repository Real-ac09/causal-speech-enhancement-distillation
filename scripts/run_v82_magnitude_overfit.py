#!/usr/bin/env python3
"""Train V8 on 16 fixed examples to test magnitude representational capacity."""
from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_METADATA = ROOT / "data/processed/voicebank_demand/metadata/train.csv"
SUBSET_METADATA = ROOT / "data/processed/voicebank_demand/metadata/v82_overfit_16.csv"
BASE_CONFIG = ROOT / "configs/v8/train_v8_teacher_predictive_vq.yaml"
CONFIG = ROOT / "configs/v8/generated/v82_overfit/magnitude_overfit_16.yaml"
CHECKPOINT_DIR = ROOT / "checkpoints/v82/magnitude_overfit_16"
RESULT_DIR = ROOT / "results/v82/magnitude_overfit_16"
CONDA = Path(os.environ.get("CONDA_BIN", str(Path.home() / "miniconda3/bin/conda")))
ENV_NAME = os.environ.get("CNVQG_ENV_NAME", "cnvqg")
ENV_PREFIX = Path(
    os.environ.get("CNVQG_ENV_PREFIX", str(Path.home() / "miniconda3/envs" / ENV_NAME))
)


def run(*command: str) -> None:
    environment = os.environ.copy()
    libraries = [str(ENV_PREFIX / "lib"), "/opt/cuda/lib64"]
    if environment.get("LD_LIBRARY_PATH"):
        libraries.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = ":".join(libraries)
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def python(*arguments: str) -> None:
    run(str(CONDA), "run", "--no-capture-output", "-n", ENV_NAME, "python", *arguments)


def create_subset() -> None:
    with SOURCE_METADATA.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        selected = []
        speakers = set()
        for row in reader:
            speaker = row.get("speaker_id", "unknown")
            if speaker in speakers:
                continue
            selected.append(row)
            speakers.add(speaker)
            if len(selected) == 16:
                break
    if fieldnames is None or len(selected) != 16:
        raise RuntimeError("Could not select 16 distinct training speakers")
    SUBSET_METADATA.parent.mkdir(parents=True, exist_ok=True)
    with SUBSET_METADATA.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)


def create_config() -> None:
    config = yaml.safe_load(BASE_CONFIG.read_text())
    config["project"]["seed"] = 8201
    data = config["data"]
    data["train_metadata"] = str(SUBSET_METADATA.relative_to(ROOT))
    data["val_metadata"] = str(SUBSET_METADATA.relative_to(ROOT))
    data["chunk_seconds"] = 4.0
    data["train_random_crop"] = False
    data["batch_size"] = 4
    data["num_workers"] = 2
    model = config["model"]
    model["auxiliary_vq"] = False
    model["phase_residual_scale"] = 0.0
    loss = config["loss"]
    loss["waveform_l1_weight"] = 0.0
    loss["si_sdr_weight"] = 0.0
    loss["stft_weight"] = 0.0
    loss["vq_weight"] = 0.0
    loss["mel_weight"] = 0.0
    loss["complex_stft_weight"] = 0.0
    loss["noise_prediction_weight"] = 0.0
    loss["noise_spectrum_weight"] = 0.0
    loss["magnitude_weight"] = 1.0
    loss["magnitude_equal_loudness"] = False
    loss["phase_weight"] = 0.0
    loss["group_delay_weight"] = 0.0
    loss["instantaneous_frequency_weight"] = 0.0
    loss["phase_confidence_weight"] = 0.0
    training = config["training"]
    training["epochs"] = 200
    training["learning_rate"] = 0.0003
    training["gradient_accumulation_steps"] = 1
    training["checkpoint_metric"] = "loss_magnitude"
    training["checkpoint_mode"] = "min"
    training["save_every_epoch"] = False
    training["val_every"] = 5
    training["lr_scheduler"] = {
        "name": "reduce_on_plateau", "mode": "min", "metric": "loss_magnitude",
        "factor": 0.5, "patience": 3, "min_lr": 0.00001,
    }
    training["early_stopping"] = {"enabled": True, "patience": 12, "min_delta": 0.00001}
    training["perceptual_validation"] = {"enabled": False}
    config["paths"]["checkpoint_dir"] = "checkpoints/v82"
    config["paths"]["experiment_name"] = "magnitude_overfit_16"
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(yaml.safe_dump(config, sort_keys=False))


def main() -> None:
    create_subset()
    create_config()
    if not (CHECKPOINT_DIR / "best.pt").exists():
        python("scripts/train.py", "--config", str(CONFIG.relative_to(ROOT)), "--device", "cuda")
    reconstruction = RESULT_DIR / "reconstruction"
    if not (reconstruction / "summary.json").exists():
        python(
            "scripts/evaluate_v5_reconstruction_ablations.py",
            "--checkpoint", str((CHECKPOINT_DIR / "best.pt").relative_to(ROOT)),
            "--metadata", str(SUBSET_METADATA.relative_to(ROOT)),
            "--output-dir", str(reconstruction.relative_to(ROOT)),
            "--max-items", "16", "--chunk-seconds", "4.0", "--device", "cuda",
        )
    summary = json.loads((reconstruction / "summary.json").read_text())
    fitted = summary["estimated_magnitude_noisy_phase"]
    ceiling = summary["clean_magnitude_noisy_phase"]
    status = {
        "complete": True,
        "checkpoint": str((CHECKPOINT_DIR / "best.pt").relative_to(ROOT)),
        "fitted": fitted,
        "oracle_ceiling": ceiling,
        "pesq_fraction_of_oracle_gain": (
            (fitted["enhanced_pesq"] - fitted["noisy_pesq"])
            / max(1e-8, ceiling["enhanced_pesq"] - ceiling["noisy_pesq"])
        ),
        "capacity_gate": fitted["enhanced_pesq"] >= 2.8,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
