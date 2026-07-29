#!/usr/bin/env python3
"""Matched V8.1 correction trial for magnitude balance and noise supervision."""
from __future__ import annotations

import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/v8/train_v8_teacher_predictive_vq.yaml"
GENERATED = ROOT / "configs/v8/generated/v81_correction"
CHECKPOINTS = ROOT / "checkpoints/v81/correction"
RESULTS = ROOT / "results/v81/correction"
SEARCH = ROOT / "data/processed/voicebank_demand/metadata/val_v51_search_100.csv"
CONDA = Path(os.environ.get("CONDA_BIN", str(Path.home() / "miniconda3/bin/conda")))
ENV_NAME = os.environ.get("CNVQG_ENV_NAME", "cnvqg")
ENV_PREFIX = Path(
    os.environ.get("CNVQG_ENV_PREFIX", str(Path.home() / "miniconda3/envs" / ENV_NAME))
)
EPOCHS = int(os.environ.get("V81_TRIAL_EPOCHS", "2"))
TRAIN_BATCHES = os.environ.get("V81_TRIAL_TRAIN_BATCHES", "300")


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


def prepare_configs() -> tuple[str, ...]:
    base = yaml.safe_load(BASE.read_text())
    names = ("magnitude_balance", "magnitude_balance_noise")
    GENERATED.mkdir(parents=True, exist_ok=True)
    for name in names:
        config = deepcopy(base)
        config["project"]["seed"] = 8101
        config["data"]["batch_size"] = 6
        model = config["model"]
        model["phase_residual_scale"] = 0.0
        model["posterior_temperature"] = 10.0
        model["prediction_label_smoothing"] = 0.02
        model["vq_usage_weight"] = 0.02
        loss = config["loss"]
        loss["si_sdr_weight"] = 0.03
        loss["complex_stft_weight"] = 0.5
        loss["magnitude_weight"] = 1.0
        loss["noise_spectrum_weight"] = (
            0.1 if name == "magnitude_balance_noise" else 0.0
        )
        loss["noise_spectrum_compression_power"] = 0.3
        training = config["training"]
        training["epochs"] = EPOCHS
        training["max_train_batches"] = int(TRAIN_BATCHES)
        training["gradient_accumulation_steps"] = 1
        training["save_every_epoch"] = True
        training["checkpoint_metric"] = "loss_total"
        training["checkpoint_mode"] = "min"
        training["lr_scheduler"] = {"name": "none"}
        training["early_stopping"] = {"enabled": False}
        training["perceptual_validation"] = {"enabled": False}
        config["paths"]["checkpoint_dir"] = "checkpoints/v81/correction"
        config["paths"]["experiment_name"] = name
        (GENERATED / f"{name}.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False)
        )
    return names


def train_epoch(name: str, epoch: int) -> Path:
    directory = CHECKPOINTS / name
    checkpoint = directory / f"epoch_{epoch:03d}.pt"
    if checkpoint.exists():
        return checkpoint
    arguments = [
        "scripts/train.py",
        "--config", str((GENERATED / f"{name}.yaml").relative_to(ROOT)),
        "--device", "cuda",
        "--epochs", str(epoch),
        "--max-train-batches", TRAIN_BATCHES,
    ]
    if epoch > 1:
        previous = directory / f"epoch_{epoch - 1:03d}.pt"
        arguments.extend(("--resume", str(previous.relative_to(ROOT))))
    python(*arguments)
    if not checkpoint.exists():
        raise RuntimeError(f"Missing checkpoint after training: {checkpoint}")
    return checkpoint


def evaluate(name: str, epoch: int, checkpoint: Path) -> dict[str, float]:
    output = RESULTS / name / f"epoch_{epoch:03d}"
    summary = output / "summary.json"
    if not summary.exists():
        python(
            "scripts/evaluate.py",
            "--checkpoint", str(checkpoint.relative_to(ROOT)),
            "--metadata", str(SEARCH.relative_to(ROOT)),
            "--output-dir", str(output.relative_to(ROOT)),
            "--device", "cuda",
        )
    return json.loads(summary.read_text())["metrics"]


def main() -> None:
    names = prepare_configs()
    history = []
    for name in names:
        for epoch in range(1, EPOCHS + 1):
            checkpoint = train_epoch(name, epoch)
            metrics = evaluate(name, epoch, checkpoint)
            history.append({"name": name, "epoch": epoch,
                            "checkpoint": str(checkpoint.relative_to(ROOT)), **metrics})
            RESULTS.mkdir(parents=True, exist_ok=True)
            (RESULTS / "status.json").write_text(json.dumps({
                "complete": False, "history": history
            }, indent=2) + "\n")

    winners = {
        name: max((row for row in history if row["name"] == name),
                  key=lambda row: row["enhanced_pesq"])
        for name in names
    }
    overall = max(winners.values(), key=lambda row: row["enhanced_pesq"])
    reference_pesq = 1.8684708881378174
    status = {
        "complete": True,
        "reference_v8_pesq": reference_pesq,
        "winners": winners,
        "overall_winner": overall,
        "delta_vs_v8": overall["enhanced_pesq"] - reference_pesq,
        "full_training_gate": overall["enhanced_pesq"] >= reference_pesq + 0.02,
        "history": history,
    }
    (RESULTS / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
