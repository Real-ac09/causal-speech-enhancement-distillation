#!/usr/bin/env python3
"""Run the matched V8 control/predictive-VQ learning gate."""
from __future__ import annotations

import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIGS = {
    "control": ROOT / "configs/v8/train_v8_teacher_control.yaml",
    "predictive_vq": ROOT / "configs/v8/train_v8_teacher_predictive_vq.yaml",
}
GENERATED = ROOT / "configs/v8/generated/matched_trial"
CHECKPOINTS = ROOT / "checkpoints/v8/matched_trial"
RESULTS = ROOT / "results/v8/matched_trial"
SEARCH = ROOT / "data/processed/voicebank_demand/metadata/val_v51_search_100.csv"
CONDA = Path(os.environ.get("CONDA_BIN", str(Path.home() / "miniconda3/bin/conda")))
ENV_NAME = os.environ.get("CNVQG_ENV_NAME", "cnvqg")
ENV_PREFIX = Path(
    os.environ.get("CNVQG_ENV_PREFIX", str(Path.home() / "miniconda3/envs" / ENV_NAME))
)
EPOCHS = int(os.environ.get("V8_TRIAL_EPOCHS", "3"))
TRAIN_BATCHES = os.environ.get("V8_TRIAL_TRAIN_BATCHES", "600")


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


def prepare_configs() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    for name, source in SOURCE_CONFIGS.items():
        config = deepcopy(yaml.safe_load(source.read_text()))
        config["training"]["epochs"] = EPOCHS
        config["training"]["max_train_batches"] = int(TRAIN_BATCHES)
        config["training"]["save_every_epoch"] = True
        config["training"]["checkpoint_metric"] = "loss_total"
        config["training"]["checkpoint_mode"] = "min"
        config["training"]["lr_scheduler"] = {"name": "none"}
        config["training"]["early_stopping"] = {"enabled": False}
        config["training"]["perceptual_validation"] = {"enabled": False}
        config["paths"]["checkpoint_dir"] = "checkpoints/v8/matched_trial"
        config["paths"]["experiment_name"] = name
        (GENERATED / f"{name}.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False)
        )


def train_epoch(name: str, epoch: int) -> Path:
    run_dir = CHECKPOINTS / name
    checkpoint = run_dir / f"epoch_{epoch:03d}.pt"
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
        previous = run_dir / f"epoch_{epoch - 1:03d}.pt"
        if not previous.exists():
            raise FileNotFoundError(previous)
        arguments.extend(("--resume", str(previous.relative_to(ROOT))))
    python(*arguments)
    if not checkpoint.exists():
        raise RuntimeError(f"Training did not produce {checkpoint}")
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
    prepare_configs()
    history = []
    for name in SOURCE_CONFIGS:
        for epoch in range(1, EPOCHS + 1):
            checkpoint = train_epoch(name, epoch)
            metrics = evaluate(name, epoch, checkpoint)
            history.append({
                "name": name,
                "epoch": epoch,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                **metrics,
            })
            RESULTS.mkdir(parents=True, exist_ok=True)
            (RESULTS / "status.json").write_text(json.dumps({
                "complete": False,
                "epochs": EPOCHS,
                "train_batches_per_epoch": int(TRAIN_BATCHES),
                "history": history,
            }, indent=2) + "\n")

    winners = {
        name: max(
            (row for row in history if row["name"] == name),
            key=lambda row: row["enhanced_pesq"],
        )
        for name in SOURCE_CONFIGS
    }
    overall = max(winners.values(), key=lambda row: row["enhanced_pesq"])
    predictive_delta = (
        winners["predictive_vq"]["enhanced_pesq"]
        - winners["control"]["enhanced_pesq"]
    )
    status = {
        "complete": True,
        "epochs": EPOCHS,
        "train_batches_per_epoch": int(TRAIN_BATCHES),
        "winners": winners,
        "overall_winner": overall,
        "predictive_vq_pesq_delta": predictive_delta,
        "history": history,
    }
    (RESULTS / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
