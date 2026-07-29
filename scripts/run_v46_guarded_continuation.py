#!/usr/bin/env python3
"""Continue V4.6 one epoch at a time, selecting on externally measured PESQ."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v4/distill_v46_privileged_continue.yaml"
RUN_DIR = ROOT / "checkpoints/v46/continuation_mag_005"
RESULT_DIR = ROOT / "results/v46/continuation"
SEARCH = ROOT / "data/processed/voicebank_demand/metadata/val_v51_search_100.csv"
LOCKED = ROOT / "data/processed/voicebank_demand/metadata/val_v5_locked_400.csv"
START_CHECKPOINT = ROOT / "checkpoints/v46/search/mag_005/epoch_004.pt"
START_METRICS = ROOT / "results/v46/search/mag_005/epoch_004/summary.json"
CONDA = Path(os.environ.get("CONDA_BIN", str(Path.home() / "miniconda3/bin/conda")))
ENV_NAME = os.environ.get("CNVQG_ENV_NAME", "cnvqg")
ENV_PREFIX = Path(os.environ.get("CNVQG_ENV_PREFIX", str(Path.home() / "miniconda3/envs" / ENV_NAME)))
MAX_EPOCH = int(os.environ.get("V46_CONTINUE_MAX_EPOCH", "12"))
TRAIN_BATCHES = os.environ.get("V46_CONTINUE_TRAIN_BATCHES", "600")
PATIENCE = int(os.environ.get("V46_CONTINUE_PATIENCE", "3"))
MIN_DELTA = float(os.environ.get("V46_CONTINUE_MIN_DELTA", "0.003"))


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


def metrics(path: Path) -> dict[str, float]:
    return json.loads(path.read_text())["metrics"]


def safe(candidate: dict[str, float], reference: dict[str, float]) -> bool:
    return (
        candidate["enhanced_si_sdr"] >= reference["enhanced_si_sdr"] - 0.15
        and candidate["enhanced_stoi"] >= reference["enhanced_stoi"] - 0.002
        and candidate["enhanced_estoi"] >= reference["enhanced_estoi"] - 0.003
    )


def train_one_epoch(resume: Path, epoch: int, reset_optimizer: bool) -> Path:
    checkpoint = RUN_DIR / f"epoch_{epoch:03d}.pt"
    if checkpoint.exists():
        return checkpoint
    arguments = [
        "scripts/train_distill.py", "--config", str(CONFIG.relative_to(ROOT)),
        "--resume", str(resume.relative_to(ROOT)), "--epochs", str(epoch),
        "--device", "cuda", "--max-train-batches", TRAIN_BATCHES,
    ]
    if reset_optimizer:
        arguments.append("--reset-optimizer")
    python(*arguments)
    if not checkpoint.exists():
        raise RuntimeError(f"Training did not produce {checkpoint}")
    return checkpoint


def evaluate(checkpoint: Path, epoch: int) -> dict[str, float]:
    output = RESULT_DIR / f"epoch_{epoch:03d}"
    summary = output / "summary.json"
    if not summary.exists():
        python(
            "scripts/evaluate.py", "--checkpoint", str(checkpoint.relative_to(ROOT)),
            "--metadata", str(SEARCH.relative_to(ROOT)),
            "--output-dir", str(output.relative_to(ROOT)), "--device", "cuda",
            "--phase-residual-scale", "0.0",
        )
    return metrics(summary)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    reference = metrics(START_METRICS)
    best = {"epoch": 4, "checkpoint": str(START_CHECKPOINT.relative_to(ROOT)), **reference}
    history: list[dict] = []
    stale = 0
    resume = START_CHECKPOINT

    existing = sorted(RUN_DIR.glob("epoch_*.pt"))
    if existing:
        resume = existing[-1]

    first_epoch = int(resume.stem.split("_")[-1]) + 1 if resume.parent == RUN_DIR else 5
    for epoch in range(5, first_epoch):
        checkpoint = RUN_DIR / f"epoch_{epoch:03d}.pt"
        candidate = evaluate(checkpoint, epoch)
        eligible = safe(candidate, reference)
        improved = eligible and candidate["enhanced_pesq"] >= best["enhanced_pesq"] + MIN_DELTA
        if improved:
            best = {"epoch": epoch, "checkpoint": str(checkpoint.relative_to(ROOT)), **candidate}
            stale = 0
        else:
            stale += 1
        history.append({"epoch": epoch, "safe": eligible, "improved": improved, **candidate})

    next_epochs = range(first_epoch, MAX_EPOCH + 1) if stale < PATIENCE else ()
    for epoch in next_epochs:
        checkpoint = train_one_epoch(resume, epoch, reset_optimizer=(epoch == 5))
        resume = checkpoint
        candidate = evaluate(checkpoint, epoch)
        eligible = safe(candidate, reference)
        improved = eligible and candidate["enhanced_pesq"] >= best["enhanced_pesq"] + MIN_DELTA
        if improved:
            best = {"epoch": epoch, "checkpoint": str(checkpoint.relative_to(ROOT)), **candidate}
            stale = 0
        else:
            stale += 1
        history.append({"epoch": epoch, "safe": eligible, "improved": improved, **candidate})
        status = {"complete": False, "stopped_early": False, "stale_epochs": stale,
                  "patience": PATIENCE, "minimum_pesq_delta": MIN_DELTA,
                  "reference_epoch": 4, "best": best, "history": history}
        (RESULT_DIR / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        if stale >= PATIENCE:
            break

    improved_over_start = best["epoch"] != 4
    locked_output = RESULT_DIR / "locked400_winner"
    if improved_over_start and not (locked_output / "summary.json").exists():
        python(
            "scripts/evaluate.py", "--checkpoint", best["checkpoint"],
            "--metadata", str(LOCKED.relative_to(ROOT)),
            "--output-dir", str(locked_output.relative_to(ROOT)), "--device", "cuda",
            "--phase-residual-scale", "0.0",
        )
    final = {"complete": True, "stopped_early": stale >= PATIENCE,
             "stale_epochs": stale, "patience": PATIENCE,
             "minimum_pesq_delta": MIN_DELTA, "reference_epoch": 4,
             "improved_over_start": improved_over_start, "best": best, "history": history}
    (RESULT_DIR / "status.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
