#!/usr/bin/env python3
"""Capacity-test the V8.3 hybrid magnitude decoder on 16 fixed examples."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml

from run_v82_magnitude_overfit import ROOT, SUBSET_METADATA, create_subset


BASE_CONFIG = ROOT / "configs/v8/generated/v82_overfit/magnitude_overfit_16.yaml"
CONFIG = ROOT / "configs/v8/generated/v83_overfit/hybrid_magnitude_overfit_16.yaml"
CHECKPOINT_DIR = ROOT / "checkpoints/v83/hybrid_magnitude_overfit_16"
RESULT_DIR = ROOT / "results/v83/hybrid_magnitude_overfit_16"
V82_STATUS = ROOT / "results/v82/magnitude_overfit_16/status.json"
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


def create_config() -> None:
    if not BASE_CONFIG.exists():
        raise RuntimeError("Run scripts/run_v82_magnitude_overfit.py first to create its baseline config")
    config = yaml.safe_load(BASE_CONFIG.read_text())
    config["project"]["seed"] = 8301
    config["model"]["architecture"] = "predictive_noise_vq_mamba_v83"
    config["model"]["reconstruction_mode"] = "hybrid_magnitude_residual"
    config["model"]["magnitude_log_gain_bound"] = 4.0
    config["model"]["magnitude_residual_bound"] = 2.0
    config["paths"]["checkpoint_dir"] = "checkpoints/v83"
    config["paths"]["experiment_name"] = "hybrid_magnitude_overfit_16"
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
    baseline = json.loads(V82_STATUS.read_text())["fitted"] if V82_STATUS.exists() else None
    status = {
        "complete": True,
        "checkpoint": str((CHECKPOINT_DIR / "best.pt").relative_to(ROOT)),
        "fitted": fitted,
        "oracle_ceiling": ceiling,
        "pesq_fraction_of_oracle_gain": (
            (fitted["enhanced_pesq"] - fitted["noisy_pesq"])
            / max(1e-8, ceiling["enhanced_pesq"] - ceiling["noisy_pesq"])
        ),
        "v82_pesq": None if baseline is None else baseline["enhanced_pesq"],
        "pesq_change_from_v82": (
            None if baseline is None else fitted["enhanced_pesq"] - baseline["enhanced_pesq"]
        ),
        "capacity_gate": fitted["enhanced_pesq"] >= 2.8,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
