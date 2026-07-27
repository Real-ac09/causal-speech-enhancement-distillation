#!/usr/bin/env python3
"""Search scale-preserving reconstruction losses on the fixed V8 capacity set."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml

from run_v82_magnitude_overfit import ROOT, SUBSET_METADATA, create_subset


BASE_CONFIG = ROOT / "configs/v8/generated/v82_overfit/magnitude_overfit_16.yaml"
CONFIG_ROOT = ROOT / "configs/v8/generated/v84_scale_search"
CHECKPOINT_ROOT = ROOT / "checkpoints/v84"
RESULT_ROOT = ROOT / "results/v84/scale_loss_search"
CONDA = Path(os.environ.get("CONDA_BIN", str(Path.home() / "miniconda3/bin/conda")))
ENV_NAME = os.environ.get("CNVQG_ENV_NAME", "cnvqg")
ENV_PREFIX = Path(
    os.environ.get("CNVQG_ENV_PREFIX", str(Path.home() / "miniconda3/envs" / ENV_NAME))
)

CANDIDATES = {
    "scale_magnitude": {
        "magnitude_weight": 1.0,
        "magnitude_log_weight": 0.0,
        "si_sdr_weight": 0.0,
        "waveform_l1_weight": 0.0,
    },
    "scale_magnitude_log": {
        "magnitude_weight": 1.0,
        "magnitude_log_weight": 0.25,
        "si_sdr_weight": 0.0,
        "waveform_l1_weight": 0.0,
    },
    "scale_balanced": {
        "magnitude_weight": 1.0,
        "magnitude_log_weight": 0.25,
        "si_sdr_weight": 0.05,
        "waveform_l1_weight": 0.05,
    },
}


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


def candidate_config(name: str, losses: dict[str, float], index: int) -> Path:
    config = yaml.safe_load(BASE_CONFIG.read_text())
    config["project"]["seed"] = 8401 + index
    model = config["model"]
    model["architecture"] = "predictive_noise_vq_mamba_v84"
    model["reconstruction_mode"] = "mixture_consistent_complex"
    model["scale_preserving_detail"] = True
    loss = config["loss"]
    loss.update(losses)
    # Keep phase and all auxiliary objectives disabled: this stage measures
    # magnitude capacity and objective alignment only.
    loss["stft_weight"] = 0.0
    loss["complex_stft_weight"] = 0.0
    loss["vq_weight"] = 0.0
    loss["noise_prediction_weight"] = 0.0
    loss["noise_spectrum_weight"] = 0.0
    loss["phase_weight"] = 0.0
    loss["group_delay_weight"] = 0.0
    loss["instantaneous_frequency_weight"] = 0.0
    loss["phase_confidence_weight"] = 0.0
    training = config["training"]
    training["epochs"] = 60
    training["checkpoint_metric"] = "loss_total"
    training["checkpoint_mode"] = "min"
    training["early_stopping"] = {"enabled": False}
    training["lr_scheduler"] = {"name": "none"}
    config["paths"]["checkpoint_dir"] = "checkpoints/v84"
    config["paths"]["experiment_name"] = name
    path = CONFIG_ROOT / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def evaluate(name: str) -> dict[str, object]:
    output = RESULT_ROOT / name
    summary_path = output / "summary.json"
    if not summary_path.exists():
        python(
            "scripts/evaluate_v5_reconstruction_ablations.py",
            "--checkpoint", str((CHECKPOINT_ROOT / name / "best.pt").relative_to(ROOT)),
            "--metadata", str(SUBSET_METADATA.relative_to(ROOT)),
            "--output-dir", str(output.relative_to(ROOT)),
            "--max-items", "16", "--chunk-seconds", "4.0", "--device", "cuda",
        )
    summary = json.loads(summary_path.read_text())
    fitted = summary["estimated_magnitude_noisy_phase"]
    clean_phase = summary["estimated_magnitude_clean_phase"]
    return {
        "name": name,
        "pesq": fitted["enhanced_pesq"],
        "stoi": fitted["enhanced_stoi"],
        "estoi": fitted["enhanced_estoi"],
        "si_sdr": fitted["enhanced_si_sdr"],
        "clean_phase_pesq": clean_phase["enhanced_pesq"],
    }


def write_status(results: list[dict[str, object]], complete: bool) -> None:
    ranking = sorted(results, key=lambda row: float(row["pesq"]), reverse=True)
    payload = {
        "complete": complete,
        "completed_candidates": len(results),
        "total_candidates": len(CANDIDATES),
        "v82_reference_pesq": 2.573778599500656,
        "capacity_gate_pesq": 2.8,
        "ranking": ranking,
        "winner": ranking[0] if ranking else None,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "status.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


def main() -> None:
    create_subset()
    results: list[dict[str, object]] = []
    for index, (name, losses) in enumerate(CANDIDATES.items()):
        config = candidate_config(name, losses, index)
        checkpoint = CHECKPOINT_ROOT / name / "best.pt"
        if not checkpoint.exists():
            python("scripts/train.py", "--config", str(config.relative_to(ROOT)), "--device", "cuda")
        results.append(evaluate(name))
        write_status(results, complete=False)
    write_status(results, complete=True)


if __name__ == "__main__":
    main()
