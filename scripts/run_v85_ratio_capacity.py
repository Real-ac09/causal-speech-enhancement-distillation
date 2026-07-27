#!/usr/bin/env python3
"""Test direct ideal-ratio-mask supervision with and without raw scale features."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml

from run_v82_magnitude_overfit import ROOT, SUBSET_METADATA, create_subset


BASE_CONFIG = ROOT / "configs/v8/generated/v82_overfit/magnitude_overfit_16.yaml"
CONFIG_ROOT = ROOT / "configs/v8/generated/v85_ratio_capacity"
CHECKPOINT_ROOT = ROOT / "checkpoints/v85"
RESULT_ROOT = ROOT / "results/v85/ratio_capacity"
CONDA = Path(os.environ.get("CONDA_BIN", str(Path.home() / "miniconda3/bin/conda")))
ENV_NAME = os.environ.get("CNVQG_ENV_NAME", "cnvqg")
ENV_PREFIX = Path(
    os.environ.get("CNVQG_ENV_PREFIX", str(Path.home() / "miniconda3/envs" / ENV_NAME))
)

CANDIDATES = {
    "direct_ratio": False,
    "direct_ratio_scale": True,
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


def make_config(name: str, scale_detail: bool) -> Path:
    config = yaml.safe_load(BASE_CONFIG.read_text())
    # Hold the seed fixed so this is an architectural comparison.
    config["project"]["seed"] = 8501
    model = config["model"]
    model["architecture"] = "predictive_noise_vq_mamba_v84"
    model["reconstruction_mode"] = "mixture_consistent_complex"
    model["scale_preserving_detail"] = scale_detail
    loss = config["loss"]
    for key in (
        "waveform_l1_weight", "si_sdr_weight", "stft_weight", "vq_weight",
        "mel_weight", "complex_stft_weight", "noise_prediction_weight",
        "noise_spectrum_weight", "magnitude_weight", "magnitude_log_weight",
        "phase_weight", "group_delay_weight", "instantaneous_frequency_weight",
        "phase_confidence_weight", "compute_weight",
    ):
        loss[key] = 0.0
    loss["magnitude_ratio_weight"] = 1.0
    loss["magnitude_ratio_cap"] = 1.0
    loss["magnitude_equal_loudness"] = False
    training = config["training"]
    training["epochs"] = 100
    training["checkpoint_metric"] = "loss_magnitude_ratio"
    training["checkpoint_mode"] = "min"
    training["early_stopping"] = {"enabled": False}
    training["lr_scheduler"] = {"name": "none"}
    config["paths"]["checkpoint_dir"] = "checkpoints/v85"
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
    return {
        "name": name,
        "pesq": fitted["enhanced_pesq"],
        "stoi": fitted["enhanced_stoi"],
        "estoi": fitted["enhanced_estoi"],
        "si_sdr": fitted["enhanced_si_sdr"],
    }


def write_status(results: list[dict[str, object]], complete: bool) -> None:
    ranking = sorted(results, key=lambda row: float(row["pesq"]), reverse=True)
    payload = {
        "complete": complete,
        "completed_candidates": len(results),
        "total_candidates": len(CANDIDATES),
        "oracle_ratio_cap_1_pesq": 3.3544124513864517,
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
    for name, scale_detail in CANDIDATES.items():
        config = make_config(name, scale_detail)
        checkpoint = CHECKPOINT_ROOT / name / "best.pt"
        if not checkpoint.exists():
            python("scripts/train.py", "--config", str(config.relative_to(ROOT)), "--device", "cuda")
        results.append(evaluate(name))
        write_status(results, complete=False)
    write_status(results, complete=True)


if __name__ == "__main__":
    main()
