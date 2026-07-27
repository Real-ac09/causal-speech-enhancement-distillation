#!/usr/bin/env python3
"""Gate the direct scalar mask on one utterance, then on the fixed 16 files."""
from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import torch
import yaml

from run_v82_magnitude_overfit import ROOT, SUBSET_METADATA, create_subset


BASE_CONFIG = ROOT / "configs/v8/generated/v85_ratio_capacity/direct_ratio.yaml"
ONE_METADATA = ROOT / "data/processed/voicebank_demand/metadata/v89_overfit_1.csv"
CONFIG_ROOT = ROOT / "configs/v8/generated/v89_direct_scalar"
CHECKPOINT_ROOT = ROOT / "checkpoints/v89"
RESULT_ROOT = ROOT / "results/v89/direct_scalar_capacity"
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


def create_one_metadata() -> None:
    create_subset()
    with SUBSET_METADATA.open(newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)
        fieldnames = reader.fieldnames
    if fieldnames is None:
        raise RuntimeError("Subset metadata has no header")
    with ONE_METADATA.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def make_config(name: str, metadata: Path, precision: str, batch_size: int) -> Path:
    config = yaml.safe_load(BASE_CONFIG.read_text())
    config["project"]["seed"] = 8901
    config["data"]["train_metadata"] = str(metadata.relative_to(ROOT))
    config["data"]["val_metadata"] = str(metadata.relative_to(ROOT))
    config["data"]["batch_size"] = batch_size
    config["data"]["num_workers"] = 0 if batch_size == 1 else 2
    config["model"]["architecture"] = "predictive_noise_vq_mamba_v89"
    config["model"]["reconstruction_mode"] = "direct_scalar_mask"
    config["model"]["scale_preserving_detail"] = False
    config["loss"]["magnitude_ratio_loss"] = "l1"
    config["loss"]["magnitude_ratio_charbonnier_eps"] = 1e-3
    training = config["training"]
    training["epochs"] = 500
    training["learning_rate"] = 1e-4
    training["precision"] = precision
    training["grad_clip_norm"] = 1.0
    training["val_every"] = 25 if batch_size == 1 else 5
    training["checkpoint_metric"] = "loss_magnitude_ratio"
    training["checkpoint_mode"] = "min"
    training["early_stopping"] = {"enabled": False}
    training["lr_scheduler"] = {"name": "none"}
    training.pop("init_checkpoint", None)
    training.pop("init_strict", None)
    config["paths"]["checkpoint_dir"] = "checkpoints/v89"
    config["paths"]["experiment_name"] = name
    path = CONFIG_ROOT / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def evaluate(name: str, metadata: Path, count: int) -> dict[str, float]:
    output = RESULT_ROOT / name
    summary_path = output / "summary.json"
    if not summary_path.exists():
        python(
            "scripts/evaluate_v5_reconstruction_ablations.py",
            "--checkpoint", str((CHECKPOINT_ROOT / name / "best.pt").relative_to(ROOT)),
            "--metadata", str(metadata.relative_to(ROOT)),
            "--output-dir", str(output.relative_to(ROOT)),
            "--max-items", str(count), "--chunk-seconds", "4.0", "--device", "cuda",
        )
    fitted = json.loads(summary_path.read_text())["estimated_magnitude_noisy_phase"]
    checkpoint = torch.load(CHECKPOINT_ROOT / name / "best.pt", map_location="cpu", weights_only=False)
    return {
        "best_epoch": int(checkpoint["epoch"]),
        "mask_l1": float(checkpoint["best_val_loss"]),
        "pesq": float(fitted["enhanced_pesq"]),
        "stoi": float(fitted["enhanced_stoi"]),
        "estoi": float(fitted["enhanced_estoi"]),
        "si_sdr": float(fitted["enhanced_si_sdr"]),
    }


def write_status(payload: dict[str, object]) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "status.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


def main() -> None:
    create_one_metadata()
    micro_name = "direct_scalar_one_fp32"
    micro_config = make_config(micro_name, ONE_METADATA, "fp32", 1)
    if not (CHECKPOINT_ROOT / micro_name / "best.pt").exists():
        python("scripts/train.py", "--config", str(micro_config.relative_to(ROOT)), "--device", "cuda")
    micro = evaluate(micro_name, ONE_METADATA, 1)
    micro_pass = micro["mask_l1"] < 0.02 and micro["pesq"] >= 2.8
    status: dict[str, object] = {
        "complete": not micro_pass,
        "stage": "micro_complete",
        "micro": micro,
        "micro_gate": micro_pass,
        "capacity_gate_pesq": 2.8,
        "sixteen_file": None,
    }
    write_status(status)
    if not micro_pass:
        return

    full_name = "direct_scalar_16_bf16"
    full_config = make_config(full_name, SUBSET_METADATA, "bf16", 4)
    if not (CHECKPOINT_ROOT / full_name / "best.pt").exists():
        python("scripts/train.py", "--config", str(full_config.relative_to(ROOT)), "--device", "cuda")
    full = evaluate(full_name, SUBSET_METADATA, 16)
    status.update(
        {
            "complete": True,
            "stage": "sixteen_file_complete",
            "sixteen_file": full,
            "capacity_gate": full["pesq"] >= 2.8,
            "v82_reference_pesq": 2.573778599500656,
        }
    )
    write_status(status)


if __name__ == "__main__":
    main()
