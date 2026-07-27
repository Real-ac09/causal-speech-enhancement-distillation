#!/usr/bin/env python3
"""Continue the stable BF16 ratio-mask capacity run with staged PESQ gates."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/v8/generated/v86_promoted/ratio_16_bf16.yaml"
CONFIG = ROOT / "configs/v8/generated/v87_continuation/ratio_16_bf16_500.yaml"
CHECKPOINT_DIR = ROOT / "checkpoints/v86/ratio_16_bf16"
RESULT_ROOT = ROOT / "results/v87/bf16_capacity_continuation"
METADATA = ROOT / "data/processed/voicebank_demand/metadata/v82_overfit_16.csv"
CONDA = Path(os.environ.get("CONDA_BIN", str(Path.home() / "miniconda3/bin/conda")))
ENV_NAME = os.environ.get("CNVQG_ENV_NAME", "cnvqg")
ENV_PREFIX = Path(
    os.environ.get("CNVQG_ENV_PREFIX", str(Path.home() / "miniconda3/envs" / ENV_NAME))
)
STAGES = (100, 200, 300, 500)


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
    config = yaml.safe_load(BASE_CONFIG.read_text())
    config["training"]["epochs"] = STAGES[-1]
    config["training"]["val_every"] = 5
    config["training"]["early_stopping"] = {"enabled": False}
    config["training"]["save_every_epoch"] = False
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(yaml.safe_dump(config, sort_keys=False))


def evaluate(stage: int) -> dict[str, float | int]:
    output = RESULT_ROOT / f"epoch_{stage:03d}"
    summary_path = output / "summary.json"
    if not summary_path.exists():
        python(
            "scripts/evaluate_v5_reconstruction_ablations.py",
            "--checkpoint", str((CHECKPOINT_DIR / "best.pt").relative_to(ROOT)),
            "--metadata", str(METADATA.relative_to(ROOT)),
            "--output-dir", str(output.relative_to(ROOT)),
            "--max-items", "16", "--chunk-seconds", "4.0", "--device", "cuda",
        )
    fitted = json.loads(summary_path.read_text())["estimated_magnitude_noisy_phase"]
    checkpoint = torch.load(CHECKPOINT_DIR / "best.pt", map_location="cpu", weights_only=False)
    return {
        "stage_epoch": stage,
        "best_checkpoint_epoch": int(checkpoint["epoch"]),
        "ratio_loss": float(checkpoint["best_val_loss"]),
        "pesq": float(fitted["enhanced_pesq"]),
        "stoi": float(fitted["enhanced_stoi"]),
        "estoi": float(fitted["enhanced_estoi"]),
        "si_sdr": float(fitted["enhanced_si_sdr"]),
    }


def write_status(stages: list[dict[str, float | int]], complete: bool, reason: str) -> None:
    best = max(stages, key=lambda row: float(row["pesq"])) if stages else None
    status = {
        "complete": complete,
        "reason": reason,
        "precision": "bf16",
        "v82_reference_pesq": 2.573778599500656,
        "capacity_gate_pesq": 2.8,
        "capacity_gate": bool(best and float(best["pesq"]) >= 2.8),
        "stages": stages,
        "best_stage": best,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2), flush=True)


def main() -> None:
    create_config()
    latest = CHECKPOINT_DIR / "latest.pt"
    if not latest.exists():
        raise FileNotFoundError(f"Missing continuation checkpoint: {latest}")
    stages: list[dict[str, float | int]] = []
    stop_reason = "maximum_epoch_reached"
    for stage in STAGES:
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        current_epoch = int(checkpoint["epoch"])
        if current_epoch < stage:
            python(
                "scripts/train.py",
                "--config", str(CONFIG.relative_to(ROOT)),
                "--resume", str(latest.relative_to(ROOT)),
                "--epochs", str(stage),
                "--device", "cuda",
            )
        result = evaluate(stage)
        stages.append(result)
        write_status(stages, complete=False, reason="stage_complete")
        if float(result["pesq"]) >= 2.8:
            stop_reason = "capacity_gate_passed"
            break
        if len(stages) >= 3:
            previous = stages[-2]
            older = stages[-3]
            recent_pesq_gain = float(result["pesq"]) - float(previous["pesq"])
            prior_pesq_gain = float(previous["pesq"]) - float(older["pesq"])
            recent_ratio_gain = (
                float(previous["ratio_loss"]) - float(result["ratio_loss"])
            ) / max(1e-8, float(previous["ratio_loss"]))
            if recent_pesq_gain < 0.01 and prior_pesq_gain < 0.01 and recent_ratio_gain < 0.02:
                stop_reason = "multi_stage_plateau"
                break
    write_status(stages, complete=True, reason=stop_reason)


if __name__ == "__main__":
    main()
