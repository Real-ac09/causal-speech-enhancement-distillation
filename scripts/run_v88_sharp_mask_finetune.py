#!/usr/bin/env python3
"""Fine-tune the best ratio model with non-vanishing sharp mask losses."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/v8/generated/v87_continuation/ratio_16_bf16_500.yaml"
INIT_CHECKPOINT = ROOT / "checkpoints/v86/ratio_16_bf16/best.pt"
CONFIG_ROOT = ROOT / "configs/v8/generated/v88_sharp_mask"
CHECKPOINT_ROOT = ROOT / "checkpoints/v88"
RESULT_ROOT = ROOT / "results/v88/sharp_mask_finetune"
METADATA = ROOT / "data/processed/voicebank_demand/metadata/v82_overfit_16.csv"
CONDA = Path(os.environ.get("CONDA_BIN", str(Path.home() / "miniconda3/bin/conda")))
ENV_NAME = os.environ.get("CNVQG_ENV_NAME", "cnvqg")
ENV_PREFIX = Path(
    os.environ.get("CNVQG_ENV_PREFIX", str(Path.home() / "miniconda3/envs" / ENV_NAME))
)
CANDIDATES = {
    "ratio_l1_finetune": "l1",
    "ratio_charbonnier_finetune": "charbonnier",
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


def make_config(name: str, loss_mode: str) -> Path:
    config = yaml.safe_load(BASE_CONFIG.read_text())
    config["project"]["seed"] = 8801
    config["loss"]["magnitude_ratio_loss"] = loss_mode
    config["loss"]["magnitude_ratio_charbonnier_eps"] = 1e-3
    training = config["training"]
    training["epochs"] = 150
    training["learning_rate"] = 3e-5
    training["precision"] = "bf16"
    training["grad_clip_norm"] = 1.0
    training["val_every"] = 5
    training["early_stopping"] = {"enabled": False}
    training["lr_scheduler"] = {"name": "none"}
    training["init_checkpoint"] = str(INIT_CHECKPOINT.relative_to(ROOT))
    training["init_strict"] = True
    config["paths"]["checkpoint_dir"] = "checkpoints/v88"
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
            "--metadata", str(METADATA.relative_to(ROOT)),
            "--output-dir", str(output.relative_to(ROOT)),
            "--max-items", "16", "--chunk-seconds", "4.0", "--device", "cuda",
        )
    fitted = json.loads(summary_path.read_text())["estimated_magnitude_noisy_phase"]
    return {
        "name": name,
        "loss_mode": CANDIDATES[name],
        "pesq": fitted["enhanced_pesq"],
        "stoi": fitted["enhanced_stoi"],
        "estoi": fitted["enhanced_estoi"],
        "si_sdr": fitted["enhanced_si_sdr"],
    }


def write_status(results: list[dict[str, object]], complete: bool) -> None:
    ranking = sorted(results, key=lambda row: float(row["pesq"]), reverse=True)
    status = {
        "complete": complete,
        "completed_candidates": len(results),
        "total_candidates": len(CANDIDATES),
        "initial_checkpoint_pesq": 2.457519382238388,
        "v82_reference_pesq": 2.573778599500656,
        "capacity_gate_pesq": 2.8,
        "ranking": ranking,
        "winner": ranking[0] if ranking else None,
        "capacity_gate": bool(ranking and float(ranking[0]["pesq"]) >= 2.8),
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2), flush=True)


def main() -> None:
    if not INIT_CHECKPOINT.exists():
        raise FileNotFoundError(INIT_CHECKPOINT)
    results = []
    for name, loss_mode in CANDIDATES.items():
        config = make_config(name, loss_mode)
        checkpoint = CHECKPOINT_ROOT / name / "best.pt"
        if not checkpoint.exists():
            python("scripts/train.py", "--config", str(config.relative_to(ROOT)), "--device", "cuda")
        results.append(evaluate(name))
        write_status(results, complete=False)
    write_status(results, complete=True)


if __name__ == "__main__":
    main()
