#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/v4/train_v45_recovery_search.yaml"
GENERATED = ROOT / "configs/v4/generated/v45"
CHECKPOINTS = ROOT / "checkpoints/v45/search"
RESULTS = ROOT / "results/v45/search"
SEARCH_METADATA = ROOT / "data/processed/voicebank_demand/metadata/val_v51_search_100.csv"
SOURCE = ROOT / "checkpoints/v44/full/causal_v43_recipe_full/best.pt"
CONDA = Path(os.environ.get("CONDA_BIN", str(Path.home() / "miniconda3/bin/conda")))
ENV = os.environ.get("CNVQG_ENV_NAME", "cnvqg")
ENV_PREFIX = Path(os.environ.get("CNVQG_ENV_PREFIX", str(Path.home() / "miniconda3/envs" / ENV)))
SEARCH_BATCHES = os.environ.get("V45_SEARCH_BATCHES", "600")


def run(*command: str) -> None:
    print("+", " ".join(command), flush=True)
    environment = os.environ.copy()
    libraries = [str(ENV_PREFIX / "lib"), "/opt/cuda/lib64"]
    if environment.get("LD_LIBRARY_PATH"):
        libraries.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = ":".join(libraries)
    subprocess.run(command, cwd=ROOT, check=True, env=environment)


def env_python(*args: str) -> None:
    run(str(CONDA), "run", "--no-capture-output", "-n", ENV, "python", *args)


def candidate_configs() -> dict[str, dict]:
    base = yaml.safe_load(BASE_CONFIG.read_text())
    configs: dict[str, dict] = {}

    control = deepcopy(base)
    control["model"].update(
        architecture="causal_aux_vq_mamba_v51",
        refinement_passes=2,
        vq_mode="train_only",
    )
    control["paths"]["experiment_name"] = "control"
    configs["control"] = control

    single = deepcopy(base)
    single["paths"]["experiment_name"] = "single_pass"
    configs["single_pass"] = single

    phase_free = deepcopy(control)
    phase_free["paths"]["experiment_name"] = "phase_free"
    phase_free["loss"].update(
        phase_weight=0.0,
        group_delay_weight=0.0,
        instantaneous_frequency_weight=0.0,
        phase_confidence_weight=0.0,
    )
    configs["phase_free"] = phase_free

    magnitude = deepcopy(phase_free)
    magnitude["paths"]["experiment_name"] = "magnitude_focus"
    magnitude["loss"].update(
        magnitude_weight=1.25,
        complex_stft_weight=0.2,
    )
    configs["magnitude_focus"] = magnitude

    hop128 = deepcopy(control)
    hop128["paths"]["experiment_name"] = "hop128"
    hop128["model"]["hop_length"] = 128
    hop128["loss"].update(
        complex_stft_hop_sizes=[128],
        noise_prediction_hop_length=128,
        tf_detail_hop_length=128,
    )
    configs["hop128"] = hop128
    return configs


def inference_pre_gates() -> dict[str, bool]:
    probe_root = ROOT / "results/v45/inference_probes"
    control = json.loads((probe_root / "control/summary.json").read_text())["metrics"]
    decisions = {"control": True, "phase_free": True, "magnitude_focus": True}
    probe_names = {"single_pass": "pass1", "hop128": "hop128"}
    for name, probe_name in probe_names.items():
        candidate = json.loads(
            (probe_root / probe_name / "summary.json").read_text()
        )["metrics"]
        decisions[name] = (
            candidate["enhanced_pesq"] >= control["enhanced_pesq"] - 0.005
            and candidate["enhanced_si_sdr"] >= control["enhanced_si_sdr"] - 0.15
            and candidate["enhanced_stoi"] >= control["enhanced_stoi"] - 0.002
            and candidate["enhanced_estoi"] >= control["enhanced_estoi"] - 0.003
        )
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "inference_pre_gates.json").write_text(
        json.dumps(decisions, indent=2) + "\n"
    )
    return decisions


def write_configs(configs: dict[str, dict]) -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    for name, config in configs.items():
        (GENERATED / f"{name}.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False)
        )


def train_and_evaluate(name: str) -> None:
    config = GENERATED / f"{name}.yaml"
    checkpoint = CHECKPOINTS / name / "best.pt"
    output = RESULTS / name
    if not checkpoint.exists():
        env_python(
            "scripts/train.py", "--config", str(config.relative_to(ROOT)),
            "--device", "cuda", "--max-train-batches", SEARCH_BATCHES,
        )
    if not (output / "summary.json").exists():
        env_python(
            "scripts/evaluate.py", "--checkpoint", str(checkpoint.relative_to(ROOT)),
            "--metadata", str(SEARCH_METADATA.relative_to(ROOT)),
            "--output-dir", str(output.relative_to(ROOT)), "--device", "cuda",
            "--phase-residual-scale", "0.0",
        )


def metrics(name: str) -> dict[str, float]:
    summary = json.loads((RESULTS / name / "summary.json").read_text())
    return summary["metrics"]


def safe_gain(candidate: dict[str, float], control: dict[str, float], delta: float = 0.005) -> bool:
    return (
        candidate["enhanced_pesq"] >= control["enhanced_pesq"] + delta
        and candidate["enhanced_si_sdr"] >= control["enhanced_si_sdr"] - 0.15
        and candidate["enhanced_stoi"] >= control["enhanced_stoi"] - 0.002
        and candidate["enhanced_estoi"] >= control["enhanced_estoi"] - 0.003
    )


def make_combined(configs: dict[str, dict]) -> bool:
    control_metrics = metrics("control")
    trained = {
        path.parent.name for path in RESULTS.glob("*/summary.json")
    }
    accepted = {
        name: name in trained and safe_gain(metrics(name), control_metrics)
        for name in ("single_pass", "phase_free", "magnitude_focus", "hop128")
    }
    if not any(accepted.values()):
        return False
    combined = deepcopy(configs["control"])
    combined["paths"]["experiment_name"] = "combined"
    if accepted["single_pass"]:
        combined["model"].update(
            architecture="causal_v43_recovery_v45",
            refinement_passes=1,
            vq_mode="disabled",
        )
    objective_source = None
    if accepted["magnitude_focus"]:
        objective_source = "magnitude_focus"
    elif accepted["phase_free"]:
        objective_source = "phase_free"
    if objective_source is not None:
        for key in (
            "magnitude_weight", "complex_stft_weight", "phase_weight",
            "group_delay_weight", "instantaneous_frequency_weight",
            "phase_confidence_weight", "noise_prediction_weight",
        ):
            combined["loss"][key] = configs[objective_source]["loss"][key]
    if accepted["hop128"]:
        combined["model"]["hop_length"] = 128
        combined["loss"].update(
            complex_stft_hop_sizes=[128],
            noise_prediction_hop_length=128,
            tf_detail_hop_length=128,
        )
    (GENERATED / "combined.yaml").write_text(yaml.safe_dump(combined, sort_keys=False))
    (RESULTS / "accepted_factors.json").parent.mkdir(parents=True, exist_ok=True)
    (RESULTS / "accepted_factors.json").write_text(
        json.dumps(accepted, indent=2) + "\n"
    )
    return True


def choose_winner(names: list[str]) -> tuple[str, dict[str, dict[str, float]]]:
    rows = {name: metrics(name) for name in names}
    control = rows["control"]
    eligible = [
        name for name, row in rows.items()
        if name == "control" or (
            row["enhanced_si_sdr"] >= control["enhanced_si_sdr"] - 0.15
            and row["enhanced_stoi"] >= control["enhanced_stoi"] - 0.002
            and row["enhanced_estoi"] >= control["enhanced_estoi"] - 0.003
        )
    ]
    winner = max(eligible, key=lambda name: rows[name]["enhanced_pesq"])
    report = {"winner": winner, "eligible": eligible, "candidates": rows}
    (RESULTS / "decision.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return winner, rows


def prepare_scratch(winner: str, winner_metrics: dict[str, float]) -> bool:
    selected = yaml.safe_load((GENERATED / f"{winner}.yaml").read_text())
    selected["project"]["seed"] = 4502
    selected["data"]["val_metadata"] = "data/processed/voicebank_demand/metadata/val_v5_locked_400.csv"
    selected["data"]["full_val_metadata"] = "data/processed/voicebank_demand/metadata/val.csv"
    selected["training"].pop("init_checkpoint", None)
    selected["training"].pop("init_strict", None)
    selected["training"]["epochs"] = 40
    selected["training"]["optimizer"]["muon_learning_rate"] = 0.001
    selected["training"]["optimizer"]["adamw_learning_rate"] = 0.0001
    selected["training"]["learning_rate"] = 0.0001
    selected["training"]["lr_scheduler"] = {
        "name": "reduce_on_plateau", "mode": "max",
        "metric": "perceptual_enhanced_pesq", "factor": 0.5,
        "patience": 3, "min_lr": 1.0e-5,
    }
    selected["training"]["early_stopping"] = {
        "enabled": True, "patience": 8, "min_delta": 0.003,
    }
    selected["training"]["perceptual_validation"]["max_items"] = 400
    selected["paths"]["checkpoint_dir"] = "checkpoints/v45/full"
    selected["paths"]["experiment_name"] = "promoted_scratch"
    scratch = ROOT / "configs/v4/generated/v45/promoted_scratch.yaml"
    scratch.write_text(yaml.safe_dump(selected, sort_keys=False))

    source_metrics = json.loads(
        (ROOT / "results/v44/full/locked400/summary.json").read_text()
    )["metrics"]
    # Search and locked-400 sets differ, so only use this gate to prevent an
    # obviously non-improving continuation from triggering a long scratch run.
    return winner != "control" and winner_metrics["enhanced_pesq"] >= metrics("control")["enhanced_pesq"] + 0.01


def run_full() -> None:
    run("bash", "scripts/run_v45_full_scratch.sh")


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    configs = candidate_configs()
    write_configs(configs)
    pre_gates = inference_pre_gates()
    names = [name for name in configs if pre_gates.get(name, False)]
    for name in names:
        train_and_evaluate(name)
    if make_combined(configs):
        train_and_evaluate("combined")
        names.append("combined")
    winner, rows = choose_winner(names)
    approved = prepare_scratch(winner, rows[winner])
    status = {
        "winner": winner,
        "scratch_gate_passed": approved,
        "full_requested": os.environ.get("V45_RUN_FULL", "0") == "1",
    }
    (RESULTS / "programme_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    if approved and os.environ.get("V45_RUN_FULL", "0") == "1":
        run_full()


if __name__ == "__main__":
    main()
