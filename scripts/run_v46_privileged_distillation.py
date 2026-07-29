#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/v4/distill_v46_privileged_search.yaml"
GENERATED = ROOT / "configs/v4/generated/v46"
CHECKPOINTS = ROOT / "checkpoints/v46/search"
RESULTS = ROOT / "results/v46/search"
LOCKED = ROOT / "data/processed/voicebank_demand/metadata/val_v5_locked_400.csv"
SEARCH = ROOT / "data/processed/voicebank_demand/metadata/val_v51_search_100.csv"
CONDA = Path(os.environ.get("CONDA_BIN", str(Path.home() / "miniconda3/bin/conda")))
ENV_NAME = os.environ.get("CNVQG_ENV_NAME", "cnvqg")
ENV_PREFIX = Path(
    os.environ.get("CNVQG_ENV_PREFIX", str(Path.home() / "miniconda3/envs" / ENV_NAME))
)
SEARCH_BATCHES = os.environ.get("V46_SEARCH_BATCHES", "600")


def run(*command: str) -> None:
    environment = os.environ.copy()
    libraries = [str(ENV_PREFIX / "lib"), "/opt/cuda/lib64"]
    if environment.get("LD_LIBRARY_PATH"):
        libraries.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = ":".join(libraries)
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def python(*arguments: str) -> None:
    run(
        str(CONDA), "run", "--no-capture-output", "-n", ENV_NAME,
        "python", *arguments,
    )


def configs() -> dict[str, dict]:
    base = yaml.safe_load(BASE.read_text())
    candidates = {}
    for name, weight in (("mag_002", 0.02), ("mag_005", 0.05)):
        config = deepcopy(base)
        config["distillation"]["log_magnitude_weight"] = weight
        config["paths"]["experiment_name"] = name
        candidates[name] = config
    return candidates


def write_configs(candidates: dict[str, dict]) -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    for name, config in candidates.items():
        (GENERATED / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


def train(name: str) -> None:
    final_epoch = CHECKPOINTS / name / "epoch_004.pt"
    if final_epoch.exists():
        return
    python(
        "scripts/train_distill.py",
        "--config", str((GENERATED / f"{name}.yaml").relative_to(ROOT)),
        "--device", "cuda",
        "--max-train-batches", SEARCH_BATCHES,
    )


def evaluate_epoch(name: str, epoch: int) -> None:
    output = RESULTS / name / f"epoch_{epoch:03d}"
    if (output / "summary.json").exists():
        return
    checkpoint = CHECKPOINTS / name / f"epoch_{epoch:03d}.pt"
    python(
        "scripts/evaluate.py",
        "--checkpoint", str(checkpoint.relative_to(ROOT)),
        "--metadata", str(SEARCH.relative_to(ROOT)),
        "--output-dir", str(output.relative_to(ROOT)),
        "--device", "cuda",
        "--phase-residual-scale", "0.0",
    )


def read_metrics(directory: Path) -> dict[str, float]:
    return json.loads((directory / "summary.json").read_text())["metrics"]


def safeguards(metrics: dict[str, float], baseline: dict[str, float]) -> bool:
    return (
        metrics["enhanced_si_sdr"] >= baseline["enhanced_si_sdr"] - 0.15
        and metrics["enhanced_stoi"] >= baseline["enhanced_stoi"] - 0.002
        and metrics["enhanced_estoi"] >= baseline["enhanced_estoi"] - 0.003
    )


def select(candidates: dict[str, dict]) -> tuple[str, int, dict[str, float], bool]:
    baseline_dir = ROOT / "results/v45/inference_probes/control"
    baseline = read_metrics(baseline_dir)
    rows = []
    for name in candidates:
        for epoch in range(1, 5):
            directory = RESULTS / name / f"epoch_{epoch:03d}"
            metrics = read_metrics(directory)
            rows.append({
                "name": name,
                "epoch": epoch,
                "directory": str(directory.relative_to(ROOT)),
                "safe": safeguards(metrics, baseline),
                **metrics,
            })
    eligible = [row for row in rows if row["safe"]]
    if not eligible:
        raise RuntimeError("No V4.6 candidate passed the metric safeguards")
    winner = max(eligible, key=lambda row: row["enhanced_pesq"])
    search_passed = (
        winner["enhanced_pesq"] >= baseline["enhanced_pesq"] + 0.01
    )
    report = {
        "baseline": baseline,
        "winner": winner,
        "search_gate_passed": search_passed,
        "candidates": rows,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "decision.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return winner["name"], int(winner["epoch"]), winner, search_passed


def locked_validation(name: str, epoch: int) -> bool:
    checkpoint = CHECKPOINTS / name / f"epoch_{epoch:03d}.pt"
    output = ROOT / "results/v46/locked400/winner"
    if not (output / "summary.json").exists():
        python(
            "scripts/evaluate.py",
            "--checkpoint", str(checkpoint.relative_to(ROOT)),
            "--metadata", str(LOCKED.relative_to(ROOT)),
            "--output-dir", str(output.relative_to(ROOT)),
            "--device", "cuda", "--phase-residual-scale", "0.0",
        )
    python(
        "scripts/compare_evaluations.py",
        "--reference", "v44_epoch17=results/v44/full/locked400",
        "--candidate", "v46_privileged=results/v46/locked400/winner",
        "--output-dir", "results/v46/locked400/v44_vs_v46",
        "--bootstrap-samples", "20000",
    )
    comparison = json.loads(
        (ROOT / "results/v46/locked400/v44_vs_v46/comparison.json").read_text()
    )
    candidate = comparison["results"][1]
    pesq_interval = comparison["paired_bootstrap"]["v46_privileged"][
        "enhanced_pesq"
    ]["ci95"]
    return (
        candidate["delta_enhanced_pesq"] >= 0.01
        and pesq_interval[0] > 0.0
        and candidate["delta_enhanced_si_sdr"] >= -0.15
        and candidate["delta_enhanced_stoi"] >= -0.002
        and candidate["delta_enhanced_estoi"] >= -0.003
    )


def main() -> None:
    candidates = configs()
    write_configs(candidates)
    for name in candidates:
        train(name)
        for epoch in range(1, 5):
            evaluate_epoch(name, epoch)
    name, epoch, winner, search_passed = select(candidates)
    locked_passed = locked_validation(name, epoch)
    status = {
        "winner": name,
        "epoch": epoch,
        "search_gate_passed": search_passed,
        "locked_validation_run": True,
        "locked_gate_passed": locked_passed,
    }
    (RESULTS / "programme_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
