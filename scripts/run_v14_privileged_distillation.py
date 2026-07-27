#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/v14/distill_privileged_base.yaml"
GENERATED = ROOT / "configs/v14/generated/distillation"
CHECKPOINTS = ROOT / "checkpoints/v14/distillation"
RESULTS = ROOT / "results/v14/distillation"
TEACHER = ROOT / "checkpoints/auxiliary_gated_tf_mamba_v43_scratch_adaptive/best.pt"
V13 = ROOT / "checkpoints/v12/full/gru_matched_time1_seed1200/best.pt"
SEARCH = ROOT / "data/processed/voicebank_demand/metadata/v12_epoch_selection_100.csv"
DEVELOPMENT = (
    ROOT
    / "data/processed/voicebank_demand/metadata/v12_architecture_selection_400.csv"
)
V13_DEVELOPMENT = ROOT / "results/v14/blend_screen_seed1200/strength_100"
TEACHER_DEVELOPMENT = ROOT / "results/v14/teacher_audit/v43_dev400"
CANDIDATES = {"mag_002": 0.02, "mag_005": 0.05}
METRIC_PAIRS = {
    "enhanced_pesq": "noisy_pesq",
    "enhanced_si_sdr": "noisy_si_sdr",
    "enhanced_stoi": "noisy_stoi",
    "enhanced_estoi": "noisy_estoi",
}


def run(*command: str) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    libraries = [str(Path(sys.executable).parents[1] / "lib")]
    if environment.get("LD_LIBRARY_PATH"):
        libraries.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = ":".join(libraries)
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def python(*arguments: str) -> None:
    run(sys.executable, *arguments)


def summary(directory: Path) -> dict[str, float]:
    raw = json.loads((directory / "summary.json").read_text())
    return {key: float(value) for key, value in raw.get("metrics", raw).items()}


def per_file(directory: Path) -> dict[str, dict[str, float | str]]:
    with (directory / "per_file_metrics.csv").open(newline="") as file:
        return {
            str(row["file_id"]): {
                key: value if key in {"file_id", "speaker_id"} else float(value)
                for key, value in row.items()
            }
            for row in csv.DictReader(file)
        }


def harm_rates(directory: Path) -> dict[str, float]:
    rows = list(per_file(directory).values())
    return {
        metric: sum(
            float(row[metric]) < float(row[noisy])
            for row in rows
        )
        / len(rows)
        for metric, noisy in METRIC_PAIRS.items()
    }


def evaluate(checkpoint: Path, metadata: Path, output: Path) -> None:
    if (output / "summary.json").is_file():
        return
    python(
        "scripts/evaluate.py",
        "--checkpoint",
        str(checkpoint.relative_to(ROOT)),
        "--metadata",
        str(metadata.relative_to(ROOT)),
        "--output-dir",
        str(output.relative_to(ROOT)),
        "--device",
        "cuda",
    )


def audit_teacher() -> None:
    evaluate(TEACHER, DEVELOPMENT, TEACHER_DEVELOPMENT)
    teacher = summary(TEACHER_DEVELOPMENT)
    baseline = summary(V13_DEVELOPMENT)
    required = (
        "enhanced_pesq",
        "enhanced_si_sdr",
        "enhanced_stoi",
        "enhanced_estoi",
    )
    failures = [
        metric for metric in required if teacher[metric] <= baseline[metric]
    ]
    if failures:
        raise RuntimeError(
            "Teacher is ineligible because it does not beat V13 on: "
            + ", ".join(failures)
        )
    print("Teacher eligibility passed:", teacher, flush=True)


def write_configs() -> dict[str, Path]:
    base = yaml.safe_load(BASE_CONFIG.read_text())
    GENERATED.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, weight in CANDIDATES.items():
        config = deepcopy(base)
        config["distillation"]["log_magnitude_weight"] = weight
        config["paths"]["experiment_name"] = name
        path = GENERATED / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        paths[name] = path
    return paths


def train(name: str, config: Path) -> None:
    directory = CHECKPOINTS / name
    final = directory / "epoch_004.pt"
    if final.is_file():
        return
    command = [
        "scripts/train_distill.py",
        "--config",
        str(config.relative_to(ROOT)),
        "--device",
        "cuda",
        "--max-train-batches",
        os.environ.get("V14_DISTILL_BATCHES", "600"),
    ]
    latest = directory / "latest.pt"
    if latest.is_file():
        command.extend(("--resume", str(latest.relative_to(ROOT))))
    python(*command)


def candidate_safe(
    metrics: dict[str, float],
    baseline: dict[str, float],
    candidate_harm: dict[str, float],
    baseline_harm: dict[str, float],
) -> bool:
    return (
        metrics["enhanced_si_sdr"] >= baseline["enhanced_si_sdr"] - 0.10
        and metrics["enhanced_stoi"] >= baseline["enhanced_stoi"] - 0.001
        and metrics["enhanced_estoi"] >= baseline["enhanced_estoi"] - 0.002
        and all(
            candidate_harm[metric] - baseline_harm[metric] <= 0.01 + 1e-12
            for metric in METRIC_PAIRS
        )
    )


def select_search_winner() -> dict[str, object]:
    baseline_directory = RESULTS / "baseline_search100"
    evaluate(V13, SEARCH, baseline_directory)
    baseline = summary(baseline_directory)
    baseline_harm = harm_rates(baseline_directory)
    rows: list[dict[str, object]] = []
    for name in CANDIDATES:
        for epoch in range(1, 5):
            directory = RESULTS / "search100" / name / f"epoch_{epoch:03d}"
            evaluate(CHECKPOINTS / name / f"epoch_{epoch:03d}.pt", SEARCH, directory)
            metrics = summary(directory)
            candidate_harm = harm_rates(directory)
            rows.append(
                {
                    "name": name,
                    "epoch": epoch,
                    "directory": str(directory.relative_to(ROOT)),
                    "safe": candidate_safe(
                        metrics,
                        baseline,
                        candidate_harm,
                        baseline_harm,
                    ),
                    "harm_rates": candidate_harm,
                    **metrics,
                }
            )
    eligible = [row for row in rows if bool(row["safe"])]
    if not eligible:
        raise RuntimeError("No V14.2 search candidate passed the no-harm gates")
    winner = max(eligible, key=lambda row: float(row["enhanced_pesq"]))
    decision = {
        "baseline": baseline,
        "baseline_harm_rates": baseline_harm,
        "winner": winner,
        "search_pesq_gate_passed": (
            float(winner["enhanced_pesq"])
            >= baseline["enhanced_pesq"] + 0.010
        ),
        "candidates": rows,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "search_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )
    print(json.dumps(decision, indent=2), flush=True)
    return winner


def confirm_development(winner: dict[str, object]) -> dict[str, object]:
    name = str(winner["name"])
    epoch = int(winner["epoch"])
    checkpoint = CHECKPOINTS / name / f"epoch_{epoch:03d}.pt"
    output = RESULTS / "winner_dev400"
    evaluate(checkpoint, DEVELOPMENT, output)
    comparison = RESULTS / "winner_vs_v13"
    python(
        "scripts/compare_evaluations.py",
        "--reference",
        "v13=results/v14/blend_screen_seed1200/strength_100",
        "--candidate",
        "v14_distilled=results/v14/distillation/winner_dev400",
        "--output-dir",
        str(comparison.relative_to(ROOT)),
        "--bootstrap-samples",
        "20000",
        "--seed",
        "14201",
    )
    report = json.loads((comparison / "comparison.json").read_text())
    result = report["results"][1]
    interval = report["paired_bootstrap"]["v14_distilled"][
        "enhanced_pesq"
    ]["ci95"]
    baseline_harm = harm_rates(V13_DEVELOPMENT)
    winner_harm = harm_rates(output)
    harm_passed = all(
        winner_harm[metric] - baseline_harm[metric] <= 0.01 + 1e-12
        for metric in METRIC_PAIRS
    )
    promoted = (
        float(result["delta_enhanced_pesq"]) >= 0.010
        and float(interval[0]) > 0.0
        and float(result["delta_enhanced_si_sdr"]) >= -0.10
        and float(result["delta_enhanced_stoi"]) >= -0.001
        and float(result["delta_enhanced_estoi"]) >= -0.002
        and harm_passed
    )
    return {
        "winner": {"name": name, "epoch": epoch, "checkpoint": str(checkpoint.relative_to(ROOT))},
        "development_comparison": result,
        "paired_pesq_ci95": interval,
        "baseline_harm_rates": baseline_harm,
        "winner_harm_rates": winner_harm,
        "harm_rate_gate_passed": harm_passed,
        "promoted": promoted,
        "standard_test_used": False,
    }


def main() -> None:
    for path in (BASE_CONFIG, TEACHER, V13, SEARCH, DEVELOPMENT):
        if not path.is_file():
            raise FileNotFoundError(path)
    audit_teacher()
    configs = write_configs()
    for name, config in configs.items():
        train(name, config)
    winner = select_search_winner()
    status = confirm_development(winner)
    (RESULTS / "programme_status.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
