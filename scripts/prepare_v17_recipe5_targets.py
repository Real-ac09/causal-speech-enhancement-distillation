#!/usr/bin/env python3
"""Pivot V17 strength sweeps into Recipe-5 utility/safety supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
METRICS = ("pesq", "si_sdr", "stoi", "estoi")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _soft_policy(
    group: pd.DataFrame,
    feasible_temperature: float,
    fallback_temperature: float,
    fallback_utility_weight: float,
) -> np.ndarray:
    feasible = group["feasible"].astype(bool).to_numpy()
    utility = group["utility"].to_numpy(float)
    violation = group["constraint_violation_normalised"].to_numpy(float)
    if feasible.any():
        logits = np.full(len(group), -np.inf)
        logits[feasible] = utility[feasible] / feasible_temperature
    else:
        logits = (
            -violation / fallback_temperature
            + fallback_utility_weight * utility
        )
    logits = logits - np.max(logits)
    probability = np.exp(logits)
    return probability / probability.sum()


def _prepare(
    metadata_path: Path,
    candidates_path: Path,
    output_path: Path,
    feasible_temperature: float,
    fallback_temperature: float,
    fallback_utility_weight: float,
) -> dict:
    metadata = pd.read_csv(metadata_path)
    candidates = pd.read_csv(candidates_path)
    sizes = candidates.groupby("file_id").size()
    if len(sizes) != len(metadata) or not bool((sizes == len(GRID)).all()):
        raise ValueError("Every Recipe-5 item must have five candidates")
    rows = []
    for file_id, group in candidates.groupby("file_id", sort=False):
        group = group.sort_values("strength").reset_index(drop=True)
        if not np.allclose(group["strength"].to_numpy(float), GRID):
            raise ValueError(f"Unexpected strength grid for {file_id}")
        probability = _soft_policy(
            group,
            feasible_temperature,
            fallback_temperature,
            fallback_utility_weight,
        )
        row: dict[str, float | int | str] = {"file_id": str(file_id)}
        for index, candidate in group.iterrows():
            row[f"r5_utility_{index}"] = float(candidate["utility"])
            row[f"r5_violation_{index}"] = float(
                candidate["constraint_violation_normalised"]
            )
            row[f"r5_feasible_{index}"] = int(bool(candidate["feasible"]))
            row[f"r5_policy_probability_{index}"] = float(
                probability[index]
            )
            available = set(str(candidate["available_metrics"]).split(";"))
            for metric in METRICS:
                enhanced = candidate[f"enhanced_{metric}"]
                noisy = candidate[f"noisy_{metric}"]
                is_available = (
                    metric in available
                    and pd.notna(enhanced)
                    and pd.notna(noisy)
                )
                row[f"r5_{metric}_available_{index}"] = int(is_available)
                row[f"r5_{metric}_delta_{index}"] = (
                    float(enhanced - noisy) if is_available else 0.0
                )
        rows.append(row)
    targets = pd.DataFrame(rows)
    prepared = metadata.merge(targets, on="file_id", validate="one_to_one")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    probabilities = prepared[
        [f"r5_policy_probability_{index}" for index in range(len(GRID))]
    ].to_numpy(float)
    hard = probabilities.argmax(axis=1)
    oracle = prepared["oracle_class"].to_numpy(int)
    return {
        "metadata": str(metadata_path.relative_to(ROOT)),
        "metadata_sha256": _sha256(metadata_path),
        "candidates": str(candidates_path.relative_to(ROOT)),
        "candidates_sha256": _sha256(candidates_path),
        "output": str(output_path.relative_to(ROOT)),
        "items": int(len(prepared)),
        "columns": int(len(prepared.columns)),
        "probability_sum_max_error": float(
            np.abs(probabilities.sum(axis=1) - 1.0).max()
        ),
        "soft_policy_argmax_oracle_agreement": float(
            np.mean(hard == oracle)
        ),
        "mean_policy_entropy": float(
            np.mean(
                -np.sum(
                    probabilities
                    * np.log(np.clip(probabilities, 1e-12, None)),
                    axis=1,
                )
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/v17_recipe5"),
    )
    parser.add_argument("--feasible-temperature", type=float, default=0.25)
    parser.add_argument("--fallback-temperature", type=float, default=0.25)
    parser.add_argument("--fallback-utility-weight", type=float, default=0.01)
    args = parser.parse_args()
    if args.feasible_temperature <= 0 or args.fallback_temperature <= 0:
        parser.error("Policy temperatures must be positive")
    output_dir = ROOT / args.output_dir
    report = {
        "status": "complete",
        "role": "privileged_training_target_preparation",
        "strength_grid": list(GRID),
        "metric_order": list(METRICS),
        "feasible_temperature": args.feasible_temperature,
        "fallback_temperature": args.fallback_temperature,
        "fallback_utility_weight": args.fallback_utility_weight,
        "development_set_used": False,
        "external_test_used": False,
        "splits": {},
    }
    for split in ("train", "calibration"):
        source = (
            ROOT / "results/v17/recipe2_local_oracles" / split
        )
        report["splits"][split] = _prepare(
            source / "metadata.csv",
            source / "strength_candidates.csv",
            output_dir / f"{split}.csv",
            args.feasible_temperature,
            args.fallback_temperature,
            args.fallback_utility_weight,
        )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
