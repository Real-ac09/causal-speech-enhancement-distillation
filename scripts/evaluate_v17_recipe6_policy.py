#!/usr/bin/env python3
"""Evaluate the frozen Recipe-6b posterior safety policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_v17_recipe5 import GRID, THRESHOLDS, _summarise


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "results/v17/recipe6_selection/selected_predictions.csv"
        ),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/v17_recipe5/calibration.csv"),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path(
            "results/v17/recipe2_local_oracles/calibration/"
            "strength_candidates.csv"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "configs/v17/frozen_recipe6b_policy_protocol.yaml"
        ),
    )
    parser.add_argument("--class-penalty", type=float, default=0.02)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/v17/recipe6b_policy_selection"),
    )
    args = parser.parse_args()
    if args.class_penalty != 0.02:
        raise ValueError("Frozen Recipe-6b requires class_penalty=0.02")

    predictions = pd.read_csv(ROOT / args.predictions)
    probability_columns = [
        f"probability_{level}" for level in range(len(GRID))
    ]
    probability = predictions[probability_columns].to_numpy(float)
    selection_score = np.log(np.clip(probability, 1e-9, None))
    selection_score -= (
        args.class_penalty * np.arange(len(GRID))[None, :]
    )
    policy_predictions = predictions[
        ["file_id", "target_class", "predicted_strength"]
    ].copy()
    policy_predictions["predicted_class"] = selection_score.argmax(axis=1)
    candidates = pd.read_csv(ROOT / args.candidates)
    labels = pd.read_csv(ROOT / args.metadata)
    result = _summarise(policy_predictions, candidates, labels)
    selected = result.pop("predictions")
    passed = result["passes_recipe5_gate"]
    decision = (
        "advance_recipe6b_to_frozen_development_benchmarks"
        if passed
        else "stop_recipe6b_and_build_prefix_aligned_supervision"
    )
    report = {
        "status": "complete",
        "protocol": str(args.protocol),
        "class_penalty": args.class_penalty,
        "thresholds": THRESHOLDS,
        "result": result,
        "decision": decision,
    }
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_dir / "selected_predictions.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
