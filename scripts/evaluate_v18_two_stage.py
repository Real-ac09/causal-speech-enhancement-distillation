#!/usr/bin/env python3
"""Evaluate Recipe 8 on training-domain calibration data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_v17_recipe5 import (
    ROOT,
    THRESHOLDS,
    _loader,
    _predictions,
    _summarise,
)


STAGE_THRESHOLDS = {
    "full_route_accuracy_minimum": 0.80,
    "reduced_route_recall_minimum": 0.75,
    "reduced_macro_accuracy_minimum": 0.30,
}


def _add_stage_checks(result: dict) -> dict:
    selected = result["predictions"]
    target = selected["target_class"].to_numpy(int)
    predicted = selected["predicted_class"].to_numpy(int)
    target_full = target == 4
    predicted_full = predicted == 4
    reduced_recalls = [
        float(np.mean(predicted[target == level] == level))
        for level in range(4)
    ]
    metrics = result["metrics"]
    metrics.update(
        full_route_accuracy=float(
            np.mean(target_full == predicted_full)
        ),
        full_route_recall=float(
            np.mean(predicted_full[target_full])
        ),
        reduced_route_recall=float(
            np.mean(~predicted_full[~target_full])
        ),
        reduced_macro_accuracy=float(np.mean(reduced_recalls)),
        reduced_per_class_recall={
            str(level): reduced_recalls[level] for level in range(4)
        },
    )
    stage_checks = {
        "full_route_accuracy": (
            metrics["full_route_accuracy"]
            >= STAGE_THRESHOLDS["full_route_accuracy_minimum"]
        ),
        "reduced_route_recall": (
            metrics["reduced_route_recall"]
            >= STAGE_THRESHOLDS["reduced_route_recall_minimum"]
        ),
        "reduced_macro_accuracy": (
            metrics["reduced_macro_accuracy"]
            >= STAGE_THRESHOLDS["reduced_macro_accuracy_minimum"]
        ),
    }
    result["checks"].update(stage_checks)
    result["passed_checks"] = int(sum(result["checks"].values()))
    result["total_checks"] = len(result["checks"])
    result["passes_recipe8_gate"] = bool(all(result["checks"].values()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/v17_recipe7/calibration.csv"),
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
        "--output-dir",
        type=Path,
        default=Path("results/v18/recipe8_selection"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    checkpoints = (
        [ROOT / args.checkpoint]
        if args.checkpoint
        else sorted((ROOT / args.checkpoint_dir).glob("epoch_*.pt"))
    )
    if not checkpoints:
        raise FileNotFoundError("No Recipe-8 checkpoints found")
    metadata = ROOT / args.metadata
    candidates = pd.read_csv(ROOT / args.candidates)
    labels = pd.read_csv(metadata)
    loader = _loader(metadata, args.batch_size, args.num_workers)
    summaries = []
    stored_predictions = {}
    for checkpoint in checkpoints:
        predictions = _predictions(
            checkpoint,
            loader,
            torch.device(args.device),
        )
        result = _add_stage_checks(
            _summarise(predictions, candidates, labels)
        )
        stored_predictions[str(checkpoint)] = result.pop("predictions")
        result["checkpoint"] = str(checkpoint.relative_to(ROOT))
        summaries.append(result)
    summaries.sort(
        key=lambda item: (
            item["passes_recipe8_gate"],
            item["passed_checks"],
            -item["metrics"]["avoidable_violation_rate"],
            item["metrics"]["reduced_macro_accuracy"],
            item["metrics"]["mean_utility"],
        ),
        reverse=True,
    )
    selected = summaries[0]
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stored_predictions[str(ROOT / selected["checkpoint"])].to_csv(
        output_dir / "selected_predictions.csv",
        index=False,
    )
    decision = (
        "advance_recipe8_to_independent_confirmation"
        if selected["passes_recipe8_gate"]
        else "stop_recipe8_and_report_two_stage_limit"
    )
    report = {
        "status": "complete",
        "data_scope": "training_domain_calibration_only",
        "development_set_used": False,
        "external_test_used": False,
        "quality_thresholds": THRESHOLDS,
        "stage_thresholds": STAGE_THRESHOLDS,
        "recipe7b_reference": {
            "full_route_accuracy": 0.8040752351097179,
            "reduced_route_recall": 0.7191358024691358,
            "reduced_macro_accuracy": 0.25643813085796685,
            "avoidable_violation_rate": 0.20765027322404372,
        },
        "epochs": summaries,
        "selected_checkpoint": selected["checkpoint"],
        "selected_summary": selected,
        "decision": decision,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
