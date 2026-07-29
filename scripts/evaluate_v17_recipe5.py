#!/usr/bin/env python3
"""Evaluate Recipe-5 selection utility and preservation on calibration data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cnvqg.models.factory import build_model
from train_v17_recipe5 import (
    UtilitySafetyGateDataset,
    recipe5_collate_fn,
)


ROOT = Path(__file__).resolve().parents[1]
GRID = np.asarray((0.0, 0.25, 0.5, 0.75, 1.0))
METRICS = ("pesq", "si_sdr", "stoi", "estoi")
THRESHOLDS = {
    "top1_accuracy_minimum": 0.45,
    "macro_accuracy_minimum": 0.35,
    "avoidable_violation_rate_maximum": 0.20,
    "mean_constraint_violation_maximum": 1.50,
    "mean_utility_minimum": 2.50,
    "mean_pesq_delta_minimum": 0.33,
    "si_sdr_violation_rate_maximum": 0.10,
    "stoi_violation_rate_maximum": 0.18,
    "estoi_violation_rate_maximum": 0.11,
}


def _loader(metadata: Path, batch_size: int, workers: int) -> DataLoader:
    dataset = UtilitySafetyGateDataset(
        metadata_csv=str(metadata),
        project_root=".",
        sample_rate=16_000,
        chunk_seconds=2.0,
        peak_normalize=False,
        pcs_target=False,
        clean_input_probability=0.0,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=recipe5_collate_fn,
    )


def _predictions(
    checkpoint_path: Path,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model = build_model(checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=checkpoint_path.stem, leave=False):
            output = model(batch["noisy"].to(device, non_blocking=True))
            frames = min(
                output.gate_probabilities.shape[1],
                batch["gate_frame_mask"].shape[1],
            )
            mask = batch["gate_frame_mask"][:, :frames].to(
                device,
                non_blocking=True,
            ).float()
            denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            probability = (
                output.gate_probabilities[:, :frames].float()
                * mask.unsqueeze(-1)
            ).sum(dim=1) / denominator
            predicted_class = probability.argmax(dim=-1)
            expected_strength = probability @ torch.as_tensor(
                GRID,
                device=device,
                dtype=probability.dtype,
            )
            for index, file_id in enumerate(batch["file_id"]):
                rows.append(
                    {
                        "file_id": str(file_id),
                        "target_class": int(
                            batch["gate_target_class"][index]
                        ),
                        "predicted_class": int(predicted_class[index]),
                        "predicted_strength": float(
                            expected_strength[index]
                        ),
                        **{
                            f"probability_{level}": float(
                                probability[index, level]
                            )
                            for level in range(len(GRID))
                        },
                    }
                )
    return pd.DataFrame(rows)


def _summarise(
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
) -> dict[str, Any]:
    selected = predictions.copy()
    selected["strength"] = GRID[
        selected["predicted_class"].to_numpy(int)
    ]
    selected = selected.merge(
        candidates,
        on=["file_id", "strength"],
        validate="one_to_one",
    ).merge(
        labels[
            [
                "file_id",
                "oracle_strength",
                "oracle_class",
                "oracle_feasible",
            ]
        ],
        on="file_id",
        validate="one_to_one",
    )
    target_class = selected["oracle_class"].to_numpy(int)
    predicted_class = selected["predicted_class"].to_numpy(int)
    recalls = [
        float(np.mean(predicted_class[target_class == level] == level))
        for level in range(len(GRID))
    ]
    feasible_context = selected["oracle_feasible"].astype(bool)
    avoidable = ~selected.loc[
        feasible_context,
        "feasible",
    ].astype(bool)
    metrics: dict[str, Any] = {
        "items": int(len(selected)),
        "top1_accuracy": float(np.mean(predicted_class == target_class)),
        "macro_accuracy": float(np.mean(recalls)),
        "per_class_recall": {
            str(level): recalls[level] for level in range(len(GRID))
        },
        "mean_predicted_strength": float(
            selected["predicted_strength"].mean()
        ),
        "actual_feasible_rate": float(
            selected["feasible"].astype(bool).mean()
        ),
        "avoidable_violation_rate": float(avoidable.mean()),
        "mean_constraint_violation": float(
            selected["constraint_violation_normalised"].mean()
        ),
        "mean_utility": float(selected["utility"].mean()),
    }
    for metric in METRICS:
        delta = (
            selected[f"enhanced_{metric}"]
            - selected[f"noisy_{metric}"]
        ).dropna()
        violation = selected[
            f"{metric}_violation_normalised"
        ].dropna()
        metrics[f"mean_{metric}_delta"] = float(delta.mean())
        metrics[f"{metric}_violation_rate"] = float(
            (violation > 0.0).mean()
        )
    checks = {
        "top1_accuracy": (
            metrics["top1_accuracy"]
            >= THRESHOLDS["top1_accuracy_minimum"]
        ),
        "macro_accuracy": (
            metrics["macro_accuracy"]
            >= THRESHOLDS["macro_accuracy_minimum"]
        ),
        "avoidable_violation_rate": (
            metrics["avoidable_violation_rate"]
            <= THRESHOLDS["avoidable_violation_rate_maximum"]
        ),
        "mean_constraint_violation": (
            metrics["mean_constraint_violation"]
            <= THRESHOLDS["mean_constraint_violation_maximum"]
        ),
        "mean_utility": (
            metrics["mean_utility"]
            >= THRESHOLDS["mean_utility_minimum"]
        ),
        "mean_pesq_delta": (
            metrics["mean_pesq_delta"]
            >= THRESHOLDS["mean_pesq_delta_minimum"]
        ),
        "si_sdr_violation_rate": (
            metrics["si_sdr_violation_rate"]
            <= THRESHOLDS["si_sdr_violation_rate_maximum"]
        ),
        "stoi_violation_rate": (
            metrics["stoi_violation_rate"]
            <= THRESHOLDS["stoi_violation_rate_maximum"]
        ),
        "estoi_violation_rate": (
            metrics["estoi_violation_rate"]
            <= THRESHOLDS["estoi_violation_rate_maximum"]
        ),
    }
    return {
        "metrics": metrics,
        "checks": checks,
        "passed_checks": int(sum(checks.values())),
        "total_checks": len(checks),
        "passes_recipe5_gate": bool(all(checks.values())),
        "predictions": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--checkpoint-dir", type=Path)
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
        "--output-dir",
        type=Path,
        default=Path("results/v17/recipe5_selection"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--failure-decision",
        default="stop_recipe5_and_reassess_quality_proxy_features",
    )
    args = parser.parse_args()
    checkpoints = (
        [ROOT / args.checkpoint]
        if args.checkpoint
        else sorted((ROOT / args.checkpoint_dir).glob("epoch_*.pt"))
    )
    if not checkpoints:
        raise FileNotFoundError("No Recipe-5 checkpoints found")
    metadata = ROOT / args.metadata
    candidates = pd.read_csv(ROOT / args.candidates)
    labels = pd.read_csv(metadata)
    loader = _loader(metadata, args.batch_size, args.num_workers)
    summaries = []
    stored_predictions: dict[str, pd.DataFrame] = {}
    for checkpoint in checkpoints:
        predictions = _predictions(
            checkpoint,
            loader,
            torch.device(args.device),
        )
        result = _summarise(predictions, candidates, labels)
        stored_predictions[str(checkpoint)] = result.pop("predictions")
        result["checkpoint"] = str(checkpoint.relative_to(ROOT))
        summaries.append(result)
    summaries.sort(
        key=lambda item: (
            item["passed_checks"],
            -item["metrics"]["avoidable_violation_rate"],
            -item["metrics"]["mean_constraint_violation"],
            item["metrics"]["mean_utility"],
            item["metrics"]["mean_pesq_delta"],
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
        "advance_recipe5_to_frozen_development_benchmarks"
        if selected["passes_recipe5_gate"]
        else args.failure_decision
    )
    report = {
        "status": "complete",
        "thresholds": THRESHOLDS,
        "recipe4_reference": {
            "avoidable_violation_rate": 0.23861566484517305,
            "mean_constraint_violation": 1.8151758937337863,
            "mean_utility": 2.4477544694358797,
            "mean_pesq_delta": 0.3455879087537948,
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
