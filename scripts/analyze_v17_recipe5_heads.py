#!/usr/bin/env python3
"""Diagnose Recipe-5 auxiliary heads and replay selector policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tqdm import tqdm

from cnvqg.models.factory import build_model
from evaluate_v17_recipe5 import GRID, _loader, _summarise


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("pesq", "si_sdr", "stoi", "estoi")
METRIC_SCALES = np.asarray((0.1, 1.0, 0.01, 0.02))


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 2:
        return 0.0
    first = first[valid]
    second = second[valid]
    if np.std(first) == 0.0 or np.std(second) == 0.0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _average(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    frames = min(values.shape[1], mask.shape[1])
    mask = mask[:, :frames].to(values.device).float()
    denominator = mask.sum(dim=1).clamp_min(1.0)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    while denominator.ndim < values.ndim - 1:
        denominator = denominator.unsqueeze(-1)
    return (
        values[:, :frames].float() * mask
    ).sum(dim=1) / denominator


def _extract(
    checkpoint_path: Path,
    metadata_path: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> pd.DataFrame:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model = build_model(checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = _loader(metadata_path, batch_size, workers)
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="recipe5-heads"):
            output = model(batch["noisy"].to(device, non_blocking=True))
            mask = batch["gate_frame_mask"]
            probability = _average(output.gate_probabilities, mask)
            utility = _average(output.gate_utility, mask)
            log_violation = _average(output.gate_log_violation, mask)
            feasible_logit = _average(
                output.gate_feasibility_logits,
                mask,
            )
            metric_delta = _average(output.gate_metric_deltas, mask)
            for item, file_id in enumerate(batch["file_id"]):
                row: dict[str, Any] = {
                    "file_id": str(file_id),
                    "target_class": int(
                        batch["gate_target_class"][item]
                    ),
                }
                for level in range(len(GRID)):
                    row[f"probability_{level}"] = float(
                        probability[item, level]
                    )
                    row[f"predicted_utility_{level}"] = float(
                        utility[item, level]
                    )
                    row[f"predicted_log_violation_{level}"] = float(
                        log_violation[item, level]
                    )
                    row[f"predicted_feasibility_logit_{level}"] = float(
                        feasible_logit[item, level]
                    )
                    row[f"target_utility_{level}"] = float(
                        batch["gate_target_utility"][item, level]
                    )
                    row[f"target_violation_{level}"] = float(
                        batch["gate_target_violation"][item, level]
                    )
                    row[f"target_feasible_{level}"] = float(
                        batch["gate_target_feasible"][item, level]
                    )
                    row[f"target_policy_{level}"] = float(
                        batch["gate_target_policy"][item, level]
                    )
                    for metric_index, metric in enumerate(METRICS):
                        row[
                            f"predicted_{metric}_delta_{level}"
                        ] = float(
                            metric_delta[item, level, metric_index]
                            * METRIC_SCALES[metric_index]
                        )
                        row[f"target_{metric}_delta_{level}"] = float(
                            batch["gate_target_metric_deltas"][
                                item, level, metric_index
                            ]
                        )
                        row[f"target_{metric}_available_{level}"] = float(
                            batch["gate_target_metric_mask"][
                                item, level, metric_index
                            ]
                        )
                rows.append(row)
    return pd.DataFrame(rows)


def _head_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    predicted_utility = np.column_stack(
        [frame[f"predicted_utility_{level}"] for level in range(len(GRID))]
    )
    target_utility = np.column_stack(
        [frame[f"target_utility_{level}"] for level in range(len(GRID))]
    )
    target_utility = np.clip(target_utility, -10.0, 10.0)
    predicted_violation = np.column_stack(
        [
            frame[f"predicted_log_violation_{level}"]
            for level in range(len(GRID))
        ]
    )
    target_violation = np.log1p(
        np.column_stack(
            [
                frame[f"target_violation_{level}"]
                for level in range(len(GRID))
            ]
        )
    ).clip(max=5.0)
    feasible_logits = np.column_stack(
        [
            frame[f"predicted_feasibility_logit_{level}"]
            for level in range(len(GRID))
        ]
    )
    feasible_probability = 1.0 / (1.0 + np.exp(-feasible_logits))
    target_feasible = np.column_stack(
        [frame[f"target_feasible_{level}"] for level in range(len(GRID))]
    ).astype(int)
    predicted_feasible = feasible_probability >= 0.5
    report: dict[str, Any] = {
        "utility": {
            "correlation": _correlation(
                target_utility.ravel(),
                predicted_utility.ravel(),
            ),
            "mae": float(
                np.mean(np.abs(predicted_utility - target_utility))
            ),
            "per_strength_correlation": {
                f"{GRID[level]:.2f}": _correlation(
                    target_utility[:, level],
                    predicted_utility[:, level],
                )
                for level in range(len(GRID))
            },
        },
        "log_violation": {
            "correlation": _correlation(
                target_violation.ravel(),
                predicted_violation.ravel(),
            ),
            "mae": float(
                np.mean(
                    np.abs(predicted_violation - target_violation)
                )
            ),
        },
        "feasibility": {
            "accuracy": float(
                np.mean(predicted_feasible == target_feasible)
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    target_feasible.ravel(),
                    predicted_feasible.ravel(),
                )
            ),
            "roc_auc": float(
                roc_auc_score(
                    target_feasible.ravel(),
                    feasible_probability.ravel(),
                )
            ),
            "per_strength_roc_auc": {
                f"{GRID[level]:.2f}": float(
                    roc_auc_score(
                        target_feasible[:, level],
                        feasible_probability[:, level],
                    )
                )
                for level in range(len(GRID))
            },
        },
        "metric_delta": {},
    }
    for metric in METRICS:
        predicted = np.column_stack(
            [
                frame[f"predicted_{metric}_delta_{level}"]
                for level in range(len(GRID))
            ]
        )
        target = np.column_stack(
            [
                frame[f"target_{metric}_delta_{level}"]
                for level in range(len(GRID))
            ]
        )
        available = np.column_stack(
            [
                frame[f"target_{metric}_available_{level}"]
                for level in range(len(GRID))
            ]
        ).astype(bool)
        report["metric_delta"][metric] = {
            "correlation": _correlation(
                target[available],
                predicted[available],
            ),
            "mae": float(
                np.mean(np.abs(predicted[available] - target[available]))
            ),
        }
    return report


def _policy_classes(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    utility = np.column_stack(
        [frame[f"predicted_utility_{level}"] for level in range(len(GRID))]
    )
    log_violation = np.column_stack(
        [
            frame[f"predicted_log_violation_{level}"]
            for level in range(len(GRID))
        ]
    )
    feasibility_logits = np.column_stack(
        [
            frame[f"predicted_feasibility_logit_{level}"]
            for level in range(len(GRID))
        ]
    )
    feasibility = 1.0 / (1.0 + np.exp(-feasibility_logits))
    model_probability = np.column_stack(
        [frame[f"probability_{level}"] for level in range(len(GRID))]
    )
    policies = {
        "trained_selector": model_probability.argmax(axis=1),
        "utility_only": utility.argmax(axis=1),
        "minimum_predicted_violation": log_violation.argmin(axis=1),
    }
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
        selected = []
        for item in range(len(frame)):
            safe = feasibility[item] >= threshold
            if safe.any():
                score = np.where(safe, utility[item], -np.inf)
                selected.append(int(np.argmax(score)))
            else:
                selected.append(int(np.argmin(log_violation[item])))
        policies[f"lexicographic_feasibility_{threshold:.1f}"] = np.asarray(
            selected
        )
    for penalty in (0.5, 1.0, 2.0, 4.0, 8.0):
        score = utility - penalty * log_violation
        policies[f"utility_minus_{penalty:.1f}_violation"] = score.argmax(
            axis=1
        )
    return policies


def _policy_report(
    frame: pd.DataFrame,
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
) -> dict[str, Any]:
    summaries = {}
    for name, predicted_class in _policy_classes(frame).items():
        predictions = pd.DataFrame(
            {
                "file_id": frame["file_id"],
                "target_class": frame["target_class"],
                "predicted_class": predicted_class,
                "predicted_strength": GRID[predicted_class],
            }
        )
        summary = _summarise(predictions, candidates, labels)
        summary.pop("predictions")
        summaries[name] = summary
    ranked = sorted(
        summaries,
        key=lambda name: (
            summaries[name]["metrics"]["avoidable_violation_rate"],
            summaries[name]["metrics"]["mean_constraint_violation"],
            -summaries[name]["metrics"]["mean_utility"],
        ),
    )
    return {
        "policies": summaries,
        "safest_policy": ranked[0],
        "safest_summary": summaries[ranked[0]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/v17/utility_safety_recipe5_seed17044/"
            "epoch_007.pt"
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
        "--output-dir",
        type=Path,
        default=Path("results/v17/recipe5_postmortem"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    frame = _extract(
        ROOT / args.checkpoint,
        ROOT / args.metadata,
        torch.device(args.device),
        args.batch_size,
        args.num_workers,
    )
    labels = pd.read_csv(ROOT / args.metadata)
    candidates = pd.read_csv(ROOT / args.candidates)
    policy = _policy_report(frame, candidates, labels)
    head_metrics = _head_metrics(frame)
    safest = policy["safest_summary"]["metrics"]
    beats_recipe4_safety = (
        safest["avoidable_violation_rate"] < 0.23861566484517305
        and safest["mean_constraint_violation"] < 1.8151758937337863
    )
    report = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "items": int(len(frame)),
        "head_metrics": head_metrics,
        **policy,
        "beats_recipe4_safety": beats_recipe4_safety,
        "diagnosis": (
            "selector_calibration_is_primary_bottleneck"
            if beats_recipe4_safety
            else "auxiliary_heads_do_not_recover_recipe4_safety"
        ),
    }
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "head_predictions.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
