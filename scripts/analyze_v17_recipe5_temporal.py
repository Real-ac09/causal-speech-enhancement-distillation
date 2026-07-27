#!/usr/bin/env python3
"""Measure Recipe-5 head quality as causal context accumulates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from analyze_v17_recipe5_heads import _average, _correlation
from cnvqg.models.factory import build_model
from evaluate_v17_recipe5 import GRID, _loader, _summarise


ROOT = Path(__file__).resolve().parents[1]
POOL_NAMES = (
    "all_frames",
    "first_quarter",
    "second_quarter",
    "third_quarter",
    "final_quarter",
    "final_valid_frame",
)


def _pool_masks(mask: torch.Tensor, frames: int) -> dict[str, torch.Tensor]:
    mask = mask[:, :frames].bool()
    boundaries = np.linspace(0, frames, 5).round().astype(int)
    result = {"all_frames": mask}
    for name, left, right in zip(
        POOL_NAMES[1:5],
        boundaries[:-1],
        boundaries[1:],
    ):
        positional = torch.zeros_like(mask)
        positional[:, left:right] = True
        selected = mask & positional
        empty = ~selected.any(dim=1)
        selected[empty] = mask[empty]
        result[name] = selected
    final = torch.zeros_like(mask)
    for item in range(mask.shape[0]):
        valid = torch.nonzero(mask[item], as_tuple=False).flatten()
        final[item, valid[-1] if len(valid) else frames - 1] = True
    result["final_valid_frame"] = final
    return result


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
        "--output",
        type=Path,
        default=Path(
            "results/v17/recipe5_postmortem/temporal_summary.json"
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(
        ROOT / args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    model = build_model(checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = _loader(
        ROOT / args.metadata,
        args.batch_size,
        args.num_workers,
    )
    collected = {
        name: {
            "file_id": [],
            "target_class": [],
            "probability": [],
            "utility": [],
            "log_violation": [],
            "feasibility_probability": [],
            "target_utility": [],
            "target_log_violation": [],
            "target_feasible": [],
        }
        for name in POOL_NAMES
    }
    with torch.inference_mode():
        for batch in tqdm(loader, desc="recipe5-temporal"):
            output = model(batch["noisy"].to(device, non_blocking=True))
            frames = min(
                output.gate_probabilities.shape[1],
                batch["gate_frame_mask"].shape[1],
            )
            pools = _pool_masks(batch["gate_frame_mask"], frames)
            for name, mask in pools.items():
                bucket = collected[name]
                bucket["file_id"].extend(str(x) for x in batch["file_id"])
                bucket["target_class"].append(
                    batch["gate_target_class"].numpy()
                )
                bucket["probability"].append(
                    _average(output.gate_probabilities, mask).cpu().numpy()
                )
                bucket["utility"].append(
                    _average(output.gate_utility, mask).cpu().numpy()
                )
                bucket["log_violation"].append(
                    _average(output.gate_log_violation, mask).cpu().numpy()
                )
                bucket["feasibility_probability"].append(
                    torch.sigmoid(
                        _average(output.gate_feasibility_logits, mask)
                    ).cpu().numpy()
                )
                bucket["target_utility"].append(
                    batch["gate_target_utility"].numpy()
                )
                bucket["target_log_violation"].append(
                    torch.log1p(
                        batch["gate_target_violation"]
                    ).clamp_max(5.0).numpy()
                )
                bucket["target_feasible"].append(
                    batch["gate_target_feasible"].numpy()
                )
    candidates = pd.read_csv(ROOT / args.candidates)
    labels = pd.read_csv(ROOT / args.metadata)
    summaries = {}
    for name, bucket in collected.items():
        target_class = np.concatenate(bucket["target_class"])
        probability = np.concatenate(bucket["probability"])
        utility = np.concatenate(bucket["utility"])
        log_violation = np.concatenate(bucket["log_violation"])
        feasibility_probability = np.concatenate(
            bucket["feasibility_probability"]
        )
        target_utility = np.clip(
            np.concatenate(bucket["target_utility"]),
            -10.0,
            10.0,
        )
        target_log_violation = np.concatenate(
            bucket["target_log_violation"]
        )
        target_feasible = np.concatenate(bucket["target_feasible"])
        predicted_class = probability.argmax(axis=1)
        policy_predictions = pd.DataFrame(
            {
                "file_id": bucket["file_id"],
                "target_class": target_class,
                "predicted_class": predicted_class,
                "predicted_strength": probability @ GRID,
            }
        )
        selection = _summarise(
            policy_predictions,
            candidates,
            labels,
        )
        selection.pop("predictions")
        summaries[name] = {
            "utility_correlation": _correlation(
                target_utility.ravel(),
                utility.ravel(),
            ),
            "log_violation_correlation": _correlation(
                target_log_violation.ravel(),
                log_violation.ravel(),
            ),
            "feasibility_roc_auc": float(
                roc_auc_score(
                    target_feasible.ravel(),
                    feasibility_probability.ravel(),
                )
            ),
            "selection": selection,
        }
    best_context = max(
        summaries,
        key=lambda name: (
            summaries[name]["feasibility_roc_auc"],
            summaries[name]["utility_correlation"],
        ),
    )
    report = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "pools": summaries,
        "best_context_pool": best_context,
        "best_context_summary": summaries[best_context],
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
