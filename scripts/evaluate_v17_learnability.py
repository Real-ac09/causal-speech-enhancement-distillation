#!/usr/bin/env python3
"""Select a V17 epoch using the frozen controller-learnability gate."""

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
from train_v17_gate import (
    OrdinalGatePairedSpeechDataset,
    ordinal_gate_collate_fn,
)
from train_v17_local_gate import LocalOrdinalGateDataset


ROOT = Path(__file__).resolve().parents[1]
GRID = np.asarray((0.0, 0.25, 0.5, 0.75, 1.0))
THRESHOLDS = {
    "dns_target_correlation_minimum": 0.30,
    "macro_nearest_grid_accuracy_minimum": 0.35,
    "dns_target_mae_maximum": 0.20,
    "monotonic_tolerance": 0.02,
    "target_one_minus_zero_mean_minimum": 0.30,
    "voice_target_mae_maximum": 0.10,
}


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or np.std(first) == 0.0 or np.std(second) == 0.0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _checkpoint_predictions(
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
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc=checkpoint_path.stem,
            leave=False,
        ):
            noisy = batch["noisy"].to(device, non_blocking=True)
            output = model(noisy)
            frames = min(
                output.gate_strength.shape[1],
                batch["gate_frame_mask"].shape[1],
            )
            mask = batch["gate_frame_mask"][:, :frames].to(
                device,
                non_blocking=True,
            )
            denominator = mask.sum(dim=1, keepdim=True).clamp_min(1)
            probabilities = output.gate_probabilities[:, :frames].float()
            mean_probabilities = (
                probabilities * mask.unsqueeze(-1)
            ).sum(dim=1) / denominator
            strengths = output.gate_strength[:, :frames, 0].float()
            mean_strength = (
                strengths * mask
            ).sum(dim=1) / denominator.squeeze(1)
            predicted_class = mean_probabilities.argmax(dim=-1)
            for index, file_id in enumerate(batch["file_id"]):
                rows.append(
                    {
                        "file_id": file_id,
                        "target_class": int(
                            batch["gate_target_class"][index]
                        ),
                        "target_strength": float(
                            batch["gate_target_strength"][index]
                        ),
                        "predicted_class": int(predicted_class[index]),
                        "predicted_strength": float(mean_strength[index]),
                        "valid_speech_frames": int(denominator[index, 0]),
                        **{
                            f"probability_{class_index}": float(
                                mean_probabilities[index, class_index]
                            )
                            for class_index in range(len(GRID))
                        },
                    }
                )
    predictions = pd.DataFrame(rows)
    metadata = loader.dataset.metadata[
        ["file_id", "domain", "speaker_id"]
    ]
    return predictions.merge(
        metadata,
        on="file_id",
        validate="one_to_one",
    )


def _summarise(predictions: pd.DataFrame) -> dict[str, Any]:
    target = predictions["target_strength"].to_numpy(float)
    predicted = predictions["predicted_strength"].to_numpy(float)
    target_class = predictions["target_class"].to_numpy(int)
    predicted_class = predictions["predicted_class"].to_numpy(int)
    recalls = [
        float(np.mean(predicted_class[target_class == label] == label))
        for label in range(len(GRID))
    ]
    dns = predictions["domain"].astype(str) == "dns1_training"
    voice = predictions["domain"].astype(str) == "voicebank_demand"
    class_means = [
        float(predicted[target_class == label].mean())
        for label in range(len(GRID))
    ]
    differences = np.diff(class_means)
    metrics = {
        "items": int(len(predictions)),
        "overall_target_correlation": _correlation(target, predicted),
        "overall_target_mae": float(np.mean(np.abs(predicted - target))),
        "nearest_grid_accuracy": float(
            np.mean(predicted_class == target_class)
        ),
        "macro_nearest_grid_accuracy": float(np.mean(recalls)),
        "per_class_recall": {
            str(label): recalls[label] for label in range(len(GRID))
        },
        "predicted_target_class_means": {
            f"{GRID[label]:.2f}": class_means[label]
            for label in range(len(GRID))
        },
        "minimum_adjacent_target_mean_difference": float(
            differences.min()
        ),
        "target_one_minus_zero_mean": float(
            class_means[-1] - class_means[0]
        ),
        "dns_target_correlation": _correlation(
            target[dns],
            predicted[dns],
        ),
        "dns_target_mae": float(
            np.mean(np.abs(predicted[dns] - target[dns]))
        ),
        "voice_target_mae": float(
            np.mean(np.abs(predicted[voice] - target[voice]))
        ),
    }
    checks = {
        "dns_target_correlation": (
            metrics["dns_target_correlation"]
            >= THRESHOLDS["dns_target_correlation_minimum"]
        ),
        "macro_nearest_grid_accuracy": (
            metrics["macro_nearest_grid_accuracy"]
            >= THRESHOLDS["macro_nearest_grid_accuracy_minimum"]
        ),
        "dns_target_mae": (
            metrics["dns_target_mae"]
            <= THRESHOLDS["dns_target_mae_maximum"]
        ),
        "predicted_target_means_monotonic": bool(
            np.all(differences >= -THRESHOLDS["monotonic_tolerance"])
        ),
        "target_one_minus_zero_mean": (
            metrics["target_one_minus_zero_mean"]
            >= THRESHOLDS["target_one_minus_zero_mean_minimum"]
        ),
        "voice_target_mae": (
            metrics["voice_target_mae"]
            <= THRESHOLDS["voice_target_mae_maximum"]
        ),
    }
    return {
        "metrics": metrics,
        "checks": checks,
        "passed_checks": int(sum(checks.values())),
        "total_checks": len(checks),
        "passes_learnability_gate": bool(all(checks.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(
            "data/processed/v17_balanced_ordinal/calibration.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/v17/recipe1_learnability"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--failure-decision",
        default="stop_recipe1_and_prepare_local_two_second_oracles",
    )
    args = parser.parse_args()

    if args.checkpoint is not None:
        checkpoints = [args.checkpoint]
    else:
        checkpoints = sorted(args.checkpoint_dir.glob("epoch_*.pt"))
    if not checkpoints:
        raise FileNotFoundError("No V17 epoch checkpoints were found")

    metadata_columns = set(pd.read_csv(args.metadata, nrows=1).columns)
    dataset_class = (
        LocalOrdinalGateDataset
        if "window_start_sample" in metadata_columns
        else OrdinalGatePairedSpeechDataset
    )
    dataset = dataset_class(
        metadata_csv=args.metadata,
        project_root=ROOT,
        sample_rate=16_000,
        chunk_seconds=(
            2.0 if dataset_class is LocalOrdinalGateDataset else 4.0
        ),
        random_crop=False,
        peak_normalize=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=ordinal_gate_collate_fn,
    )
    device = torch.device(args.device)
    reports = []
    predictions_by_checkpoint = {}
    for checkpoint in checkpoints:
        predictions = _checkpoint_predictions(checkpoint, loader, device)
        summary = _summarise(predictions)
        summary["checkpoint"] = str(checkpoint)
        reports.append(summary)
        predictions_by_checkpoint[str(checkpoint)] = predictions

    def selection_key(report):
        metrics = report["metrics"]
        return (
            report["passed_checks"],
            metrics["macro_nearest_grid_accuracy"],
            metrics["dns_target_correlation"],
            -metrics["dns_target_mae"],
        )

    selected = max(reports, key=selection_key)
    decision = (
        "advance_selected_epoch_to_frozen_development_benchmarks"
        if selected["passes_learnability_gate"]
        else args.failure_decision
    )
    report = {
        "status": "complete",
        "thresholds": THRESHOLDS,
        "epochs": reports,
        "selected_checkpoint": selected["checkpoint"],
        "selected_summary": selected,
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    predictions_by_checkpoint[selected["checkpoint"]].to_csv(
        args.output_dir / "selected_predictions.csv",
        index=False,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
