#!/usr/bin/env python3
"""Estimate how predictable V17 local oracle targets are from Recipe-4 inputs.

This is a diagnostic upper-bound experiment, not a deployable enhancement
model.  It aggregates only causal, clean-reference-free features available to
the Recipe-4 controller and fits conventional CPU baselines on the frozen
training/calibration split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from cnvqg.models.factory import build_model
from train_v17_gate import ordinal_gate_collate_fn
from train_v17_local_gate import LocalOrdinalGateDataset


ROOT = Path(__file__).resolve().parents[1]
GRID = np.asarray((0.0, 0.25, 0.5, 0.75, 1.0), dtype=np.float64)
THRESHOLDS = {
    "dns_target_correlation_minimum": 0.30,
    "macro_nearest_grid_accuracy_minimum": 0.35,
    "dns_target_mae_maximum": 0.20,
    "monotonic_tolerance": 0.02,
    "target_one_minus_zero_mean_minimum": 0.30,
    "voice_target_mae_maximum": 0.10,
}
STAT_NAMES = ("mean", "std", "q10", "median", "q90", "delta_abs_mean")


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or np.std(first) == 0.0 or np.std(second) == 0.0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _dataset(metadata: Path) -> LocalOrdinalGateDataset:
    return LocalOrdinalGateDataset(
        metadata_csv=str(metadata),
        project_root=".",
        sample_rate=16_000,
        chunk_seconds=2.0,
        peak_normalize=False,
        pcs_target=False,
        clean_input_probability=0.0,
    )


def _statistics(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Feature statistics require [frames, features]")
    if values.shape[0] > 1:
        delta = values[1:].sub(values[:-1]).abs().mean(dim=0)
    else:
        delta = torch.zeros_like(values[0])
    return torch.stack(
        (
            values.mean(dim=0),
            values.std(dim=0, unbiased=False),
            torch.quantile(values, 0.10, dim=0),
            torch.quantile(values, 0.50, dim=0),
            torch.quantile(values, 0.90, dim=0),
            delta,
        ),
        dim=0,
    )


def _extract(
    checkpoint_path: Path,
    metadata_path: Path,
    output_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    if output_path.is_file():
        return pd.read_csv(output_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model = build_model(checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = _dataset(metadata_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=ordinal_gate_collate_fn,
    )
    feature_names: list[str] | None = None
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"features:{metadata_path.stem}"):
            noisy = batch["noisy"].to(device, non_blocking=True)
            base = model.backbone._forward_waveform(noisy, pad_end=True)
            mixture_spectrum = (
                base.speech_spectrum + base.noise_spectrum
            ).detach()
            rich = model.confidence_gate.rich_summaries(
                mixture_spectrum,
                base.speech_spectrum.detach(),
            )
            noise = base.continuous_noise_state.detach().float()
            frames = min(
                rich.shape[1],
                noise.shape[1],
                batch["gate_frame_mask"].shape[1],
            )
            combined = torch.cat(
                (rich[:, :frames].float(), noise[:, :frames]),
                dim=-1,
            )
            if feature_names is None:
                raw_names = [
                    *(f"acoustic_{name}" for name in model.confidence_gate.FEATURE_NAMES),
                    *(f"noise_state_{index:02d}" for index in range(noise.shape[-1])),
                ]
                feature_names = [
                    f"{raw}_{stat}"
                    for stat in STAT_NAMES
                    for raw in raw_names
                ]
            masks = batch["gate_frame_mask"][:, :frames]
            for index, file_id in enumerate(batch["file_id"]):
                selected = combined[index][masks[index]].cpu()
                flattened = _statistics(selected).reshape(-1).numpy()
                row = dataset.metadata.loc[
                    dataset.metadata["file_id"] == file_id
                ].iloc[0]
                rows.append(
                    {
                        "file_id": str(file_id),
                        "domain": str(row["domain"]),
                        "target_class": int(
                            batch["gate_target_class"][index]
                        ),
                        "target_strength": float(
                            batch["gate_target_strength"][index]
                        ),
                        **dict(zip(feature_names, flattened.tolist())),
                    }
                )
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def _sample_weights(frame: pd.DataFrame) -> np.ndarray:
    pairs = frame.groupby(["domain", "target_class"]).size()
    domains = frame["domain"].value_counts()
    weights = []
    for domain, label in zip(frame["domain"], frame["target_class"]):
        domain_count = float(domains[domain])
        pair_count = float(pairs.loc[(domain, label)])
        weights.append(
            (1.0 / domain_count) * (domain_count / pair_count) ** 0.5
        )
    result = np.asarray(weights, dtype=np.float64)
    return result / result.mean()


def _checks(metrics: dict[str, Any]) -> dict[str, bool]:
    means = np.asarray(
        list(metrics["predicted_target_class_means"].values())
    )
    return {
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
            np.all(np.diff(means) >= -THRESHOLDS["monotonic_tolerance"])
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


def _summarise(
    frame: pd.DataFrame,
    predicted_strength: np.ndarray,
    predicted_class: np.ndarray,
) -> dict[str, Any]:
    target = frame["target_strength"].to_numpy(float)
    target_class = frame["target_class"].to_numpy(int)
    dns = frame["domain"].astype(str).to_numpy() == "dns1_training"
    voice = frame["domain"].astype(str).to_numpy() == "voicebank_demand"
    recalls = [
        float(np.mean(predicted_class[target_class == label] == label))
        for label in range(len(GRID))
    ]
    class_means = [
        float(predicted_strength[target_class == label].mean())
        for label in range(len(GRID))
    ]
    metrics = {
        "items": int(len(frame)),
        "overall_target_correlation": _correlation(
            target, predicted_strength
        ),
        "overall_target_mae": float(
            np.mean(np.abs(predicted_strength - target))
        ),
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
        "target_one_minus_zero_mean": float(
            class_means[-1] - class_means[0]
        ),
        "dns_target_correlation": _correlation(
            target[dns], predicted_strength[dns]
        ),
        "dns_target_mae": float(
            np.mean(np.abs(predicted_strength[dns] - target[dns]))
        ),
        "voice_target_mae": float(
            np.mean(np.abs(predicted_strength[voice] - target[voice]))
        ),
    }
    checks = _checks(metrics)
    return {
        "metrics": metrics,
        "checks": checks,
        "passed_checks": int(sum(checks.values())),
        "total_checks": len(checks),
        "passes_learnability_gate": bool(all(checks.values())),
    }


def _feature_matrix(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    excluded = {"file_id", "domain", "target_class", "target_strength"}
    feature_names = [
        column for column in train.columns if column not in excluded
    ]
    return (
        train[feature_names].to_numpy(np.float32),
        calibration[feature_names].to_numpy(np.float32),
        feature_names,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/v17/feature_recipe4_seed17043/epoch_007.pt"
        ),
    )
    parser.add_argument(
        "--train-metadata",
        type=Path,
        default=Path(
            "results/v17/recipe2_local_oracles/train/metadata.csv"
        ),
    )
    parser.add_argument(
        "--calibration-metadata",
        type=Path,
        default=Path(
            "results/v17/recipe2_local_oracles/calibration/metadata.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/v17/recipe4_predictability"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    device = torch.device(args.device)
    output_dir = ROOT / args.output_dir
    train = _extract(
        ROOT / args.checkpoint,
        ROOT / args.train_metadata,
        output_dir / "train_features.csv",
        device,
        args.batch_size,
        args.num_workers,
    )
    calibration = _extract(
        ROOT / args.checkpoint,
        ROOT / args.calibration_metadata,
        output_dir / "calibration_features.csv",
        device,
        args.batch_size,
        args.num_workers,
    )
    x_train, x_calibration, feature_names = _feature_matrix(
        train.copy(), calibration.copy()
    )
    y_strength = train["target_strength"].to_numpy(float)
    y_class = train["target_class"].to_numpy(int)
    weights = _sample_weights(train)

    regressors = {
        "dummy_median": DummyRegressor(strategy="median"),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=17044,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=5,
            max_features=0.75,
            n_jobs=-1,
            random_state=17044,
        ),
    }
    classifiers = {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                max_iter=1000,
                class_weight="balanced",
                random_state=17044,
            ),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=17044,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            max_features=0.75,
            class_weight="balanced",
            n_jobs=-1,
            random_state=17044,
        ),
    }
    report: dict[str, Any] = {
        "status": "complete",
        "claim_scope": (
            "CPU diagnostic using aggregated causal Recipe-4 inputs; "
            "not deployable performance and no development/test data."
        ),
        "checkpoint": str(args.checkpoint),
        "train_items": int(len(train)),
        "calibration_items": int(len(calibration)),
        "features": len(feature_names),
        "thresholds": THRESHOLDS,
        "regressors": {},
        "classifiers": {},
    }
    target_calibration = calibration["target_strength"].to_numpy(float)
    for name, estimator in regressors.items():
        estimator.fit(
            x_train,
            y_strength,
            **({"sample_weight": weights} if name != "ridge" else {}),
        )
        predicted = np.clip(estimator.predict(x_calibration), 0.0, 1.0)
        predicted_class = np.abs(
            predicted[:, None] - GRID[None]
        ).argmin(axis=1)
        report["regressors"][name] = _summarise(
            calibration, predicted, predicted_class
        )

    for name, estimator in classifiers.items():
        estimator.fit(
            x_train,
            y_class,
            **(
                {"sample_weight": weights}
                if name not in {"logistic"}
                else {}
            ),
        )
        probability = estimator.predict_proba(x_calibration)
        complete = np.zeros((len(calibration), len(GRID)))
        complete[:, np.asarray(estimator.classes_, dtype=int)] = probability
        predicted = complete @ GRID
        predicted_class = complete.argmax(axis=1)
        report["classifiers"][name] = _summarise(
            calibration, predicted, predicted_class
        )

    candidates = [
        (f"regressor:{name}", value)
        for name, value in report["regressors"].items()
    ] + [
        (f"classifier:{name}", value)
        for name, value in report["classifiers"].items()
    ]
    candidates.sort(
        key=lambda item: (
            item[1]["passed_checks"],
            item[1]["metrics"]["overall_target_correlation"],
            -item[1]["metrics"]["overall_target_mae"],
        ),
        reverse=True,
    )
    report["selected_diagnostic"] = candidates[0][0]
    report["selected_summary"] = candidates[0][1]
    report["conclusion"] = (
        "recipe4_controller_is_primary_bottleneck"
        if candidates[0][1]["passes_learnability_gate"]
        else "recipe4_inputs_do_not_support_frozen_gate"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
