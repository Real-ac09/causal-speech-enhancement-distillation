#!/usr/bin/env python3
"""Fit an offline Recipe-5 upper bound to cached causal proxy statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_v17_recipe5 import GRID, _summarise


ROOT = Path(__file__).resolve().parents[1]


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if np.std(first) == 0.0 or np.std(second) == 0.0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _features(
    feature_path: Path,
    target_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    features = pd.read_csv(feature_path)
    targets = pd.read_csv(target_path)
    merged = features.merge(
        targets,
        on="file_id",
        suffixes=("", "_target"),
        validate="one_to_one",
    )
    excluded = {
        "file_id",
        "domain",
        "target_class",
        "target_strength",
    }
    feature_names = [
        column
        for column in features.columns
        if column not in excluded
    ]
    return (
        merged,
        merged[feature_names].to_numpy(np.float32),
        feature_names,
    )


def _targets(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    utility = np.column_stack(
        [frame[f"r5_utility_{level}"] for level in range(len(GRID))]
    ).clip(-10.0, 10.0)
    log_violation = np.log1p(
        np.column_stack(
            [frame[f"r5_violation_{level}"] for level in range(len(GRID))]
        )
    ).clip(max=5.0)
    feasible = np.column_stack(
        [frame[f"r5_feasible_{level}"] for level in range(len(GRID))]
    ).astype(int)
    return utility, log_violation, feasible


def _fit_feasibility(
    x_train: np.ndarray,
    train_target: np.ndarray,
    x_calibration: np.ndarray,
) -> np.ndarray:
    probabilities = []
    for level in range(len(GRID)):
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                max_iter=1000,
                class_weight="balanced",
                random_state=17045,
            ),
        )
        estimator.fit(x_train, train_target[:, level])
        probabilities.append(
            estimator.predict_proba(x_calibration)[:, 1]
        )
    return np.column_stack(probabilities)


def _policies(
    utility: np.ndarray,
    log_violation: np.ndarray,
    feasibility: np.ndarray,
    direct_probability: np.ndarray,
) -> dict[str, np.ndarray]:
    policies = {
        "direct_balanced_logistic": direct_probability.argmax(axis=1),
        "utility_only": utility.argmax(axis=1),
        "minimum_log_violation": log_violation.argmin(axis=1),
    }
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
        selected = []
        for item in range(len(utility)):
            safe = feasibility[item] >= threshold
            if safe.any():
                selected.append(
                    int(np.argmax(np.where(safe, utility[item], -np.inf)))
                )
            else:
                selected.append(int(np.argmin(log_violation[item])))
        policies[f"lexicographic_{threshold:.1f}"] = np.asarray(selected)
    for penalty in (0.5, 1.0, 2.0, 4.0, 8.0):
        policies[f"utility_minus_{penalty:.1f}_violation"] = (
            utility - penalty * log_violation
        ).argmax(axis=1)
    return policies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("results/v17/recipe4_predictability"),
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("data/processed/v17_recipe5"),
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
            "results/v17/recipe5_postmortem/proxy_ceiling.json"
        ),
    )
    args = parser.parse_args()
    train, x_train, feature_names = _features(
        ROOT / args.feature_dir / "train_features.csv",
        ROOT / args.target_dir / "train.csv",
    )
    calibration, x_calibration, calibration_names = _features(
        ROOT / args.feature_dir / "calibration_features.csv",
        ROOT / args.target_dir / "calibration.csv",
    )
    if feature_names != calibration_names:
        raise ValueError("Train/calibration feature schemas differ")
    train_utility, train_violation, train_feasible = _targets(train)
    cal_utility, cal_violation, cal_feasible = _targets(calibration)
    utility_model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0),
    ).fit(x_train, train_utility)
    violation_model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0),
    ).fit(x_train, train_violation)
    predicted_utility = utility_model.predict(x_calibration)
    predicted_violation = violation_model.predict(x_calibration).clip(
        0.0, 5.0
    )
    predicted_feasibility = _fit_feasibility(
        x_train,
        train_feasible,
        x_calibration,
    )
    direct = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            max_iter=1000,
            class_weight="balanced",
            random_state=17045,
        ),
    )
    direct.fit(x_train, train["oracle_class"].to_numpy(int))
    direct_probability = direct.predict_proba(x_calibration)
    complete_probability = np.zeros((len(calibration), len(GRID)))
    complete_probability[:, direct.classes_.astype(int)] = direct_probability

    candidates = pd.read_csv(ROOT / args.candidates)
    labels = pd.read_csv(ROOT / args.target_dir / "calibration.csv")
    policy_summaries: dict[str, Any] = {}
    for name, predicted_class in _policies(
        predicted_utility,
        predicted_violation,
        predicted_feasibility,
        complete_probability,
    ).items():
        predictions = pd.DataFrame(
            {
                "file_id": calibration["file_id"],
                "target_class": calibration["oracle_class"],
                "predicted_class": predicted_class,
                "predicted_strength": GRID[predicted_class],
            }
        )
        summary = _summarise(predictions, candidates, labels)
        summary.pop("predictions")
        policy_summaries[name] = summary
    eligible = [
        name
        for name, summary in policy_summaries.items()
        if summary["metrics"]["mean_pesq_delta"] >= 0.33
    ]
    safest = min(
        eligible or list(policy_summaries),
        key=lambda name: (
            policy_summaries[name]["metrics"]["avoidable_violation_rate"],
            policy_summaries[name]["metrics"]["mean_constraint_violation"],
            -policy_summaries[name]["metrics"]["mean_utility"],
        ),
    )
    report = {
        "status": "complete",
        "claim_scope": (
            "Offline upper bound using utterance-aggregated, clean-free "
            "Recipe-4 features; not a deployable streaming result."
        ),
        "train_items": int(len(train)),
        "calibration_items": int(len(calibration)),
        "features": len(feature_names),
        "heads": {
            "utility": {
                "correlation": _correlation(
                    cal_utility.ravel(),
                    predicted_utility.ravel(),
                ),
                "mae": float(
                    np.mean(np.abs(cal_utility - predicted_utility))
                ),
            },
            "log_violation": {
                "correlation": _correlation(
                    cal_violation.ravel(),
                    predicted_violation.ravel(),
                ),
                "mae": float(
                    np.mean(np.abs(cal_violation - predicted_violation))
                ),
            },
            "feasibility": {
                "roc_auc": float(
                    roc_auc_score(
                        cal_feasible.ravel(),
                        predicted_feasibility.ravel(),
                    )
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(
                        cal_feasible.ravel(),
                        (predicted_feasibility >= 0.5).ravel(),
                    )
                ),
            },
        },
        "policies": policy_summaries,
        "safest_pesq_preserving_policy": safest,
        "safest_pesq_preserving_summary": policy_summaries[safest],
    }
    safest_metrics = policy_summaries[safest]["metrics"]
    report["beats_recipe4_safety"] = bool(
        safest_metrics["avoidable_violation_rate"] < 0.23861566484517305
        and safest_metrics["mean_constraint_violation"] < 1.8151758937337863
        and safest_metrics["mean_pesq_delta"] >= 0.33
    )
    report["diagnosis"] = (
        "causal_temporal_aggregation_is_justified"
        if report["beats_recipe4_safety"]
        else "current_clean_free_proxy_information_is_insufficient"
    )
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
