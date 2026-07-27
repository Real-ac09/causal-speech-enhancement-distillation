#!/usr/bin/env python3
"""Post-test DNS1 audit separating baseline and distillation failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ("pesq", "si_sdr", "stoi", "estoi")


def _load_seed(directory: Path) -> pd.DataFrame:
    frame = pd.read_csv(directory / "per_file_metrics.csv")
    return frame.set_index("file_id").sort_index()


def _bootstrap(
    values: np.ndarray, *, samples: int, rng: np.random.Generator
) -> list[float]:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return np.quantile(means, (0.025, 0.975)).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-analysis", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=14_207)
    args = parser.parse_args()
    if len(args.reference) != len(args.candidate):
        parser.error("reference and candidate seed counts must match")
    if len(args.reference) < 2:
        parser.error("post-test audit requires multiple paired seeds")

    frame = pd.read_csv(args.feature_analysis).set_index("file_id").sort_index()
    references = [_load_seed(path) for path in args.reference]
    candidates = [_load_seed(path) for path in args.candidate]
    for values in [*references, *candidates]:
        if not values.index.equals(frame.index):
            raise ValueError("Evaluation and feature-analysis IDs differ")

    noisy_columns = {
        "pesq": "noisy_pesq",
        "si_sdr": "noisy_si_sdr",
        "stoi": "noisy_stoi",
        "estoi": "noisy_estoi",
    }
    enhanced_columns = {
        "pesq": "enhanced_pesq",
        "si_sdr": "enhanced_si_sdr",
        "stoi": "enhanced_stoi",
        "estoi": "enhanced_estoi",
    }
    rng = np.random.default_rng(args.bootstrap_seed)
    report: dict[str, object] = {
        "status": "post_test_descriptive_analysis_only",
        "selection_use_permitted": False,
        "uncertainty_method": "file_bootstrap_after_averaging_training_seeds",
        "seed_hierarchical_inference_permitted": False,
        "items": len(frame),
        "seed_pairs": len(references),
        "metrics": {},
        "tradeoffs": {},
        "condition_findings": {},
    }
    for metric in METRICS:
        noisy_column = noisy_columns[metric]
        enhanced_column = enhanced_columns[metric]
        noisy = candidates[0][noisy_column].to_numpy(float)
        reference = np.stack(
            [seed[enhanced_column].to_numpy(float) for seed in references]
        )
        candidate = np.stack(
            [seed[enhanced_column].to_numpy(float) for seed in candidates]
        )
        reference_gain = reference - noisy
        candidate_gain = candidate - noisy
        paired_delta = candidate - reference
        reference_mean_gain = reference_gain.mean(axis=0)
        candidate_mean_gain = candidate_gain.mean(axis=0)
        paired_mean = paired_delta.mean(axis=0)
        frame[f"v13_{metric}_gain_mean"] = reference_mean_gain
        frame[f"v14_{metric}_gain_mean"] = candidate_mean_gain
        frame[f"v14_minus_v13_{metric}_mean"] = paired_mean
        frame[f"v14_{metric}_all_seed_harm"] = (
            candidate_gain.max(axis=0) < 0.0
        )
        report["metrics"][metric] = {
            "v13_gain_mean": float(reference_mean_gain.mean()),
            "v13_harm_rate": float((reference_mean_gain < 0.0).mean()),
            "v14_gain_mean": float(candidate_mean_gain.mean()),
            "v14_gain_ci95": _bootstrap(
                candidate_mean_gain,
                samples=args.bootstrap_samples,
                rng=rng,
            ),
            "v14_harm_rate": float((candidate_mean_gain < 0.0).mean()),
            "v14_all_seed_harm_rate": float(
                (candidate_gain.max(axis=0) < 0.0).mean()
            ),
            "v14_minus_v13_mean": float(paired_mean.mean()),
            "v14_minus_v13_ci95": _bootstrap(
                paired_mean,
                samples=args.bootstrap_samples,
                rng=rng,
            ),
        }

    pesq_positive = frame["v14_pesq_gain_mean"] > 0.0
    stoi_negative = frame["v14_stoi_gain_mean"] < 0.0
    estoi_negative = frame["v14_estoi_gain_mean"] < 0.0
    si_sdr_negative = frame["v14_si_sdr_gain_mean"] < 0.0
    report["tradeoffs"] = {
        "pesq_up_stoi_down": {
            "items": int((pesq_positive & stoi_negative).sum()),
            "rate": float((pesq_positive & stoi_negative).mean()),
        },
        "pesq_up_estoi_down": {
            "items": int((pesq_positive & estoi_negative).sum()),
            "rate": float((pesq_positive & estoi_negative).mean()),
        },
        "pesq_up_si_sdr_down": {
            "items": int((pesq_positive & si_sdr_negative).sum()),
            "rate": float((pesq_positive & si_sdr_negative).mean()),
        },
        "pesq_up_all_intelligibility_down": {
            "items": int(
                (
                    pesq_positive
                    & stoi_negative
                    & estoi_negative
                    & si_sdr_negative
                ).sum()
            ),
            "rate": float(
                (
                    pesq_positive
                    & stoi_negative
                    & estoi_negative
                    & si_sdr_negative
                ).mean()
            ),
        },
    }

    frame["clean_rms_quartile"] = pd.qcut(
        frame["clean_rms_dbfs"],
        4,
        labels=("quietest", "quiet", "loud", "loudest"),
    )
    condition_columns = (
        "snr_band",
        "input_pesq_quartile",
        "noise_flatness_tertile",
        "noise_centroid_tertile",
        "clean_rms_quartile",
    )
    condition_rows: list[dict[str, object]] = []
    for condition in condition_columns:
        for group_name, group in frame.groupby(
            condition, observed=True, sort=False
        ):
            condition_rows.append(
                {
                    "condition": condition,
                    "group": str(group_name),
                    "items": len(group),
                    "v14_pesq_gain": group["v14_pesq_gain_mean"].mean(),
                    "v14_si_sdr_gain": group["v14_si_sdr_gain_mean"].mean(),
                    "v14_stoi_gain": group["v14_stoi_gain_mean"].mean(),
                    "v14_estoi_gain": group["v14_estoi_gain_mean"].mean(),
                    "v14_stoi_harm_rate": (
                        group["v14_stoi_gain_mean"] < 0.0
                    ).mean(),
                    "v14_minus_v13_stoi": group[
                        "v14_minus_v13_stoi_mean"
                    ].mean(),
                }
            )
    conditions = pd.DataFrame(condition_rows)
    worst_condition = conditions.sort_values("v14_stoi_gain").iloc[0]
    report["condition_findings"]["worst_mean_stoi_condition"] = {
        "condition": str(worst_condition["condition"]),
        "group": str(worst_condition["group"]),
        "items": int(worst_condition["items"]),
        "v14_stoi_gain": float(worst_condition["v14_stoi_gain"]),
        "v14_stoi_harm_rate": float(
            worst_condition["v14_stoi_harm_rate"]
        ),
    }

    predictors = (
        "estimated_input_snr_db",
        "noisy_pesq",
        "noisy_si_sdr",
        "noisy_stoi",
        "noisy_estoi",
        "clean_rms_dbfs",
        "noise_rms_dbfs",
        "noise_spectral_centroid_hz",
        "noise_spectral_flatness",
        "noise_low_band_fraction",
        "noise_high_band_fraction",
        "noise_nonstationarity",
        "clean_active_fraction",
    )
    correlations = {
        predictor: float(
            frame[[predictor, "v14_stoi_gain_mean"]]
            .corr(method="spearman")
            .iloc[0, 1]
        )
        for predictor in predictors
    }
    report["condition_findings"]["stoi_gain_spearman"] = correlations
    report["condition_findings"]["strongest_stoi_association"] = max(
        correlations.items(), key=lambda item: abs(item[1])
    )

    worst_columns = [
        "speaker_id",
        "estimated_input_snr_db",
        "noisy_pesq",
        "noisy_stoi",
        "clean_rms_dbfs",
        "noise_spectral_flatness",
        "noise_nonstationarity",
        "v13_pesq_gain_mean",
        "v14_pesq_gain_mean",
        "v13_stoi_gain_mean",
        "v14_stoi_gain_mean",
        "v14_minus_v13_stoi_mean",
        "v14_stoi_all_seed_harm",
    ]
    worst = (
        frame.sort_values("v14_stoi_gain_mean")
        .head(30)[worst_columns]
        .reset_index()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.reset_index().to_csv(
        args.output_dir / "paired_per_file_analysis.csv", index=False
    )
    conditions.to_csv(args.output_dir / "condition_summary.csv", index=False)
    worst.to_csv(args.output_dir / "worst_stoi_cases.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
