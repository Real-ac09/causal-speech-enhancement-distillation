#!/usr/bin/env python3
"""Summarize paired V13/V14.2 performance on the frozen V15 dev set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ("pesq", "si_sdr", "stoi", "estoi")


def _bootstrap_ci(
    values: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> list[float]:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def _load_evaluation(directory: Path, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(directory / "per_file_metrics.csv")
    frame = frame.set_index("file_id").sort_index()
    return frame.rename(
        columns={
            column: f"{prefix}_{column}"
            for column in frame.columns
            if column != "speaker_id"
        }
    ).drop(columns=["speaker_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=15_003)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata).set_index("file_id").sort_index()
    reference = _load_evaluation(args.reference, "v13")
    candidate = _load_evaluation(args.candidate, "v14_2")
    frame = metadata.join(reference, how="inner").join(candidate, how="inner")
    if len(frame) != len(metadata):
        raise ValueError("Evaluation outputs do not cover every development item")
    if not frame.index.equals(metadata.index):
        raise ValueError("Evaluation and metadata IDs differ")

    rng = np.random.default_rng(args.bootstrap_seed)
    report: dict[str, object] = {
        "status": "frozen_cross_domain_development_baseline",
        "selection_role": "V15 development and promotion screening",
        "external_test_claim_permitted": False,
        "items": len(frame),
        "bootstrap": {
            "method": "paired_file_bootstrap",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "metrics": {},
        "conditions": {},
    }

    for metric in METRICS:
        noisy_column = f"v14_2_noisy_{metric}"
        reference_column = f"v13_enhanced_{metric}"
        candidate_column = f"v14_2_enhanced_{metric}"
        noisy = frame[noisy_column].to_numpy(float)
        reference_values = frame[reference_column].to_numpy(float)
        candidate_values = frame[candidate_column].to_numpy(float)
        reference_gain = reference_values - noisy
        candidate_gain = candidate_values - noisy
        paired_delta = candidate_values - reference_values
        frame[f"v13_{metric}_gain"] = reference_gain
        frame[f"v14_2_{metric}_gain"] = candidate_gain
        frame[f"v14_2_minus_v13_{metric}"] = paired_delta
        report["metrics"][metric] = {
            "noisy_mean": float(noisy.mean()),
            "v13_enhanced_mean": float(reference_values.mean()),
            "v13_gain_mean": float(reference_gain.mean()),
            "v13_harm_rate": float((reference_gain < 0.0).mean()),
            "v14_2_enhanced_mean": float(candidate_values.mean()),
            "v14_2_gain_mean": float(candidate_gain.mean()),
            "v14_2_gain_ci95": _bootstrap_ci(
                candidate_gain,
                samples=args.bootstrap_samples,
                rng=rng,
            ),
            "v14_2_harm_rate": float((candidate_gain < 0.0).mean()),
            "v14_2_minus_v13_mean": float(paired_delta.mean()),
            "v14_2_minus_v13_ci95": _bootstrap_ci(
                paired_delta,
                samples=args.bootstrap_samples,
                rng=rng,
            ),
        }

    condition_rows: list[dict[str, object]] = []
    for condition in ("target_snr_db", "target_clean_rms_dbfs"):
        for value, group in frame.groupby(condition, sort=True):
            row: dict[str, object] = {
                "condition": condition,
                "value": float(value),
                "items": len(group),
            }
            for metric in METRICS:
                gain = group[f"v14_2_{metric}_gain"]
                row[f"v14_2_{metric}_gain"] = float(gain.mean())
                row[f"v14_2_{metric}_harm_rate"] = float((gain < 0.0).mean())
                row[f"v14_2_minus_v13_{metric}"] = float(
                    group[f"v14_2_minus_v13_{metric}"].mean()
                )
            condition_rows.append(row)
    conditions = pd.DataFrame(condition_rows)
    report["conditions"]["quietest_clean_level"] = (
        conditions[
            (conditions["condition"] == "target_clean_rms_dbfs")
            & (conditions["value"] == conditions[
                conditions["condition"] == "target_clean_rms_dbfs"
            ]["value"].min())
        ]
        .iloc[0]
        .to_dict()
    )
    report["conditions"]["lowest_snr"] = (
        conditions[
            (conditions["condition"] == "target_snr_db")
            & (conditions["value"] == conditions[
                conditions["condition"] == "target_snr_db"
            ]["value"].min())
        ]
        .iloc[0]
        .to_dict()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.reset_index().to_csv(args.output_dir / "per_file_analysis.csv", index=False)
    conditions.to_csv(args.output_dir / "condition_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
