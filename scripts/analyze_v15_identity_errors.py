#!/usr/bin/env python3
"""Paired error analysis for V15 quiet-level candidates A and B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ("pesq", "si_sdr", "stoi", "estoi")
CATASTROPHIC_DELTA_THRESHOLDS = {
    "pesq": -0.10,
    "si_sdr": -1.0,
    "stoi": -0.01,
    "estoi": -0.01,
}


def _load_evaluation(directory: Path, prefix: str) -> pd.DataFrame:
    frame = (
        pd.read_csv(directory / "per_file_metrics.csv")
        .set_index("file_id")
        .sort_index()
    )
    return frame.rename(
        columns={
            column: f"{prefix}_{column}"
            for column in frame.columns
            if column != "speaker_id"
        }
    ).drop(columns=["speaker_id"])


def _bootstrap_ci(
    values: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> list[float]:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def _trimmed_mean(values: np.ndarray, proportion: float = 0.05) -> float:
    ordered = np.sort(values)
    count = int(np.floor(len(ordered) * proportion))
    trimmed = ordered[count:-count] if count else ordered
    return float(trimmed.mean())


def _validate_and_join(
    *,
    metadata_path: Path,
    candidate_a: Path,
    candidate_b: Path,
) -> pd.DataFrame:
    metadata = (
        pd.read_csv(metadata_path).set_index("file_id").sort_index()
    )
    a_frame = _load_evaluation(candidate_a, "a")
    b_frame = _load_evaluation(candidate_b, "b")
    if not (
        metadata.index.equals(a_frame.index)
        and a_frame.index.equals(b_frame.index)
    ):
        raise ValueError(
            f"Metadata and evaluation IDs differ for {metadata_path}"
        )
    frame = metadata.join(a_frame).join(b_frame)
    for metric in METRICS:
        difference = (
            frame[f"a_noisy_{metric}"] - frame[f"b_noisy_{metric}"]
        ).abs()
        if float(difference.max()) > 1e-7:
            raise ValueError(f"Noisy {metric} values differ between A and B")
        noisy = frame[f"a_noisy_{metric}"]
        frame[f"a_{metric}_gain"] = frame[f"a_enhanced_{metric}"] - noisy
        frame[f"b_{metric}_gain"] = frame[f"b_enhanced_{metric}"] - noisy
        frame[f"b_minus_a_{metric}"] = (
            frame[f"b_enhanced_{metric}"]
            - frame[f"a_enhanced_{metric}"]
        )
        a_harm = frame[f"a_{metric}_gain"] < 0.0
        b_harm = frame[f"b_{metric}_gain"] < 0.0
        frame[f"{metric}_harm_transition"] = np.select(
            [
                a_harm & b_harm,
                a_harm & ~b_harm,
                ~a_harm & b_harm,
            ],
            ["persistent_harm", "repaired", "introduced"],
            default="persistent_non_harm",
        )
    frame["pesq_stoi_tradeoff"] = np.select(
        [
            (frame["b_minus_a_pesq"] > 0.0)
            & (frame["b_minus_a_stoi"] > 0.0),
            (frame["b_minus_a_pesq"] <= 0.0)
            & (frame["b_minus_a_stoi"] > 0.0),
            (frame["b_minus_a_pesq"] > 0.0)
            & (frame["b_minus_a_stoi"] <= 0.0),
        ],
        [
            "both_improved",
            "stoi_up_pesq_down",
            "pesq_up_stoi_down",
        ],
        default="both_regressed",
    )
    frame["si_sdr_identity_collapse"] = (
        (frame["a_si_sdr_gain"] >= 1.0)
        & (frame["b_si_sdr_gain"] <= 0.25 * frame["a_si_sdr_gain"])
    )
    frame["pesq_identity_collapse"] = (
        (frame["a_pesq_gain"] >= 0.20)
        & (frame["b_pesq_gain"] <= 0.25 * frame["a_pesq_gain"])
    )
    frame["identity_collapse_diagnostic"] = (
        frame["si_sdr_identity_collapse"]
        | frame["pesq_identity_collapse"]
    )
    return frame


def _metric_summary(
    frame: pd.DataFrame,
    *,
    metric: str,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    delta = frame[f"b_minus_a_{metric}"].to_numpy(float)
    a_gain = frame[f"a_{metric}_gain"]
    b_gain = frame[f"b_{metric}_gain"]
    transition = frame[f"{metric}_harm_transition"]
    return {
        "candidate_a_enhanced_mean": float(
            frame[f"a_enhanced_{metric}"].mean()
        ),
        "candidate_b_enhanced_mean": float(
            frame[f"b_enhanced_{metric}"].mean()
        ),
        "b_minus_a_mean": float(delta.mean()),
        "b_minus_a_ci95": _bootstrap_ci(
            delta, samples=samples, rng=rng
        ),
        "b_minus_a_median": float(np.median(delta)),
        "b_minus_a_5pct_trimmed_mean": _trimmed_mean(delta),
        "b_win_rate": float((delta > 0.0).mean()),
        "candidate_a_harm_rate": float((a_gain < 0.0).mean()),
        "candidate_b_harm_rate": float((b_gain < 0.0).mean()),
        "repaired_items": int((transition == "repaired").sum()),
        "introduced_harm_items": int((transition == "introduced").sum()),
        "catastrophic_regression_threshold": (
            CATASTROPHIC_DELTA_THRESHOLDS[metric]
        ),
        "catastrophic_regression_items": int(
            (
                delta
                <= CATASTROPHIC_DELTA_THRESHOLDS[metric]
            ).sum()
        ),
    }


def _dataset_summary(
    frame: pd.DataFrame,
    *,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "items": int(len(frame)),
        "metrics": {},
        "pesq_stoi_tradeoff_counts": {
            str(key): int(value)
            for key, value in frame["pesq_stoi_tradeoff"]
            .value_counts()
            .sort_index()
            .items()
        },
        "identity_collapse_diagnostic": {
            "definition": (
                "A gain >= 1 dB SI-SDR or 0.20 PESQ and B retains at "
                "most 25% of that gain"
            ),
            "items": int(frame["identity_collapse_diagnostic"].sum()),
            "rate": float(frame["identity_collapse_diagnostic"].mean()),
            "file_ids": frame.index[
                frame["identity_collapse_diagnostic"]
            ].tolist(),
        },
    }
    for metric in METRICS:
        summary["metrics"][metric] = _metric_summary(
            frame,
            metric=metric,
            samples=samples,
            rng=rng,
        )
    return summary


def _group_summary(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_column in group_columns:
        for value, group in frame.groupby(group_column, sort=True):
            row: dict[str, object] = {
                "condition": group_column,
                "value": value,
                "items": int(len(group)),
                "identity_collapse_items": int(
                    group["identity_collapse_diagnostic"].sum()
                ),
            }
            for metric in METRICS:
                delta = group[f"b_minus_a_{metric}"]
                row[f"b_minus_a_{metric}_mean"] = float(delta.mean())
                row[f"b_minus_a_{metric}_median"] = float(delta.median())
                row[f"b_{metric}_win_rate"] = float((delta > 0.0).mean())
                row[f"a_{metric}_harm_rate"] = float(
                    (group[f"a_{metric}_gain"] < 0.0).mean()
                )
                row[f"b_{metric}_harm_rate"] = float(
                    (group[f"b_{metric}_gain"] < 0.0).mean()
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _ranked_cases(
    frames: dict[str, pd.DataFrame],
    *,
    largest: bool,
    count: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, frame in frames.items():
        for metric in METRICS:
            ranked = frame[f"b_minus_a_{metric}"].sort_values(
                ascending=not largest
            )
            for rank, (file_id, value) in enumerate(
                ranked.iloc[:count].items(), start=1
            ):
                source = frame.loc[file_id]
                rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "rank": rank,
                        "file_id": file_id,
                        "speaker_id": source["speaker_id"],
                        "duration_seconds": source["duration_seconds"],
                        "b_minus_a": float(value),
                        "noisy": float(source[f"a_noisy_{metric}"]),
                        "candidate_a": float(
                            source[f"a_enhanced_{metric}"]
                        ),
                        "candidate_b": float(
                            source[f"b_enhanced_{metric}"]
                        ),
                        "candidate_a_gain": float(
                            source[f"a_{metric}_gain"]
                        ),
                        "candidate_b_gain": float(
                            source[f"b_{metric}_gain"]
                        ),
                        "identity_collapse_diagnostic": bool(
                            source["identity_collapse_diagnostic"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cross-metadata",
        type=Path,
        default=Path("data/processed/dns_cross_domain_dev/metadata.csv"),
    )
    parser.add_argument(
        "--voice-metadata",
        type=Path,
        default=Path(
            "data/processed/voicebank_demand/metadata/"
            "v12_architecture_selection_400.csv"
        ),
    )
    parser.add_argument(
        "--candidate-a-root",
        type=Path,
        default=Path("results/v15/preservation/quiet_level_seed1200"),
    )
    parser.add_argument(
        "--candidate-b-root",
        type=Path,
        default=Path(
            "results/v15/preservation/quiet_level_identity_seed1200"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/v15/preservation/quiet_level_identity_seed1200/"
            "error_analysis_vs_quiet_level"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=15_021)
    args = parser.parse_args()

    cross = _validate_and_join(
        metadata_path=args.cross_metadata,
        candidate_a=args.candidate_a_root / "cross_domain_dev",
        candidate_b=args.candidate_b_root / "cross_domain_dev",
    )
    voice = _validate_and_join(
        metadata_path=args.voice_metadata,
        candidate_a=args.candidate_a_root / "voicebank_dev400",
        candidate_b=args.candidate_b_root / "voicebank_dev400",
    )
    rng = np.random.default_rng(args.bootstrap_seed)
    report = {
        "status": "complete",
        "analysis_role": "paired development error analysis",
        "external_test_used": False,
        "candidate_a": "v15_quiet_level_seed1200_epoch3",
        "candidate_b": "v15_quiet_level_identity_seed1200_epoch3",
        "bootstrap": {
            "method": "paired_file",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "datasets": {
            "cross_domain_dev": _dataset_summary(
                cross, samples=args.bootstrap_samples, rng=rng
            ),
            "voicebank_dev400": _dataset_summary(
                voice, samples=args.bootstrap_samples, rng=rng
            ),
        },
    }

    cross_conditions = _group_summary(
        cross,
        group_columns=("target_clean_rms_dbfs", "target_snr_db"),
    )
    voice_conditions = _group_summary(
        voice,
        group_columns=("speaker_id",),
    )
    regressions = _ranked_cases(
        {"cross_domain_dev": cross, "voicebank_dev400": voice},
        largest=False,
    )
    improvements = _ranked_cases(
        {"cross_domain_dev": cross, "voicebank_dev400": voice},
        largest=True,
    )
    collapse_cases = pd.concat(
        [
            cross[cross["identity_collapse_diagnostic"]].assign(
                dataset="cross_domain_dev"
            ),
            voice[voice["identity_collapse_diagnostic"]].assign(
                dataset="voicebank_dev400"
            ),
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cross.reset_index().to_csv(
        args.output_dir / "cross_per_file.csv", index=False
    )
    voice.reset_index().to_csv(
        args.output_dir / "voice_per_file.csv", index=False
    )
    cross_conditions.to_csv(
        args.output_dir / "cross_condition_summary.csv", index=False
    )
    voice_conditions.to_csv(
        args.output_dir / "voice_speaker_summary.csv", index=False
    )
    regressions.to_csv(
        args.output_dir / "top_regressions.csv", index=False
    )
    improvements.to_csv(
        args.output_dir / "top_improvements.csv", index=False
    )
    collapse_cases.reset_index().to_csv(
        args.output_dir / "identity_collapse_cases.csv", index=False
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
