#!/usr/bin/env python3
"""Analyse multi-seed V13 errors without running enhancement inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
METRIC_SPECS = {
    "pesq": ("noisy_pesq", "enhanced_pesq"),
    "si_sdr": ("noisy_si_sdr", "enhanced_si_sdr"),
    "stoi": ("noisy_stoi", "enhanced_stoi"),
    "estoi": ("noisy_estoi", "enhanced_estoi"),
}
CONDITION_COLUMNS = (
    "speaker_id",
    "snr_band",
    "duration_band",
    "input_pesq_quartile",
    "noise_centroid_tertile",
    "noise_flatness_tertile",
)


def _rms(values: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64)) + eps))


def _frames(values: np.ndarray, length: int = 320, hop: int = 160) -> np.ndarray:
    if len(values) < length:
        values = np.pad(values, (0, length - len(values)))
    count = 1 + (len(values) - length) // hop
    shape = (count, length)
    strides = (values.strides[0] * hop, values.strides[0])
    return np.lib.stride_tricks.as_strided(
        values, shape=shape, strides=strides
    ).copy()


def _audio_features(row: pd.Series) -> dict[str, float]:
    noisy, noisy_rate = sf.read(ROOT / row["noisy_path"], dtype="float32")
    clean, clean_rate = sf.read(ROOT / row["clean_path"], dtype="float32")
    if noisy_rate != clean_rate or noisy_rate != int(row["sample_rate"]):
        raise ValueError(f"Sample-rate mismatch for {row['file_id']}")
    noisy = np.asarray(noisy).squeeze()
    clean = np.asarray(clean).squeeze()
    length = min(len(noisy), len(clean))
    noisy = noisy[:length]
    clean = clean[:length]
    noise = noisy - clean

    clean_rms = _rms(clean)
    noise_rms = _rms(noise)
    noisy_rms = _rms(noisy)
    snr_db = 20.0 * np.log10((clean_rms + 1e-12) / (noise_rms + 1e-12))

    noise_frames = _frames(noise)
    window = np.hanning(noise_frames.shape[1]).astype(np.float32)
    spectrum = np.fft.rfft(noise_frames * window, n=512, axis=1)
    power = np.square(np.abs(spectrum), dtype=np.float64).mean(axis=0) + 1e-16
    frequencies = np.fft.rfftfreq(512, d=1.0 / noisy_rate)
    total_power = power.sum()
    centroid = float((frequencies * power).sum() / total_power)
    flatness = float(np.exp(np.log(power).mean()) / power.mean())
    low_fraction = float(power[frequencies < 500].sum() / total_power)
    high_fraction = float(power[frequencies >= 4000].sum() / total_power)

    frame_rms = np.sqrt(
        np.mean(np.square(noise_frames, dtype=np.float64), axis=1) + 1e-12
    )
    nonstationarity = float(frame_rms.std() / (frame_rms.mean() + 1e-12))
    clean_frames = _frames(clean)
    clean_frame_rms = np.sqrt(
        np.mean(np.square(clean_frames, dtype=np.float64), axis=1) + 1e-12
    )
    active_threshold = max(clean_frame_rms.max() * 0.01, 1e-5)
    active_fraction = float(np.mean(clean_frame_rms >= active_threshold))
    clipped_fraction = float(np.mean(np.abs(noisy) >= 0.999))
    return {
        "estimated_input_snr_db": float(snr_db),
        "clean_rms_dbfs": float(20.0 * np.log10(clean_rms + 1e-12)),
        "noise_rms_dbfs": float(20.0 * np.log10(noise_rms + 1e-12)),
        "noisy_rms_dbfs": float(20.0 * np.log10(noisy_rms + 1e-12)),
        "noise_spectral_centroid_hz": centroid,
        "noise_spectral_flatness": flatness,
        "noise_low_band_fraction": low_fraction,
        "noise_high_band_fraction": high_fraction,
        "noise_nonstationarity": nonstationarity,
        "clean_active_fraction": active_fraction,
        "noisy_clipped_fraction": clipped_fraction,
    }


def _load_seed_metrics(directories: Iterable[Path]) -> tuple[pd.DataFrame, list[int]]:
    frames: list[pd.DataFrame] = []
    seeds: list[int] = []
    for index, directory in enumerate(directories):
        frame = pd.read_csv(directory / "per_file_metrics.csv").sort_values(
            "file_id"
        )
        if frame["file_id"].duplicated().any():
            raise ValueError(f"Duplicate file IDs in {directory}")
        seed_text = directory.name.removeprefix("seed")
        seed = int(seed_text) if seed_text.isdigit() else index
        seeds.append(seed)
        frames.append(frame.set_index("file_id"))
    if len(frames) < 2:
        raise ValueError("Multi-seed error analysis requires at least two seeds")
    expected = frames[0].index
    for frame in frames[1:]:
        if not frame.index.equals(expected):
            raise ValueError("All seed evaluations must have identical file IDs")

    combined = pd.DataFrame(index=expected)
    combined["speaker_id"] = frames[0]["speaker_id"]
    for name, (noisy, enhanced) in METRIC_SPECS.items():
        noisy_values = np.stack([frame[noisy].to_numpy(float) for frame in frames])
        if not np.allclose(noisy_values, noisy_values[0], atol=1e-10, rtol=0.0):
            raise ValueError(f"Noisy {name} values differ between seed evaluations")
        enhanced_values = np.stack(
            [frame[enhanced].to_numpy(float) for frame in frames]
        )
        gains = enhanced_values - noisy_values
        combined[f"noisy_{name}"] = noisy_values[0]
        combined[f"enhanced_{name}_mean"] = enhanced_values.mean(axis=0)
        combined[f"enhanced_{name}_seed_std"] = enhanced_values.std(
            axis=0, ddof=1
        )
        combined[f"{name}_gain_mean"] = gains.mean(axis=0)
        combined[f"{name}_gain_min_seed"] = gains.min(axis=0)
        combined[f"{name}_gain_max_seed"] = gains.max(axis=0)
        combined[f"{name}_gain_seed_range"] = gains.max(axis=0) - gains.min(axis=0)
        combined[f"{name}_sign_disagreement"] = (
            (gains.min(axis=0) < 0.0) & (gains.max(axis=0) > 0.0)
        )
        for seed_index, seed in enumerate(seeds):
            combined[f"enhanced_{name}_seed{seed}"] = enhanced_values[seed_index]
            combined[f"{name}_gain_seed{seed}"] = gains[seed_index]
    return combined.reset_index(), seeds


def _bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator, samples: int
) -> list[float]:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    estimates = values[indices].mean(axis=1)
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def _condition_labels(frame: pd.DataFrame) -> None:
    frame["snr_band"] = pd.cut(
        frame["estimated_input_snr_db"],
        bins=[-np.inf, 5.0, 10.0, 15.0, np.inf],
        labels=["<5 dB", "5-10 dB", "10-15 dB", ">=15 dB"],
        ordered=True,
    )
    frame["duration_band"] = pd.cut(
        frame["duration_seconds"],
        bins=[-np.inf, 2.0, 3.0, np.inf],
        labels=["<2 s", "2-3 s", ">=3 s"],
        ordered=True,
    )
    frame["input_pesq_quartile"] = pd.qcut(
        frame["noisy_pesq"],
        4,
        labels=["Q1 hardest", "Q2", "Q3", "Q4 easiest"],
    )
    frame["noise_centroid_tertile"] = pd.qcut(
        frame["noise_spectral_centroid_hz"],
        3,
        labels=["low centroid", "mid centroid", "high centroid"],
    )
    frame["noise_flatness_tertile"] = pd.qcut(
        frame["noise_spectral_flatness"],
        3,
        labels=["tonal", "mixed", "noise-like"],
    )


def _group_summary(
    frame: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for condition in CONDITION_COLUMNS:
        for value, group in frame.groupby(condition, observed=True, sort=False):
            row: dict[str, object] = {
                "condition": condition,
                "group": str(value),
                "items": len(group),
                "noisy_pesq_mean": group["noisy_pesq"].mean(),
                "estimated_input_snr_db_mean": group[
                    "estimated_input_snr_db"
                ].mean(),
            }
            for metric in METRIC_SPECS:
                gain = group[f"{metric}_gain_mean"].to_numpy(float)
                row[f"{metric}_gain_mean"] = gain.mean()
                low, high = _bootstrap_ci(gain, rng, bootstrap_samples)
                row[f"{metric}_gain_ci95_low"] = low
                row[f"{metric}_gain_ci95_high"] = high
                row[f"{metric}_harm_items"] = int(np.sum(gain < 0.0))
                row[f"{metric}_harm_rate"] = float(np.mean(gain < 0.0))
            rows.append(row)
    return pd.DataFrame(rows)


def _summary(
    frame: pd.DataFrame,
    *,
    seeds: list[int],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    report: dict[str, object] = {
        "items": len(frame),
        "training_seeds": seeds,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "metrics": {},
        "seed_stability": {},
        "correlations": {},
    }
    any_harm = np.zeros(len(frame), dtype=bool)
    multi_harm_count = np.zeros(len(frame), dtype=int)
    for metric in METRIC_SPECS:
        gain = frame[f"{metric}_gain_mean"].to_numpy(float)
        harm = gain < 0.0
        any_harm |= harm
        multi_harm_count += harm.astype(int)
        report["metrics"][metric] = {
            "gain_mean": float(gain.mean()),
            "gain_ci95": _bootstrap_ci(gain, rng, bootstrap_samples),
            "gain_percentiles": {
                key: float(value)
                for key, value in zip(
                    ("p05", "p25", "p50", "p75", "p95"),
                    np.quantile(gain, [0.05, 0.25, 0.50, 0.75, 0.95]),
                )
            },
            "harm_items": int(harm.sum()),
            "harm_rate": float(harm.mean()),
        }
        seed_std = frame[f"enhanced_{metric}_seed_std"].to_numpy(float)
        disagreement = frame[f"{metric}_sign_disagreement"].to_numpy(bool)
        report["seed_stability"][metric] = {
            "mean_per_file_seed_std": float(seed_std.mean()),
            "p95_per_file_seed_std": float(np.quantile(seed_std, 0.95)),
            "sign_disagreement_items": int(disagreement.sum()),
            "sign_disagreement_rate": float(disagreement.mean()),
        }

    report["failure_overlap"] = {
        "any_metric_harm_items": int(any_harm.sum()),
        "any_metric_harm_rate": float(any_harm.mean()),
        "two_or_more_metrics_harmed_items": int(np.sum(multi_harm_count >= 2)),
        "two_or_more_metrics_harmed_rate": float(
            np.mean(multi_harm_count >= 2)
        ),
        "all_four_metrics_harmed_items": int(np.sum(multi_harm_count == 4)),
        "all_four_metrics_harmed_rate": float(np.mean(multi_harm_count == 4)),
    }

    predictors = (
        "estimated_input_snr_db",
        "duration_seconds",
        "noisy_pesq",
        "noisy_si_sdr",
        "noisy_stoi",
        "noisy_estoi",
        "clean_rms_dbfs",
        "noise_spectral_centroid_hz",
        "noise_spectral_flatness",
        "noise_low_band_fraction",
        "noise_high_band_fraction",
        "noise_nonstationarity",
        "clean_active_fraction",
    )
    for metric in METRIC_SPECS:
        correlations = {}
        for predictor in predictors:
            correlations[predictor] = float(
                frame[[predictor, f"{metric}_gain_mean"]]
                .corr(method="spearman")
                .iloc[0, 1]
            )
        report["correlations"][f"{metric}_gain"] = correlations
    return report


def _worst_cases(frame: pd.DataFrame, count: int = 20) -> pd.DataFrame:
    columns = [
        "file_id",
        "speaker_id",
        "duration_seconds",
        "estimated_input_snr_db",
        "noisy_pesq",
        "enhanced_pesq_mean",
        "pesq_gain_mean",
        "si_sdr_gain_mean",
        "stoi_gain_mean",
        "estoi_gain_mean",
        "noise_spectral_centroid_hz",
        "noise_spectral_flatness",
        "noise_nonstationarity",
    ]
    rows = []
    criteria = {
        "lowest_pesq_gain": "pesq_gain_mean",
        "lowest_si_sdr_gain": "si_sdr_gain_mean",
        "lowest_stoi_gain": "stoi_gain_mean",
        "lowest_estoi_gain": "estoi_gain_mean",
        "lowest_enhanced_pesq": "enhanced_pesq_mean",
        "highest_pesq_seed_variability": "enhanced_pesq_seed_std",
    }
    for criterion, column in criteria.items():
        ascending = criterion != "highest_pesq_seed_variability"
        selected = frame.sort_values(column, ascending=ascending).head(count)
        for rank, (_, row) in enumerate(selected.iterrows(), start=1):
            record = {"criterion": criterion, "rank": rank}
            record.update(row[columns].to_dict())
            rows.append(record)
    return pd.DataFrame(rows)


def _plots(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colours = frame["estimated_input_snr_db"]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for axis, metric in zip(axes.flat, METRIC_SPECS):
        axis.scatter(
            frame[f"noisy_{metric}"],
            frame[f"{metric}_gain_mean"],
            c=colours,
            cmap="viridis",
            s=13,
            alpha=0.65,
            linewidths=0,
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xlabel(f"Noisy {metric.upper()}")
        axis.set_ylabel(f"{metric.upper()} gain")
        axis.grid(alpha=0.2)
    colourbar = figure.colorbar(
        axes[0, 0].collections[0],
        ax=axes.ravel().tolist(),
        location="right",
        shrink=0.85,
        pad=0.02,
    )
    colourbar.set_label("Estimated input SNR (dB)")
    figure.suptitle("Per-file gain versus input quality (three-seed mean)")
    figure.savefig(output_dir / "gain_vs_input_quality.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    labels = ["<5 dB", "5-10 dB", "10-15 dB", ">=15 dB"]
    for axis, metric in zip(axes.flat, METRIC_SPECS):
        groups = [
            frame.loc[frame["snr_band"].astype(str) == label, f"{metric}_gain_mean"]
            for label in labels
        ]
        axis.boxplot(groups, tick_labels=labels, showfliers=False)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(f"{metric.upper()} gain")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Gain by signal-derived input-SNR band")
    figure.tight_layout()
    figure.savefig(output_dir / "gain_by_input_snr.png", dpi=180)
    plt.close(figure)

    harm_rates = [
        100.0 * np.mean(frame[f"{metric}_gain_mean"] < 0.0)
        for metric in METRIC_SPECS
    ]
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    bars = axis.bar([name.upper() for name in METRIC_SPECS], harm_rates)
    axis.bar_label(bars, fmt="%.1f%%", padding=3)
    axis.set_ylabel("Files harmed (%)")
    axis.set_ylim(0.0, max(5.0, max(harm_rates) * 1.2))
    axis.set_title("Mean enhancement harm rate across three seeds")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "metric_harm_rates.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, metric in zip(axes.flat, METRIC_SPECS):
        axis.hist(
            frame[f"enhanced_{metric}_seed_std"],
            bins=30,
            color="#4472C4",
            alpha=0.85,
        )
        axis.set_xlabel(f"Per-file {metric.upper()} seed std")
        axis.set_ylabel("Files")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Training-seed sensitivity by file")
    figure.tight_layout()
    figure.savefig(output_dir / "seed_sensitivity.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/voicebank_demand/metadata/test.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/v13/error_analysis"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=13001)
    args = parser.parse_args()

    frame, seeds = _load_seed_metrics(args.evaluation)
    metadata = pd.read_csv(args.metadata)
    if metadata["file_id"].duplicated().any():
        raise ValueError("Metadata contains duplicate file IDs")
    frame = frame.merge(metadata, on=["file_id", "speaker_id"], how="left")
    if frame["duration_seconds"].isna().any():
        raise ValueError("Some evaluated files are missing from metadata")

    features = pd.DataFrame(
        [_audio_features(row) for _, row in metadata.iterrows()]
    )
    features.insert(0, "file_id", metadata["file_id"].to_numpy())
    frame = frame.merge(features, on="file_id", how="left")
    _condition_labels(frame)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "per_file_analysis.csv", index=False)
    groups = _group_summary(
        frame,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    groups.to_csv(output_dir / "condition_summary.csv", index=False)
    worst = _worst_cases(frame)
    worst.to_csv(output_dir / "worst_cases.csv", index=False)
    report = _summary(
        frame,
        seeds=seeds,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    _plots(frame, output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
