#!/usr/bin/env python3
"""Aggregate absolute final-model metrics across training seeds and files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    "noisy_pesq",
    "enhanced_pesq",
    "noisy_si_sdr",
    "enhanced_si_sdr",
    "si_sdr_improvement",
    "noisy_stoi",
    "enhanced_stoi",
    "noisy_estoi",
    "enhanced_estoi",
)


def _load(directory: Path) -> pd.DataFrame:
    path = directory / "per_file_metrics.csv"
    frame = pd.read_csv(path).set_index("file_id").sort_index()
    if frame.index.has_duplicates:
        raise ValueError(f"{path} contains duplicate file_id values")
    missing = sorted(set(METRICS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing metrics: {', '.join(missing)}")
    if frame[list(METRICS)].isna().any().any():
        raise ValueError(f"{path} contains missing metric values")
    return frame


def _hierarchical_bootstrap(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
    batch_size: int = 250,
) -> np.ndarray:
    """Resample training seeds and paired file IDs in bounded-memory batches."""
    metrics, num_seeds, num_files = values.shape
    rng = np.random.default_rng(seed)
    estimates = np.empty((samples, metrics), dtype=np.float64)
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        batch = stop - start
        seed_indices = rng.integers(0, num_seeds, size=(batch, num_seeds))
        file_indices = rng.integers(0, num_files, size=(batch, num_files))
        for metric_index in range(metrics):
            sampled = values[metric_index][
                seed_indices[:, :, None], file_indices[:, None, :]
            ]
            estimates[start:stop, metric_index] = sampled.mean(axis=(1, 2))
    return estimates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--model-name", default="CN-VQG-GRU-T1")
    parser.add_argument("--expected-files", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=13_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.evaluation) < 2:
        parser.error("seed-aware aggregation requires at least two evaluations")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")

    frames = [_load(path) for path in args.evaluation]
    expected_ids = frames[0].index
    for frame in frames[1:]:
        if not frame.index.equals(expected_ids):
            raise ValueError(
                "All evaluations must contain identical ordered file_id values"
            )
    if args.expected_files is not None and len(expected_ids) != args.expected_files:
        raise ValueError(
            f"Expected {args.expected_files} files, found {len(expected_ids)}"
        )

    # [metric, seed, file]
    values = np.stack(
        [
            np.stack([frame[metric].to_numpy(dtype=np.float64) for frame in frames])
            for metric in METRICS
        ]
    )
    bootstrap = _hierarchical_bootstrap(
        values,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    report: dict[str, object] = {
        "model": args.model_name,
        "num_seeds": len(frames),
        "num_files_per_seed": len(expected_ids),
        "evaluation_directories": [str(path) for path in args.evaluation],
        "bootstrap": {
            "method": "hierarchical_seed_and_file",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "metrics": {},
    }
    for metric_index, metric in enumerate(METRICS):
        seed_values = values[metric_index].mean(axis=1)
        report["metrics"][metric] = {
            "mean": float(seed_values.mean()),
            "seed_std": float(seed_values.std(ddof=1)),
            "seed_values": seed_values.tolist(),
            "hierarchical_ci95": np.quantile(
                bootstrap[:, metric_index], [0.025, 0.975]
            ).tolist(),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
