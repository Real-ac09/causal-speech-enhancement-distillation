#!/usr/bin/env python3
"""Paired hierarchical comparison of two multi-seed model evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    "enhanced_pesq",
    "enhanced_si_sdr",
    "si_sdr_improvement",
    "enhanced_stoi",
    "enhanced_estoi",
)


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path / "per_file_metrics.csv").set_index("file_id")
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError(f"{path} contains duplicate file IDs")
    missing = sorted(set(METRICS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing metrics: {missing}")
    return frame


def _hierarchical_bootstrap(
    differences: np.ndarray,
    *,
    samples: int,
    seed: int,
    batch_size: int = 250,
) -> np.ndarray:
    """Resample paired training seeds and paired files in bounded memory."""
    metrics, num_seeds, num_files = differences.shape
    rng = np.random.default_rng(seed)
    estimates = np.empty((samples, metrics), dtype=np.float64)
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        batch = stop - start
        seed_indices = rng.integers(0, num_seeds, size=(batch, num_seeds))
        file_indices = rng.integers(0, num_files, size=(batch, num_files))
        for metric_index in range(metrics):
            sampled = differences[metric_index][
                seed_indices[:, :, None], file_indices[:, None, :]
            ]
            estimates[start:stop, metric_index] = sampled.mean(axis=(1, 2))
    return estimates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--reference-name", default="V13")
    parser.add_argument("--candidate-name", default="V14.2")
    parser.add_argument("--expected-files", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=14_203)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.reference) != len(args.candidate):
        parser.error("reference and candidate seed counts must match")
    if len(args.reference) < 2:
        parser.error("multi-seed comparison requires at least two seed pairs")
    if args.bootstrap_samples < 1:
        parser.error("bootstrap-samples must be positive")

    references = [_load(path) for path in args.reference]
    candidates = [_load(path) for path in args.candidate]
    expected_ids = references[0].index
    for frame in [*references[1:], *candidates]:
        if not frame.index.equals(expected_ids):
            raise ValueError(
                "All evaluations must contain identical ordered file IDs"
            )
    if (
        args.expected_files is not None
        and len(expected_ids) != args.expected_files
    ):
        raise ValueError(
            f"Expected {args.expected_files} files, found {len(expected_ids)}"
        )

    differences = np.stack(
        [
            np.stack(
                [
                    candidate[metric].to_numpy(np.float64)
                    - reference[metric].to_numpy(np.float64)
                    for reference, candidate in zip(references, candidates)
                ]
            )
            for metric in METRICS
        ]
    )
    bootstrap = _hierarchical_bootstrap(
        differences,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    report: dict[str, object] = {
        "reference": args.reference_name,
        "candidate": args.candidate_name,
        "num_seed_pairs": len(references),
        "num_files_per_seed": len(expected_ids),
        "reference_directories": [str(path) for path in args.reference],
        "candidate_directories": [str(path) for path in args.candidate],
        "bootstrap": {
            "method": "paired_hierarchical_seed_and_file",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "metrics": {},
    }
    for metric_index, metric in enumerate(METRICS):
        values = differences[metric_index]
        seed_deltas = values.mean(axis=1)
        report["metrics"][metric] = {
            "mean_delta": float(values.mean()),
            "seed_deltas": seed_deltas.tolist(),
            "seed_delta_std": float(seed_deltas.std(ddof=1)),
            "hierarchical_ci95": np.quantile(
                bootstrap[:, metric_index], (0.025, 0.975)
            ).tolist(),
            "seed_file_win_rate": float((values > 0).mean()),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
