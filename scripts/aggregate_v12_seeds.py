#!/usr/bin/env python3
"""Aggregate paired V12 evaluations across training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ("enhanced_pesq", "enhanced_si_sdr", "enhanced_stoi", "enhanced_estoi")


def _load(directory: Path) -> pd.DataFrame:
    path = directory / "per_file_metrics.csv"
    frame = pd.read_csv(path).set_index("file_id").sort_index()
    if frame.index.has_duplicates:
        raise ValueError(f"{path} contains duplicate file_id values")
    return frame


def _ci(values: np.ndarray) -> list[float]:
    return np.quantile(values, [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate paired reference/candidate V12 evaluations."
    )
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--reference-name", default="mamba_control")
    parser.add_argument("--candidate-name", default="gru_matched_time1")
    parser.add_argument("--noninferiority-margin", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=12_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if len(args.reference) != len(args.candidate):
        raise ValueError("Reference and candidate evaluation counts must match")
    if len(args.reference) < 2:
        raise ValueError("Seed-aware aggregation requires at least two seed pairs")

    references = [_load(path) for path in args.reference]
    candidates = [_load(path) for path in args.candidate]
    expected_ids = references[0].index
    for frame in [*references[1:], *candidates]:
        if not frame.index.equals(expected_ids):
            raise ValueError("All evaluations must contain identical ordered file_id values")

    rng = np.random.default_rng(args.bootstrap_seed)
    num_seeds = len(references)
    num_files = len(expected_ids)
    report: dict[str, object] = {
        "reference": args.reference_name,
        "candidate": args.candidate_name,
        "num_seeds": num_seeds,
        "num_files_per_seed": num_files,
        "noninferiority_margin": args.noninferiority_margin,
        "metrics": {},
    }

    for metric in METRICS:
        reference = np.stack([frame[metric].to_numpy() for frame in references])
        candidate = np.stack([frame[metric].to_numpy() for frame in candidates])
        paired_delta = candidate - reference
        reference_by_seed = reference.mean(axis=1)
        candidate_by_seed = candidate.mean(axis=1)
        delta_by_seed = paired_delta.mean(axis=1)

        file_indices = rng.integers(
            0, num_files, size=(args.bootstrap_samples, num_files)
        )
        fixed_seed_file_delta = paired_delta.mean(axis=0)
        fixed_seed_bootstrap = fixed_seed_file_delta[file_indices].mean(axis=1)

        hierarchical_bootstrap = np.empty(args.bootstrap_samples, dtype=np.float64)
        sampled_seeds = rng.integers(
            0, num_seeds, size=(args.bootstrap_samples, num_seeds)
        )
        for sample in range(args.bootstrap_samples):
            seed_means = [
                paired_delta[seed, file_indices[sample]].mean()
                for seed in sampled_seeds[sample]
            ]
            hierarchical_bootstrap[sample] = np.mean(seed_means)

        fixed_ci = _ci(fixed_seed_bootstrap)
        hierarchical_ci = _ci(hierarchical_bootstrap)
        metric_report: dict[str, object] = {
            "reference_mean": float(reference_by_seed.mean()),
            "reference_std": float(reference_by_seed.std(ddof=1)),
            "candidate_mean": float(candidate_by_seed.mean()),
            "candidate_std": float(candidate_by_seed.std(ddof=1)),
            "delta_mean": float(delta_by_seed.mean()),
            "delta_std": float(delta_by_seed.std(ddof=1)),
            "reference_seed_values": reference_by_seed.tolist(),
            "candidate_seed_values": candidate_by_seed.tolist(),
            "delta_seed_values": delta_by_seed.tolist(),
            "fixed_seeds_paired_bootstrap_ci95": fixed_ci,
            "hierarchical_seed_file_bootstrap_ci95": hierarchical_ci,
        }
        if metric == "enhanced_pesq":
            boundary = -args.noninferiority_margin
            metric_report["noninferiority"] = {
                "boundary": boundary,
                "point_estimate_passes": bool(delta_by_seed.mean() > boundary),
                "fixed_seeds_ci_passes": bool(fixed_ci[0] > boundary),
                "hierarchical_ci_passes": bool(hierarchical_ci[0] > boundary),
            }
        report["metrics"][metric] = metric_report

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
