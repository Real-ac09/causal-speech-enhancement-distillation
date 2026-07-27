#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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


def evaluation_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=EVALUATION_DIRECTORY")
    name, directory = value.split("=", 1)
    if not name or not directory:
        raise argparse.ArgumentTypeError("Expected NAME=EVALUATION_DIRECTORY")
    return name, Path(directory)


def read_evaluation(name: str, directory: Path) -> tuple[dict[str, object], pd.DataFrame]:
    summary_path = directory / "summary.json"
    per_file_path = directory / "per_file_metrics.csv"
    with summary_path.open() as file:
        summary = json.load(file)
    metrics = summary.get("metrics", summary)
    row: dict[str, object] = {"name": name, "directory": str(directory)}
    row.update({metric: float(metrics[metric]) for metric in METRICS})
    frame = pd.read_csv(per_file_path).set_index("file_id").sort_index()
    return row, frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare paired enhancement evaluations.")
    parser.add_argument("--reference", type=evaluation_argument, required=True)
    parser.add_argument("--candidate", type=evaluation_argument, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    reference_row, reference_frame = read_evaluation(*args.reference)
    rows = [reference_row]
    paired: dict[str, object] = {}
    rng = np.random.default_rng(args.seed)

    for candidate in args.candidate:
        row, frame = read_evaluation(*candidate)
        frame, aligned_reference = frame.align(reference_frame, join="inner", axis=0)
        sample_ids = rng.integers(
            0,
            len(frame),
            size=(args.bootstrap_samples, len(frame)),
        )
        candidate_deltas: dict[str, object] = {}
        for metric in METRICS:
            delta = (frame[metric] - aligned_reference[metric]).to_numpy(float)
            bootstrapped = delta[sample_ids].mean(axis=1)
            low, high = np.quantile(bootstrapped, (0.025, 0.975))
            mean_delta = float(delta.mean())
            row[f"delta_{metric}"] = mean_delta
            candidate_deltas[metric] = {
                "mean": mean_delta,
                "ci95": [float(low), float(high)],
                "win_rate": float((delta > 0).mean()),
            }
        rows.append(row)
        paired[str(row["name"])] = candidate_deltas

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output_dir / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "comparison.json").open("w") as file:
        json.dump(
            {
                "reference": str(reference_row["name"]),
                "results": rows,
                "paired_bootstrap": paired,
            },
            file,
            indent=2,
        )

    print(f"{'name':28s} {'PESQ':>8s} {'delta':>9s} {'SI-SDR':>9s} {'delta':>9s}")
    for row in rows:
        print(
            f"{str(row['name']):28s} {float(row['enhanced_pesq']):8.4f} "
            f"{float(row.get('delta_enhanced_pesq', 0.0)):+9.4f} "
            f"{float(row['enhanced_si_sdr']):9.3f} "
            f"{float(row.get('delta_enhanced_si_sdr', 0.0)):+9.3f}"
        )


if __name__ == "__main__":
    main()
