#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["enhanced_pesq", "enhanced_stoi", "enhanced_estoi", "enhanced_si_sdr"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate three V5 seed evaluations.")
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=505)
    args = parser.parse_args()
    if len(args.evaluation) != 3:
        raise ValueError("Final V5 reporting requires exactly three seed evaluations")
    frames = [pd.read_csv(path / "per_file_metrics.csv").set_index("file_id") for path in args.evaluation]
    common = frames[0].index.intersection(frames[1].index).intersection(frames[2].index)
    rng = np.random.default_rng(args.seed)
    sample_ids = rng.integers(0, len(common), size=(args.bootstrap_samples, len(common)))
    report = {}
    for metric in METRICS:
        per_seed = np.asarray([frame.loc[common, metric].mean() for frame in frames])
        per_file_seed_mean = np.stack([frame.loc[common, metric].to_numpy() for frame in frames]).mean(0)
        bootstrap = per_file_seed_mean[sample_ids].mean(1)
        report[metric] = {"mean": float(per_seed.mean()), "std": float(per_seed.std(ddof=1)),
                          "seed_values": per_seed.tolist(),
                          "paired_bootstrap_ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
