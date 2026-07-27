#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["enhanced_pesq", "enhanced_stoi", "enhanced_estoi", "enhanced_si_sdr"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratify V5 metrics and bootstrap uncertainty.")
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=505)
    args = parser.parse_args()
    frame = pd.read_csv(args.evaluation / "per_file_metrics.csv")
    metadata = pd.read_csv(args.metadata)
    frame = frame.merge(metadata[[column for column in metadata if column in {
        "file_id", "duration_seconds", "noise_type", "input_snr_db"}]], on="file_id", how="left")
    frame["duration"] = pd.qcut(frame["duration_seconds"], 3, labels=["short", "medium", "long"])
    frame["difficulty"] = pd.qcut(frame["noisy_pesq"], 3, labels=["hard", "medium", "easy"])
    if "noise_type" not in frame:
        frame["noise_type"] = "not_available_in_metadata"
    if "input_snr_db" in frame:
        frame["input_snr"] = pd.cut(frame["input_snr_db"], [-np.inf, 5, 10, np.inf])
    groups = ["duration", "difficulty", "noise_type"] + (["input_snr"] if "input_snr" in frame else [])
    stratified = {}
    for group in groups:
        stratified[group] = frame.groupby(group, observed=True)[METRICS].mean().reset_index().to_dict("records")
    rng = np.random.default_rng(args.seed)
    sample_ids = rng.integers(0, len(frame), size=(args.bootstrap_samples, len(frame)))
    bootstrap = {}
    for metric in METRICS:
        samples = frame[metric].to_numpy(float)[sample_ids].mean(1)
        bootstrap[metric] = {"mean": float(frame[metric].mean()),
                             "ci95": np.quantile(samples, [0.025, 0.975]).tolist()}
    report = {"count": len(frame), "bootstrap": bootstrap, "stratified": stratified}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
