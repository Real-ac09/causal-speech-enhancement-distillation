#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratify per-file enhancement metrics.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-by", nargs="*", default=["speaker_id", "noise_type", "snr_db"])
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_csv(args.metrics)
    if args.metadata:
        metadata = pd.read_csv(args.metadata)
        extra = [column for column in metadata.columns if column not in frame.columns or column == "file_id"]
        frame = frame.merge(metadata[extra], on="file_id", how="left")
    metric_columns = [
        column for column in frame.columns
        if column.startswith("enhanced_") or column == "si_sdr_improvement"
    ]
    rng = np.random.default_rng(args.seed)
    summary: dict[str, object] = {"items": len(frame), "overall": {}, "groups": {}, "worst": {}}
    for metric in metric_columns:
        values = frame[metric].dropna().to_numpy(float)
        boot = np.empty(args.bootstrap_samples)
        for index in range(args.bootstrap_samples):
            boot[index] = rng.choice(values, size=len(values), replace=True).mean()
        summary["overall"][metric] = {
            "mean": float(values.mean()),
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        }
        worst = frame.nsmallest(min(20, len(frame)), metric)[["file_id", metric]]
        summary["worst"][metric] = worst.to_dict(orient="records")

    for group in args.group_by:
        if group not in frame.columns:
            continue
        grouped = frame.groupby(group, dropna=False)[metric_columns].agg(["count", "mean"])
        summary["groups"][group] = {
            str(index): {
                metric: {
                    "count": int(row[(metric, "count")]),
                    "mean": float(row[(metric, "mean")]),
                }
                for metric in metric_columns
            }
            for index, row in grouped.iterrows()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
