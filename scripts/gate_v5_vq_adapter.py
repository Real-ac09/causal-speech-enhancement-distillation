#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the locked V5 VQ no-harm gate.")
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--candidate-name", default="bounded_adapter")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text())
    pesq = comparison["paired_bootstrap"][args.candidate_name]["enhanced_pesq"]
    mean, interval = float(pesq["mean"]), list(map(float, pesq["ci95"]))
    keep = mean >= 0.01 and interval[0] > 0.0
    decision = {
        "deployment_vq_mode": "bounded_adapter" if keep else "train_only",
        "mean_pesq_delta": mean,
        "paired_bootstrap_ci95": interval,
        "required_delta": 0.01,
        "interval_excludes_zero": interval[0] > 0.0,
        "keep_adapter": keep,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
