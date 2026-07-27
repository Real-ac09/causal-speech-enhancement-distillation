#!/usr/bin/env python3
"""Prepare Recipe-7b metadata using explicit one-second prefix targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_v17_recipe5_targets import _prepare


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix-root",
        type=Path,
        default=Path("results/v17/recipe7_prefix_oracles"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/v17_recipe7"),
    )
    parser.add_argument("--feasible-temperature", type=float, default=0.25)
    parser.add_argument("--fallback-temperature", type=float, default=0.25)
    parser.add_argument("--fallback-utility-weight", type=float, default=0.01)
    args = parser.parse_args()
    report = {
        "status": "complete",
        "role": "one_second_causal_prefix_training_targets",
        "prefix_seconds": 1.0,
        "burn_in_fraction": 0.5,
        "splits": {},
        "development_set_used": False,
        "external_test_used": False,
    }
    for split in ("train", "calibration"):
        report["splits"][split] = _prepare(
            ROOT / "results/v17/recipe2_local_oracles" / split / "metadata.csv",
            ROOT / args.prefix_root / split / "strength_candidates.csv",
            ROOT / args.output_dir / f"{split}.csv",
            args.feasible_temperature,
            args.fallback_temperature,
            args.fallback_utility_weight,
        )
    output = ROOT / args.output_dir / "summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
