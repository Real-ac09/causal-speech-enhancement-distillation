#!/usr/bin/env python3
"""Prepare, but do not launch, Recipe-7b confirmation seed configs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17049, 17050, 17051)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-config",
        type=Path,
        default=Path(
            "configs/v17/utility_safety_recipe7b_prefix_seed17048.yaml"
        ),
    )
    parser.add_argument(
        "--primary-summary",
        type=Path,
        default=Path("results/v17/recipe7b_selection/summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs/v17/generated/recipe7b_confirmation"),
    )
    args = parser.parse_args()
    summary_path = ROOT / args.primary_summary
    if not summary_path.is_file():
        raise RuntimeError(
            "Recipe 7b primary result is not ready; confirmation remains gated"
        )
    summary = json.loads(summary_path.read_text())
    if summary.get("decision") != "advance_recipe7b_to_confirmation":
        raise RuntimeError(
            "Recipe 7b did not pass the confirmation entry gate"
        )
    primary_path = ROOT / args.primary_config
    primary = yaml.safe_load(primary_path.read_text())
    selected_epoch = int(
        Path(summary["selected_checkpoint"]).stem.split("_")[-1]
    )
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "prepared",
        "primary_config": str(args.primary_config),
        "primary_selected_epoch": selected_epoch,
        "configs": [],
        "launched": False,
    }
    for seed in SEEDS:
        config = copy.deepcopy(primary)
        config["project"]["seed"] = seed
        experiment = f"utility_safety_recipe7b_confirmation_seed{seed}"
        config["paths"]["experiment_name"] = experiment
        path = output_dir / f"seed{seed}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        manifest["configs"].append(
            {
                "seed": seed,
                "config": str(path.relative_to(ROOT)),
                "fixed_evaluation_epoch": selected_epoch,
            }
        )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
