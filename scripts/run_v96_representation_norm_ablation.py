#!/usr/bin/env python3
"""Gate V8-style input representation and normalization on the V9.2 control."""

from __future__ import annotations

import json

from run_v94_topology_ablation import (
    BASELINE_PESQ,
    MINIMUM_GAIN,
    ROOT,
    best_metrics,
    configure,
    run,
)


def main() -> None:
    candidates = {
        "v96_v8_input_bf16": configure(
            "v96_v8_input_bf16", "causal_representation_norm_mamba_v96",
            v8_input_features=True, frame_group_norm=False,
        ),
        "v96_frame_group_norm_bf16": configure(
            "v96_frame_group_norm_bf16", "causal_representation_norm_mamba_v96",
            v8_input_features=False, frame_group_norm=True,
        ),
    }
    for config in candidates.values():
        config["project"]["seed"] = 9201

    summary: dict = {
        "control": {"name": "v92_one_file_capacity_bf16", "pesq": BASELINE_PESQ},
        "promotion_threshold": BASELINE_PESQ + MINIMUM_GAIN,
        "candidates": {},
    }
    for name, config in candidates.items():
        run(name, config)
        metrics = best_metrics(name)
        metrics["promoted"] = metrics["pesq"] >= BASELINE_PESQ + MINIMUM_GAIN
        summary["candidates"][name] = metrics

    winners = [name for name, value in summary["candidates"].items() if value["promoted"]]
    summary["independent_winners"] = winners
    if len(winners) == 2:
        name = "v96_v8_input_group_norm_bf16"
        config = configure(
            name, "causal_representation_norm_mamba_v96",
            v8_input_features=True, frame_group_norm=True,
        )
        config["project"]["seed"] = 9201
        run(name, config)
        summary["combined"] = best_metrics(name)
    else:
        summary["combined"] = {
            "skipped": True,
            "reason": "Both changes must improve independently before combination.",
        }

    result_dir = ROOT / "results/v9/v96_representation_norm_ablation"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
