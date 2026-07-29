#!/usr/bin/env python3
"""Run the isolated continuous-noise-conditioning gate on V9.2."""

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
    name = "v95_continuous_noise_bf16"
    config = configure(
        name,
        "causal_noise_conditioned_mamba_v95",
        use_noise_conditioning=True,
        use_auxiliary_vq=False,
        use_phase_detail=False,
        phase_residual_limit=0.0,
    )
    # Match the original BF16 V9.2 control seed; the conditioning projection is
    # zero initialized, so this begins as the exact control enhancement path.
    config["project"]["seed"] = 9201
    run(name, config)
    result = best_metrics(name)
    result["promoted"] = result["pesq"] >= BASELINE_PESQ + MINIMUM_GAIN
    summary = {
        "control": {"name": "v92_one_file_capacity_bf16", "pesq": BASELINE_PESQ},
        "promotion_threshold": BASELINE_PESQ + MINIMUM_GAIN,
        "candidate": result,
        "vq_enabled": False,
        "phase_enabled": False,
    }
    result_dir = ROOT / "results/v9/v95_noise_conditioning_ablation"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
