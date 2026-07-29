#!/usr/bin/env python3
"""Compare temporal-core latency before committing to full training runs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cnvqg.models import PredictiveNoiseVQMambaV8, V8NativeStreamer


VARIANTS: dict[str, dict[str, object]] = {
    "mamba_control": {
        "use_mamba": True,
        "temporal_core": "mamba",
        "time_kernel_size": 3,
    },
    "gru_matched": {
        "use_mamba": False,
        "temporal_core": "gru",
        "temporal_hidden_dim": 232,
        "time_kernel_size": 3,
    },
    "gru_matched_time1": {
        "use_mamba": False,
        "temporal_core": "gru",
        "temporal_hidden_dim": 232,
        "time_kernel_size": 1,
    },
    "gru128_time1": {
        "use_mamba": False,
        "temporal_core": "gru",
        "temporal_hidden_dim": 128,
        "time_kernel_size": 1,
    },
}


def _measure(
    model: PredictiveNoiseVQMambaV8,
    warmup_frames: int,
    measured_frames: int,
    seed: int,
) -> dict[str, object]:
    streamer = V8NativeStreamer(model.eval())
    state = streamer.init_state(1, "cpu", torch.float32)
    generator = torch.Generator().manual_seed(seed)
    chunks = torch.randn(
        warmup_frames + measured_frames,
        1,
        1,
        model.hop_length,
        generator=generator,
    ).mul_(0.03)
    timings = []
    state_sizes = []
    with torch.inference_mode():
        for index, chunk in enumerate(chunks):
            start = time.perf_counter_ns()
            _, state = streamer.process_chunk(chunk, state)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if index >= warmup_frames:
                timings.append(elapsed)
                state_sizes.append(state.tensor_elements())
    values = np.asarray(timings)
    hop_ms = 1000.0 * model.hop_length / model.sample_rate
    return {
        "parameters": model.parameter_count(),
        "temporal_parameters": sum(
            parameter.numel() for parameter in model.bottleneck.temporal.parameters()
        ),
        "frame_time_ms": {
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "mean": float(values.mean()),
            "maximum": float(values.max()),
        },
        "streaming_rtf": float(values.sum() / (measured_frames * hop_ms)),
        "passes_p95_deadline": bool(np.quantile(values, 0.95) < hop_ms),
        "state_tensor_elements": max(state_sizes),
        "state_tensor_mebibytes_fp32": max(state_sizes) * 4 / (1024**2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-frames", type=int, default=50)
    parser.add_argument("--measured-frames", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup_frames < 2 or args.measured_frames < 1:
        parser.error("warmup-frames must be at least 2 and measured-frames positive")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report: dict[str, object] = {
        "purpose": "Structural latency only; random weights are not quality results.",
        "threads": 1,
        "hop_deadline_ms": 10.0,
        "variants": {},
    }
    for index, (name, overrides) in enumerate(VARIANTS.items()):
        torch.manual_seed(args.seed)
        model = PredictiveNoiseVQMambaV8(
            variant="student",
            auxiliary_vq=False,
            phase_residual_scale=0.0,
            reconstruction_mode="direct_scalar_mask",
            **overrides,
        )
        report["variants"][name] = _measure(
            model,
            warmup_frames=args.warmup_frames,
            measured_frames=args.measured_frames,
            seed=args.seed + index,
        )

    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
