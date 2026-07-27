#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch

from cnvqg.models import (
    CausalConfidenceGateV14,
    CausalOracleResidualGateV16,
    CausalOrdinalResidualGateV17,
    CausalCumulativeOrdinalGateV17,
    CausalFeatureCumulativeGateV17,
    CausalUtilitySafetyGateV17,
    CausalStatisticsUtilityGateV17,
    PredictiveNoiseVQMambaV8,
    V14NativeStreamer,
    V16NativeStreamer,
    V17NativeStreamer,
    V17CumulativeNativeStreamer,
    V17FeatureNativeStreamer,
    V17UtilityNativeStreamer,
    V17StatisticsNativeStreamer,
    V8NativeStreamer,
)
from cnvqg.models.factory import build_model


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure end-to-end, one-thread V8 streaming latency."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=8902)
    args = parser.parse_args()
    if args.seconds <= 0.0 or args.warmup_seconds < 0.0:
        parser.error("seconds must be positive and warmup-seconds non-negative")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if type(model) is PredictiveNoiseVQMambaV8:
        streamer = V8NativeStreamer(model)
    elif type(model) is CausalConfidenceGateV14:
        streamer = V14NativeStreamer(model)
    elif type(model) is CausalOracleResidualGateV16:
        streamer = V16NativeStreamer(model)
    elif type(model) is CausalOrdinalResidualGateV17:
        streamer = V17NativeStreamer(model)
    elif type(model) is CausalCumulativeOrdinalGateV17:
        streamer = V17CumulativeNativeStreamer(model)
    elif type(model) is CausalFeatureCumulativeGateV17:
        streamer = V17FeatureNativeStreamer(model)
    elif type(model) is CausalUtilitySafetyGateV17:
        streamer = V17UtilityNativeStreamer(model)
    elif type(model) is CausalStatisticsUtilityGateV17:
        streamer = V17StatisticsNativeStreamer(model)
    else:
        raise TypeError(
            "Native benchmark requires PredictiveNoiseVQMambaV8 or "
            "a validated residual gate, got "
            f"{type(model).__name__}"
        )
    state = streamer.init_state(1, "cpu", torch.float32)
    hop_seconds = model.hop_length / model.sample_rate
    warmup_frames = int(np.ceil(args.warmup_seconds / hop_seconds))
    measured_frames = int(np.ceil(args.seconds / hop_seconds))
    chunks = torch.randn(
        warmup_frames + measured_frames, 1, 1, model.hop_length
    ).mul_(0.03)

    timings = []
    state_sizes = []
    with torch.inference_mode():
        for index, chunk in enumerate(chunks):
            start = time.perf_counter_ns()
            _, state = streamer.process_chunk(chunk, state)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            if index >= warmup_frames:
                timings.append(elapsed_ms)
                state_sizes.append(state.tensor_elements())

    frame_times = np.asarray(timings, dtype=np.float64)
    percentiles = _quantiles(frame_times)
    hop_ms = 1000.0 * hop_seconds
    state_elements = max(state_sizes)
    report = {
        "checkpoint": str(args.checkpoint),
        "architecture": checkpoint["config"]["model"]["architecture"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "cpu": _cpu_name(),
        "torch_version": torch.__version__,
        "threads": 1,
        "sample_rate": model.sample_rate,
        "window_samples": model.win_length,
        "hop_samples": model.hop_length,
        "algorithmic_latency_ms": 1000.0
        * model.algorithmic_latency_samples
        / model.sample_rate,
        "hop_deadline_ms": hop_ms,
        "warmup_seconds": args.warmup_seconds,
        "measured_seconds": measured_frames * hop_seconds,
        "measured_frames": measured_frames,
        "frame_time_ms": percentiles,
        "streaming_rtf": float(frame_times.sum() / (measured_frames * hop_ms)),
        "passes_hop_deadline_p95": bool(percentiles["p95"] < hop_ms),
        "passes_hop_deadline_p99": bool(percentiles["p99"] < hop_ms),
        "state_tensor_elements_min": min(state_sizes),
        "state_tensor_elements_max": state_elements,
        "state_tensor_mebibytes_fp32": state_elements * 4 / (1024**2),
        "peak_process_rss_mebibytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
