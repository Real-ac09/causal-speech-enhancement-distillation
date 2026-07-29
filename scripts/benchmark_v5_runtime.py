#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cnvqg.models.factory import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure V5 one-thread streaming runtime.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=0.25,
                        help="Reference streamer is intentionally benchmarked briefly; native kernels may use longer runs.")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    hop = model.hop_length
    waveform = torch.randn(1, 1, int(args.seconds * model.sample_rate)) * 0.03
    state = model.init_stream_state(1, "cpu", waveform.dtype)
    timings = []
    with torch.inference_mode():
        for chunk_index, chunk in enumerate(waveform.split(hop, dim=-1)):
            start = time.perf_counter_ns()
            _, state = model.forward_chunk(chunk, state)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            if chunk_index >= 2:
                timings.append(elapsed_ms)
        start = time.perf_counter_ns()
        _, state = model.flush(state)
        flush_ms = (time.perf_counter_ns() - start) / 1e6
    frame = np.asarray(timings)
    report = {
        "threads": 1,
        "algorithmic_latency_ms": 1000.0 * model.algorithmic_latency_samples / model.sample_rate,
        "hop_ms": 1000.0 * hop / model.sample_rate,
        "frame_time_p50_ms": float(np.quantile(frame, 0.50)),
        "frame_time_p95_ms": float(np.quantile(frame, 0.95)),
        "frame_time_max_ms": float(frame.max()),
        "streaming_rtf": float(frame.sum() / (len(frame) * 1000.0 * hop / model.sample_rate)),
        "flush_ms": flush_ms,
        "passes_realtime_gate": bool(np.quantile(frame, 0.95) < 1000.0 * hop / model.sample_rate),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
