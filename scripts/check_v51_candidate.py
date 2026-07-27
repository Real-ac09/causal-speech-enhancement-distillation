#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import yaml

from cnvqg.models.factory import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hard structural gates for a V5.1 candidate.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = build_model(config["model"]).to(device).eval()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    parameters = sum(parameter.numel() for parameter in model.parameters())
    cap = int(getattr(model, "parameter_cap", 2_700_000))
    latency_ms = 1000.0 * model.algorithmic_latency_samples / model.sample_rate

    # Random reconstruction heads ensure an identity-initialised model cannot
    # conceal future leakage in its feature extractor or normalisation layers.
    causal_probe = copy.deepcopy(model).eval()
    torch.manual_seed(710)
    decoder = getattr(causal_probe, "decoder", None)
    for name in ("magnitude_head", "phase_head", "mask_head"):
        head = getattr(decoder, name, None)
        if head is not None:
            head.weight.data.normal_(0.0, 0.02)
    original = torch.randn(1, 1, 1920, device=device)
    changed = original.clone()
    boundary = 1280
    changed[..., boundary:] = 3.0 * torch.randn_like(changed[..., boundary:])
    with torch.inference_mode():
        first = causal_probe(original).enhanced
        second = causal_probe(changed).enhanced
    finalized = boundary - causal_probe.algorithmic_latency_samples
    future_difference = float(
        (first[..., :finalized] - second[..., :finalized]).abs().max().cpu()
    )

    waveform = torch.randn(1, 1, 1753, device=device)
    with torch.inference_mode():
        whole = model(waveform).enhanced
        state = model.init_stream_state(1, device, waveform.dtype)
        outputs = []
        offset = 0
        for size in (73, 247, 1, 640, 511, 281):
            output, state = model.forward_chunk(waveform[..., offset : offset + size], state)
            outputs.append(output)
            offset += size
        tail, _ = model.flush(state)
        outputs.append(tail)
    stream_difference = float((torch.cat(outputs, -1) - whole).abs().max().cpu())

    finite_gradients = True
    if device.type == "cuda":
        gradient_probe = copy.deepcopy(model).train()
        audio = torch.randn(1, 1, 960, device=device)
        gradient_probe.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = gradient_probe(audio)
            loss = result.enhanced.square().mean() + result.vq.loss
        loss.backward()
        finite_gradients = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in gradient_probe.parameters()
        )
        del gradient_probe

    gates = {
        "parameter_cap": parameters <= cap,
        "algorithmic_latency": latency_ms <= 20.0,
        "future_independence": future_difference <= 1e-6,
        "streaming_equivalence": stream_difference <= 1e-4,
        "finite_bf16_gradients": finite_gradients,
    }
    report = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "architecture": config["model"]["architecture"],
        "parameters": parameters,
        "parameter_cap": cap,
        "algorithmic_latency_ms": latency_ms,
        "future_max_abs_difference": future_difference,
        "streaming_max_abs_difference": stream_difference,
        "gates": gates,
        "passed": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
