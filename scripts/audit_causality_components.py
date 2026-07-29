#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

from cnvqg.models.factory import build_model
from cnvqg.models.streaming_hybrid_v2 import CausalConv2d


def named_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=CHECKPOINT")
    name, path = value.split("=", 1)
    return name, Path(path)


def first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def max_prefix_difference(first: torch.Tensor, second: torch.Tensor, end: int) -> float:
    if end <= 0:
        return float("nan")
    return float((first[..., :end] - second[..., :end]).abs().max().cpu())


def audit(name: str, path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(checkpoint["config"]["model"]).to(device)
    state = checkpoint.get("ema_model_state_dict") or checkpoint["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    captures: dict[str, list[torch.Tensor]] = {}
    hooks = []
    for module_name in ("encoder", "cell", "core", "decoder"):
        module = getattr(model, module_name, None)
        if module is None:
            continue

        def save(_module, _inputs, output, key=module_name):
            tensor = first_tensor(output)
            if tensor is not None:
                captures.setdefault(key, []).append(tensor.detach())

        hooks.append(module.register_forward_hook(save))

    length = 4096
    boundary = 2880
    torch.manual_seed(4501)
    original = torch.randn(1, 1, length, device=device)
    changed = original.clone()
    changed[..., boundary:] = 3.0 * torch.randn_like(changed[..., boundary:])
    latency = int(getattr(model, "algorithmic_latency_samples", model.win_length))
    finalized = max(0, boundary - latency)
    with torch.inference_mode():
        first_output = model(original).enhanced
        first_captures = {key: values[-1] for key, values in captures.items()}
        captures.clear()
        second_output = model(changed).enhanced
        second_captures = {key: values[-1] for key, values in captures.items()}
    for hook in hooks:
        hook.remove()

    causal_frames = max(0, 1 + (boundary - int(model.win_length)) // int(model.hop_length))
    feature_differences = {
        key: max_prefix_difference(first_captures[key], second_captures[key], causal_frames)
        for key in first_captures.keys() & second_captures.keys()
    }
    output_difference = max_prefix_difference(first_output, second_output, finalized)

    stream_difference = None
    if all(hasattr(model, attr) for attr in ("init_stream_state", "forward_chunk", "flush")):
        waveform = torch.randn(1, 1, 1753, device=device)
        with torch.inference_mode():
            whole = model(waveform).enhanced
            stream_state = model.init_stream_state(1, device, waveform.dtype)
            pieces = []
            offset = 0
            for size in (73, 247, 1, 640, 511, 281):
                piece, stream_state = model.forward_chunk(
                    waveform[..., offset : offset + size], stream_state
                )
                pieces.append(piece)
                offset += size
            tail, _ = model.flush(stream_state)
            pieces.append(tail)
        stream_difference = float((torch.cat(pieces, -1) - whole).abs().max().cpu())

    vq_independence = None
    if getattr(model, "vq_mode", None) == "disabled" and hasattr(model, "noise_vq"):
        probe = original[..., :1920]
        with torch.inference_mode():
            before = model(probe).enhanced
            model.noise_vq.codebook.copy_(torch.randn_like(model.noise_vq.codebook) * 100.0)
            after = model(probe).enhanced
        vq_independence = bool(torch.equal(before, after))

    modules = list(model.modules())
    frame_group_norms = sum(
        module.__class__.__name__ == "FrameGroupNorm" for module in modules
    )
    all_group_norms = sum(isinstance(module, torch.nn.GroupNorm) for module in modules)
    ordinary_group_norms = max(0, all_group_norms - frame_group_norms)
    report = {
        "checkpoint": str(path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "architecture": checkpoint["config"]["model"]["architecture"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "center": bool(getattr(model, "center", False)),
        "win_length": int(model.win_length),
        "hop_length": int(model.hop_length),
        "algorithmic_latency_ms": 1000.0 * latency / int(model.sample_rate),
        "causal_conv2d_modules": sum(isinstance(module, CausalConv2d) for module in modules),
        "frame_local_group_norm_modules": frame_group_norms,
        "ordinary_group_norm_modules": ordinary_group_norms,
        "future_output_max_abs_difference": output_difference,
        "future_feature_max_abs_difference": feature_differences,
        "streaming_max_abs_difference": stream_difference,
        "vq_off_bitwise_independent": vq_independence,
        "future_independence_pass": output_difference <= 1e-6,
        "streaming_equivalence_pass": stream_difference is None or stream_difference <= 1e-4,
    }
    return {"name": name, **report}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit causality by model component.")
    parser.add_argument("--model", type=named_checkpoint, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    results = [audit(name, path, device) for name, path in args.model]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"models": results}, indent=2) + "\n")
    print(json.dumps({"models": results}, indent=2))


if __name__ == "__main__":
    main()
