#!/usr/bin/env python3
"""Dependency-light structural acceptance checks for the V8 model."""
from __future__ import annotations

import copy
import json

import torch

from cnvqg.models import PredictiveNoiseVQMambaV8
from cnvqg.models.factory import build_model


def small(**kwargs) -> PredictiveNoiseVQMambaV8:
    return PredictiveNoiseVQMambaV8(
        variant="student",
        channels=64,
        noise_dim=32,
        temporal_layers=1,
        use_mamba=False,
        **kwargs,
    ).eval()


def main() -> None:
    torch.manual_seed(8001)
    checks: dict[str, object] = {}

    built = build_model(
        {"architecture": "predictive_noise_vq_mamba_v8", "use_mamba": False}
    )
    checks["factory"] = isinstance(built, PredictiveNoiseVQMambaV8)
    checks["student_parameters"] = built.parameter_count()
    teacher = PredictiveNoiseVQMambaV8(variant="teacher", use_mamba=False)
    checks["teacher_parameters"] = teacher.parameter_count()
    checks["parameter_caps"] = (
        built.parameter_count() <= 1_100_000 and teacher.parameter_count() <= 2_700_000
    )

    model = small()
    waveform = torch.randn(1, 1, 1753)
    spectrum, length = model._analysis(waveform.squeeze(1), pad_end=True)
    expected = model._synthesis(spectrum, length).unsqueeze(1)
    with torch.no_grad():
        output = model(waveform)
    checks["identity_max_difference"] = float((output.enhanced - expected).abs().max())
    checks["mixture_max_residual"] = float(output.mixture_residual.abs().max())
    checks["identity"] = checks["identity_max_difference"] <= 5e-6
    checks["mixture_consistency"] = checks["mixture_max_residual"] <= 1e-6

    clone = copy.deepcopy(model)
    clone.noise_vq.codebook.normal_(100.0, 20.0)
    with torch.no_grad():
        first = model(waveform).enhanced
        second = clone(waveform).enhanced
    checks["codebook_independence"] = torch.equal(first, second)

    causal = small(auxiliary_vq=False)
    causal.decoder.mask_head.weight.data.normal_(0.0, 0.02)
    original = torch.randn(1, 1, 1920)
    changed = original.clone()
    boundary = 1280
    changed[..., boundary:] = torch.randn_like(changed[..., boundary:]) * 3.0
    with torch.no_grad():
        first = causal(original).enhanced
        second = causal(changed).enhanced
    finalized = boundary - causal.algorithmic_latency_samples
    future_difference = float((first[..., :finalized] - second[..., :finalized]).abs().max())
    checks["future_max_difference"] = future_difference
    checks["causal"] = future_difference <= 1e-6

    stream_input = torch.randn(1, 1, 1753)
    whole = causal(stream_input).enhanced
    state = causal.init_stream_state(1, "cpu", torch.float32)
    pieces = []
    offset = 0
    for size in (73, 247, 1, 640, 511, 281):
        piece, state = causal.forward_chunk(stream_input[..., offset : offset + size], state)
        pieces.append(piece)
        offset += size
    tail, _ = causal.flush(state)
    pieces.append(tail)
    stream_difference = float((torch.cat(pieces, -1) - whole).abs().max().detach())
    checks["streaming_max_difference"] = stream_difference
    checks["streaming_equivalence"] = stream_difference <= 1e-4

    trainable = small().train()
    train_output = trainable(torch.randn(2, 1, 1600))
    train_output.vq.loss.backward()
    gradient = trainable.prototype_predictor.weight.grad
    checks["prototype_shape"] = list(train_output.prototype_logits.shape)
    checks["prototype_gradient"] = bool(
        gradient is not None and torch.isfinite(gradient).all()
    )

    if torch.cuda.is_available():
        cuda_model = PredictiveNoiseVQMambaV8(
            variant="student",
            channels=64,
            noise_dim=32,
            temporal_layers=1,
            use_mamba=True,
        ).cuda().train()
        cuda_input = torch.randn(1, 1, 1280, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cuda_output = cuda_model(cuda_input)
            cuda_loss = cuda_output.enhanced.square().mean() + cuda_output.vq.loss
        cuda_loss.backward()
        checks["bf16_finite"] = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in cuda_model.parameters()
        )
    else:
        checks["bf16_finite"] = "not_run_no_cuda"

    required = (
        "factory",
        "parameter_caps",
        "identity",
        "mixture_consistency",
        "codebook_independence",
        "causal",
        "streaming_equivalence",
        "prototype_gradient",
    )
    passed = all(checks[name] is True for name in required) and checks["bf16_finite"] in (
        True,
        "not_run_no_cuda",
    )
    checks["passed"] = passed
    print(json.dumps(checks, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
