from __future__ import annotations

from pathlib import Path

import torch

from cnvqg.models.causal_confidence_gate_v14 import (
    CausalConfidenceGateV14,
    CausalResidualConfidenceGate,
)
from cnvqg.models.predictive_noise_vq_mamba_v8 import PredictiveNoiseVQMambaV8


def _backbone_checkpoint(path: Path) -> Path:
    config = {
        "architecture": "causal_temporal_core_v12",
        "variant": "student",
        "sample_rate": 16_000,
        "n_fft": 64,
        "win_length": 32,
        "hop_length": 16,
        "magnitude_power": 0.3,
        "channels": 8,
        "noise_dim": 4,
        "temporal_layers": 1,
        "auxiliary_vq": False,
        "use_mamba": False,
        "temporal_core": "gru",
        "temporal_hidden_dim": 8,
        "time_kernel_size": 1,
        "phase_residual_scale": 0.0,
        "reconstruction_mode": "direct_scalar_mask",
        "enforce_parameter_cap": False,
    }
    architecture = config.pop("architecture")
    backbone = PredictiveNoiseVQMambaV8(**config)
    config["architecture"] = architecture
    torch.save(
        {
            "config": {"model": config},
            "model_state_dict": backbone.state_dict(),
        },
        path,
    )
    return path


def test_gate_is_causal_and_bounded() -> None:
    torch.manual_seed(14)
    gate = CausalResidualConfidenceGate(
        noise_dim=4,
        hidden_dim=8,
        minimum_strength=0.0,
        initial_strength=0.995,
    ).eval()
    noise_a = torch.randn(1, 20, 4)
    spectrum_a = torch.randn(1, 17, 20, dtype=torch.complex64)
    noise_b = noise_a.clone()
    spectrum_b = spectrum_a.clone()
    noise_b[:, 10:] = torch.randn_like(noise_b[:, 10:])
    spectrum_b[..., 10:] = torch.randn_like(spectrum_b[..., 10:])

    first = gate(noise_a, spectrum_a)
    second = gate(noise_b, spectrum_b)

    assert torch.equal(first[:, :10], second[:, :10])
    assert first.min() >= 0.0
    assert first.max() <= 1.0


def test_wrapper_freezes_backbone_and_routes_gate_gradients(
    tmp_path: Path,
) -> None:
    checkpoint = _backbone_checkpoint(tmp_path / "backbone.pt")
    model = CausalConfidenceGateV14(
        backbone_checkpoint=str(checkpoint),
        gate_hidden_dim=8,
    )
    model.train()
    noisy = torch.randn(2, 1, 320)
    output = model(noisy)
    output.enhanced.abs().mean().backward()

    assert model.backbone.training is False
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(
        parameter.grad is not None
        for parameter in model.confidence_gate.parameters()
    )
    assert model.gate_parameter_count() < 10_000
    assert output.enhanced.shape == noisy.shape
    assert output.gate_strength.shape[0] == noisy.shape[0]
    assert output.mixture_residual.abs().max() == 0.0


def test_wrapper_waveform_prefix_is_future_independent(tmp_path: Path) -> None:
    checkpoint = _backbone_checkpoint(tmp_path / "backbone.pt")
    model = CausalConfidenceGateV14(
        backbone_checkpoint=str(checkpoint),
        gate_hidden_dim=8,
    ).eval()
    first = torch.randn(1, 1, 640)
    second = first.clone()
    second[..., 320:] = torch.randn_like(second[..., 320:])

    with torch.inference_mode():
        first_output = model(first)
        second_output = model(second)

    assert torch.equal(
        first_output.gate_strength[:, :19],
        second_output.gate_strength[:, :19],
    )
    assert torch.equal(
        first_output.enhanced[..., :304],
        second_output.enhanced[..., :304],
    )
