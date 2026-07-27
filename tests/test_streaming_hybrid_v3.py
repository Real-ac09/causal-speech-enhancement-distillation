from __future__ import annotations

import torch

from cnvqg.models.factory import build_model
from cnvqg.models.streaming_hybrid_v3 import StreamingHybridCNVQGV3


def test_factory_builds_ordered_v3_sizes() -> None:
    sizes = []
    for variant in ("tiny", "student", "teacher"):
        model = build_model({
            "architecture": "streaming_hybrid_v3",
            "variant": variant,
            "use_mamba": False,
        })
        sizes.append(sum(parameter.numel() for parameter in model.parameters()))
    assert sizes[0] < sizes[1] < sizes[2]


def test_v3_compresses_tf_noise_conditioning() -> None:
    model = StreamingHybridCNVQGV3(
        variant="tiny", tf_condition_dim=8, use_mamba=False
    ).eval()
    assert model.tf_input[0].conv.in_channels == 13
    assert model.tf_condition_compression == 8.0
    noise = torch.randn(1, 9, model.noise_dim)
    aligned = model._align_noise_to_tf(noise, frames=7, bins=17)
    assert aligned.shape == (1, 8, 17, 7)


def test_v3_forward_is_identity_safe_at_tf_initialization() -> None:
    torch.manual_seed(0)
    model = StreamingHybridCNVQGV3(variant="tiny", use_mamba=False).eval()
    noisy = torch.randn(1, 1, 4096)
    with torch.no_grad():
        output = model(noisy)
    assert output.enhanced.shape == noisy.shape
    torch.testing.assert_close(output.gain, torch.ones_like(output.gain))
    torch.testing.assert_close(output.phase_delta, torch.zeros_like(output.phase_delta))
    torch.testing.assert_close(output.enhanced, output.base_enhanced, atol=2e-5, rtol=2e-5)


def test_waveform_only_v3_freezes_all_tf_conditioning() -> None:
    model = StreamingHybridCNVQGV3(
        variant="tiny", use_mamba=False, enable_tf_refiner=False
    )
    assert all(not parameter.requires_grad for parameter in model.tf_input.parameters())
    assert all(
        not parameter.requires_grad for parameter in model.noise_tf_projection.parameters()
    )


def test_v3_joint_handles_arbitrary_utterance_length() -> None:
    model = StreamingHybridCNVQGV3(variant="tiny", use_mamba=False).eval()
    noisy = torch.randn(1, 1, 68173)
    with torch.no_grad():
        output = model(noisy)
    assert output.enhanced.shape == noisy.shape
