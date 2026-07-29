from __future__ import annotations

import torch

from cnvqg.models.factory import build_model
from cnvqg.models.noise_adaptive_tf_mamba import NoiseAdaptiveTFMamba


def test_v4_factory_and_size_order() -> None:
    sizes = []
    for variant in ("tiny", "base", "large"):
        model = build_model({
            "architecture": "noise_adaptive_tf_mamba",
            "variant": variant,
            "use_mamba": False,
        })
        sizes.append(sum(parameter.numel() for parameter in model.parameters()))
    assert sizes[0] < sizes[1] < sizes[2]


def test_v4_forward_shapes_and_halting_distribution() -> None:
    model = NoiseAdaptiveTFMamba(variant="tiny", use_mamba=False).eval()
    noisy = torch.randn(2, 1, 8192)
    with torch.no_grad():
        output = model(noisy)
    assert output.enhanced.shape == noisy.shape
    assert output.estimated_magnitude.shape == output.estimated_phase.shape
    assert output.phase_confidence.shape == output.estimated_phase.shape
    assert output.halting_probabilities.shape == (2, model.max_iterations)
    torch.testing.assert_close(
        output.halting_probabilities.sum(dim=-1), torch.ones(2), atol=1e-6, rtol=1e-6
    )
    assert torch.all(output.expected_iterations >= 1.0)
    assert torch.all(output.expected_iterations <= model.max_iterations)


def test_v4_ablation_switches_disable_vq_and_adaptive_depth() -> None:
    model = NoiseAdaptiveTFMamba(
        variant="tiny",
        use_mamba=False,
        use_noise_codebook=False,
        condition_dynamics=False,
        adaptive_iterations=False,
        max_iterations=2,
    ).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 1, 4096))
    assert int(output.code_indices.min()) == -1
    torch.testing.assert_close(output.expected_iterations, torch.tensor([2.0]))


def test_v4_backward_reaches_mask_phase_and_halting_heads() -> None:
    model = NoiseAdaptiveTFMamba(variant="tiny", use_mamba=False).train()
    output = model(torch.randn(1, 1, 4096))
    loss = output.enhanced.square().mean() + 0.01 * output.expected_iterations.mean()
    loss.backward()
    assert model.decoder[-1].weight.grad is not None
    assert model.halting_head.weight.grad is not None
