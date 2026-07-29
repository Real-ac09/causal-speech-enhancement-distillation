from __future__ import annotations

import pytest
import torch

from cnvqg.models.causal_low_rank_frequency_mamba_v93 import (
    CausalLowRankFrequencyMambaV93,
)
from cnvqg.models.factory import build_model


def _small(**overrides) -> CausalLowRankFrequencyMambaV93:
    kwargs = dict(
        variant="student",
        detail_channels=16,
        half_channels=24,
        core_channels=32,
        blocks=1,
        frequency_rank=8,
        use_mamba=False,
        enforce_parameter_cap=False,
    )
    kwargs.update(overrides)
    return CausalLowRankFrequencyMambaV93(**kwargs)


def test_factory_parameter_caps_and_rank() -> None:
    student = build_model(
        {"architecture": "cnvqg_v93", "variant": "student", "use_mamba": False}
    )
    teacher = CausalLowRankFrequencyMambaV93(variant="teacher", use_mamba=False)
    assert student.parameter_count() <= 1_100_000
    assert teacher.parameter_count() <= 2_700_000
    assert student.frequency_rank == 32
    assert teacher.frequency_rank == 48


def test_adapter_is_zero_initialised_identity() -> None:
    model = _small().eval()
    core = torch.randn(2, model.core_channels, 129, 7)
    with torch.no_grad():
        assert torch.equal(model.frequency_context(core), core)
        output = model(torch.randn(1, 1, 1600))
    assert torch.equal(output.magnitude_mask, torch.ones_like(output.magnitude_mask))


def test_frequency_adapter_has_no_time_mixing() -> None:
    model = _small().eval()
    with torch.no_grad():
        model.frequency_context.output_projection.weight.normal_(std=0.02)
    core = torch.randn(1, model.core_channels, 129, 8)
    changed = core.clone()
    changed[..., 5:] = torch.randn_like(changed[..., 5:])
    with torch.no_grad():
        first = model.frequency_context(core)
        second = model.frequency_context(changed)
    assert torch.allclose(first[..., :5], second[..., :5], atol=1e-6, rtol=1e-6)


def test_waveform_path_remains_causal() -> None:
    model = _small().eval()
    noisy = torch.randn(1, 1, 3200)
    changed = noisy.clone()
    changed[..., 2080:] = torch.randn_like(changed[..., 2080:])
    with torch.no_grad():
        first = model(noisy).mamba_features[..., :12]
        second = model(changed).mamba_features[..., :12]
    assert torch.allclose(first, second, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for BF16 Mamba")
def test_bf16_forward_backward_is_finite() -> None:
    model = _small(use_mamba=True).cuda().to(torch.bfloat16).train()
    noisy = torch.randn(1, 1, 1600, device="cuda", dtype=torch.bfloat16)
    output = model(noisy)
    output.enhanced.float().square().mean().backward()
    assert torch.isfinite(output.enhanced).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
