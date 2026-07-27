from __future__ import annotations

import pytest
import torch

from cnvqg.models.causal_noise_conditioned_mamba_v95 import (
    CausalNoiseConditionedMambaV95,
)
from cnvqg.models.causal_single_scale_mamba_v92 import CausalSingleScaleMambaV92
from cnvqg.models.factory import build_model


def _small(**overrides) -> CausalNoiseConditionedMambaV95:
    kwargs = dict(
        variant="student",
        detail_channels=16,
        half_channels=24,
        core_channels=32,
        blocks=1,
        noise_dim=12,
        use_mamba=False,
        enforce_parameter_cap=False,
    )
    kwargs.update(overrides)
    return CausalNoiseConditionedMambaV95(**kwargs)


def test_factory_parameter_caps() -> None:
    student = build_model({"architecture": "cnvqg_v95", "variant": "student", "use_mamba": False})
    teacher = CausalNoiseConditionedMambaV95(variant="teacher", use_mamba=False)
    assert student.parameter_count() <= 1_100_000
    assert teacher.parameter_count() <= 2_700_000


def test_zero_initialised_conditioning_matches_control() -> None:
    common = dict(
        variant="student", detail_channels=16, half_channels=24, core_channels=32,
        blocks=1, noise_dim=12, use_mamba=False, enforce_parameter_cap=False,
    )
    torch.manual_seed(42)
    control = CausalSingleScaleMambaV92(**common).eval()
    torch.manual_seed(42)
    conditioned = CausalNoiseConditionedMambaV95(**common).eval()
    noisy = torch.randn(1, 1, 1920)
    with torch.no_grad():
        expected = control(noisy)
        actual = conditioned(noisy)
    assert torch.equal(actual.enhanced, expected.enhanced)
    assert actual.continuous_noise_state.shape[-1] == 12
    assert torch.count_nonzero(actual.continuous_noise_state) > 0


def test_noise_condition_is_bounded_and_active_after_update() -> None:
    model = _small().eval()
    with torch.no_grad():
        model.noise_film.weight.fill_(0.5)
    core = torch.randn(1, 32, 129, 8)
    with torch.no_grad():
        conditioned, _ = model._prepare_core(core)
        unconditioned = model._process_core(core)
    assert not torch.equal(conditioned, unconditioned)


def test_future_audio_does_not_change_past_features() -> None:
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
