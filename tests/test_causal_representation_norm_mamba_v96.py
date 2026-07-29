from __future__ import annotations

import pytest
import torch

from cnvqg.models.causal_aux_vq_mamba_v51 import FrameGroupNorm
from cnvqg.models.causal_prototype_dual_axis_mamba_v9 import FrameChannelRMSNorm
from cnvqg.models.causal_representation_norm_mamba_v96 import (
    CausalRepresentationNormMambaV96,
)
from cnvqg.models.factory import build_model


def _small(**overrides) -> CausalRepresentationNormMambaV96:
    kwargs = dict(
        variant="student", detail_channels=16, half_channels=24,
        core_channels=32, blocks=1, use_mamba=False,
        enforce_parameter_cap=False,
    )
    kwargs.update(overrides)
    return CausalRepresentationNormMambaV96(**kwargs)


@pytest.mark.parametrize(
    ("features", "group_norm"), ((True, False), (False, True), (True, True))
)
def test_factory_caps_for_all_switches(features: bool, group_norm: bool) -> None:
    model = build_model(
        {
            "architecture": "cnvqg_v96", "variant": "student", "use_mamba": False,
            "v8_input_features": features, "frame_group_norm": group_norm,
        }
    )
    assert model.parameter_count() <= 1_100_000


def test_v8_input_representation_is_exact() -> None:
    model = _small(v8_input_features=True)
    magnitude = torch.rand(1, 257, 4).add_(0.1)
    phase = torch.rand_like(magnitude)
    unit = torch.polar(torch.ones_like(phase), phase)
    compressed = magnitude.pow(model.magnitude_power)
    features = model._input_features(magnitude, compressed, unit)
    assert torch.equal(features[:, 0], compressed)
    assert torch.allclose(features[:, 1], torch.cos(phase))
    assert torch.allclose(features[:, 2], torch.sin(phase))


def test_group_norm_replaces_every_rms_norm() -> None:
    model = _small(frame_group_norm=True)
    assert any(isinstance(module, FrameGroupNorm) for module in model.modules())
    assert not any(isinstance(module, FrameChannelRMSNorm) for module in model.modules())


def test_future_audio_does_not_change_past_features() -> None:
    model = _small(v8_input_features=True, frame_group_norm=True).eval()
    noisy = torch.randn(1, 1, 3200)
    changed = noisy.clone()
    changed[..., 2080:] = torch.randn_like(changed[..., 2080:])
    with torch.no_grad():
        first = model(noisy).mamba_features[..., :12]
        second = model(changed).mamba_features[..., :12]
    assert torch.allclose(first, second, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for BF16 Mamba")
def test_bf16_forward_backward_is_finite() -> None:
    model = _small(
        v8_input_features=True, frame_group_norm=True, use_mamba=True
    ).cuda().to(torch.bfloat16).train()
    noisy = torch.randn(1, 1, 1600, device="cuda", dtype=torch.bfloat16)
    output = model(noisy)
    output.enhanced.float().square().mean().backward()
    assert torch.isfinite(output.enhanced).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
