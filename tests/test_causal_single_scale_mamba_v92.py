from __future__ import annotations

import pytest
import torch

from cnvqg.models.causal_single_scale_mamba_v92 import CausalSingleScaleMambaV92
from cnvqg.models.factory import build_model


def _small(**overrides) -> CausalSingleScaleMambaV92:
    kwargs = dict(
        variant="student",
        detail_channels=16,
        half_channels=24,
        core_channels=32,
        blocks=1,
        use_mamba=False,
        enforce_parameter_cap=False,
    )
    kwargs.update(overrides)
    return CausalSingleScaleMambaV92(**kwargs)


def test_factory_caps_and_single_reduction_shape() -> None:
    student = build_model(
        {"architecture": "cnvqg_v92", "variant": "student", "use_mamba": False}
    )
    teacher = CausalSingleScaleMambaV92(variant="teacher", use_mamba=False)
    assert student.parameter_count() <= 1_100_000
    assert teacher.parameter_count() <= 2_700_000
    output = _small()(torch.randn(1, 1, 1600))
    assert output.encoder_features.shape[-2] == 129


def test_identity_initialisation_and_phase_isolation() -> None:
    model = _small().eval()
    noisy = torch.randn(1, 1, 1920)
    with torch.no_grad():
        output = model(noisy)
        spectrum, _ = model._analysis(noisy.squeeze(1), True)
    assert torch.equal(output.magnitude_mask, torch.ones_like(output.magnitude_mask))
    assert torch.equal(output.predicted_phase, torch.angle(spectrum))


def test_future_audio_does_not_change_past_features() -> None:
    model = _small().eval()
    noisy = torch.randn(1, 1, 3200)
    changed = noisy.clone()
    changed[..., 2080:] = torch.randn_like(changed[..., 2080:])
    with torch.no_grad():
        first = model(noisy).mamba_features[..., :12]
        second = model(changed).mamba_features[..., :12]
    assert torch.allclose(first, second, atol=1e-5, rtol=1e-5)


def test_arbitrary_chunk_streaming_matches_whole_file() -> None:
    model = _small().eval()
    noisy = torch.randn(1, 1, 2317)
    with torch.no_grad():
        expected = model(noisy).enhanced
        state = model.init_stream_state(1, noisy.device, noisy.dtype)
        pieces = []
        offset = 0
        for size in (17, 500, 3, 901, 896):
            piece, state = model.forward_chunk(noisy[..., offset : offset + size], state)
            pieces.append(piece)
            offset += size
        tail, _ = model.flush(state)
    assert torch.allclose(torch.cat((*pieces, tail), -1), expected, atol=1e-4, rtol=1e-4)


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
