from __future__ import annotations

import pytest
import torch

from cnvqg.models.causal_scale_preserving_mamba_v91 import (
    CausalScalePreservingMambaV91,
    ScalePreservingEncoderV91,
)
from cnvqg.models.factory import build_model


def _small(**overrides) -> CausalScalePreservingMambaV91:
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
    return CausalScalePreservingMambaV91(**kwargs)


def test_factory_presets_fit_parameter_caps() -> None:
    student = build_model(
        {"architecture": "cnvqg_v91", "variant": "student", "use_mamba": False}
    )
    teacher = CausalScalePreservingMambaV91(variant="teacher", use_mamba=False)
    assert student.parameter_count() <= 1_100_000
    assert teacher.parameter_count() <= 2_700_000


def test_explicit_polar_decoder_starts_at_identity_with_phase_disabled() -> None:
    model = _small().eval()
    noisy = torch.randn(2, 1, 1920)
    with torch.no_grad():
        output = model(noisy)
        spectrum, _ = model._analysis(noisy.squeeze(1), pad_end=True)
    assert torch.equal(output.magnitude_mask, torch.ones_like(output.magnitude_mask))
    assert torch.equal(output.predicted_phase, torch.angle(spectrum))
    assert torch.equal(output.complex_mask.real, torch.ones_like(output.complex_mask.real))
    assert torch.equal(output.complex_mask.imag, torch.zeros_like(output.complex_mask.imag))


def test_full_resolution_path_preserves_input_scale() -> None:
    torch.manual_seed(11)
    encoder = ScalePreservingEncoderV91(8, 16, 24).eval()
    x = torch.randn(1, 3, 31, 7)
    with torch.no_grad():
        detail = encoder(x)[2]
        scaled_detail = encoder(4.0 * x)[2]
    ratio = scaled_detail.square().mean().sqrt() / detail.square().mean().sqrt()
    # Per-bin RMS normalisation would force this ratio close to one.
    assert ratio > 2.0


def test_raw_magnitude_is_a_direct_decoder_input() -> None:
    model = _small()
    assert model.decoder.detail_fuse[0].in_channels == model.detail_channels * 2 + 1
    assert model.decoder.magnitude_head.in_channels == model.detail_channels + 1


def test_magnitude_head_receives_gradient_while_disabled_phase_does_not() -> None:
    model = _small(phase_residual_limit=0.0).train()
    noisy = torch.randn(1, 1, 1600)
    output = model(noisy)
    output.enhanced.square().mean().backward()
    assert model.decoder.magnitude_head.weight.grad is not None
    assert model.decoder.magnitude_head.weight.grad.abs().sum() > 0
    phase_grad = model.decoder.phase_head.weight.grad
    assert phase_grad is None or torch.equal(phase_grad, torch.zeros_like(phase_grad))


def test_streaming_matches_whole_utterance() -> None:
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
        actual = torch.cat((*pieces, tail), dim=-1)
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


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
