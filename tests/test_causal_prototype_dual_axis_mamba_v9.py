from __future__ import annotations

import pytest
import torch

from cnvqg.models.causal_prototype_dual_axis_mamba_v9 import (
    CausalPrototypeDualAxisMambaV9,
    FrameChannelRMSNorm,
)
from cnvqg.models.factory import build_model


def _small(**overrides) -> CausalPrototypeDualAxisMambaV9:
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
    return CausalPrototypeDualAxisMambaV9(**kwargs)


def test_factory_and_parameter_caps() -> None:
    student = build_model(
        {"architecture": "cnvqg_v9", "variant": "student", "use_mamba": False}
    )
    teacher = CausalPrototypeDualAxisMambaV9(variant="teacher", use_mamba=False)
    assert student.parameter_count() <= 1_100_000
    assert teacher.parameter_count() <= 2_700_000


def test_control_a_is_identity_initialised_in_tf_domain() -> None:
    model = _small().eval()
    noisy = torch.randn(2, 1, 1920)
    with torch.no_grad():
        output = model(noisy)
    assert torch.equal(output.complex_mask.real, torch.ones_like(output.complex_mask.real))
    assert torch.equal(output.complex_mask.imag, torch.zeros_like(output.complex_mask.imag))
    assert torch.allclose(output.magnitude_mask, torch.ones_like(output.magnitude_mask), atol=2e-6)
    assert output.vq.loss.item() == 0.0
    assert output.noise_prediction is None


def test_future_samples_cannot_change_past_frames() -> None:
    torch.manual_seed(7)
    model = _small().eval()
    noisy = torch.randn(1, 1, 3200)
    changed = noisy.clone()
    changed[..., 2080:] = torch.randn_like(changed[..., 2080:])
    with torch.no_grad():
        first = model(noisy)
        second = model(changed)
    # Frames 0..11 end no later than sample 2080 and are therefore identical.
    assert torch.allclose(
        first.mamba_features[..., :12], second.mamba_features[..., :12], atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(
        first.complex_mask[..., :12], second.complex_mask[..., :12], atol=1e-6, rtol=0.0
    )


def test_arbitrary_chunks_match_whole_utterance() -> None:
    torch.manual_seed(3)
    model = _small().eval()
    noisy = torch.randn(1, 1, 2371)
    with torch.no_grad():
        expected = model(noisy).enhanced
        state = model.init_stream_state(1, noisy.device, noisy.dtype)
        chunks = []
        offset = 0
        for size in (13, 401, 7, 963, 211, 776):
            chunk, state = model.forward_chunk(noisy[..., offset : offset + size], state)
            chunks.append(chunk)
            offset += size
        tail, state = model.flush(state)
        chunks.append(tail)
        actual = torch.cat(chunks, dim=-1)
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_state_reset_prevents_cross_utterance_leakage() -> None:
    model = _small().eval()
    first = torch.randn(1, 1, 800)
    second = torch.randn(1, 1, 997)
    state = model.init_stream_state(1, first.device, first.dtype)
    model.forward_chunk(first, state)
    state = model.init_stream_state(1, second.device, second.dtype)
    part, state = model.forward_chunk(second, state)
    tail, _ = model.flush(state)
    actual = torch.cat((part, tail), dim=-1)
    assert torch.allclose(actual, model(second).enhanced, atol=1e-4, rtol=1e-4)


def test_frame_rms_norm_does_not_mix_time_or_frequency() -> None:
    norm = FrameChannelRMSNorm(8).eval()
    x = torch.randn(2, 8, 11, 9)
    changed = x.clone()
    changed[..., 6] *= 100.0
    assert torch.equal(norm(x)[..., :6], norm(changed)[..., :6])


def test_unvalidated_optional_branches_fail_loudly() -> None:
    with pytest.raises(ValueError, match="separate validation gates"):
        _small(use_auxiliary_vq=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for BF16 Mamba")
def test_bf16_mamba_forward_backward_is_finite() -> None:
    model = _small(use_mamba=True).cuda().to(torch.bfloat16).train()
    noisy = torch.randn(1, 1, 1600, device="cuda", dtype=torch.bfloat16)
    output = model(noisy)
    loss = output.enhanced.float().square().mean()
    loss.backward()
    assert torch.isfinite(output.enhanced).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
