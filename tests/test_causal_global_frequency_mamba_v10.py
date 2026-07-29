from __future__ import annotations

import pytest
import torch

from cnvqg.models.causal_global_frequency_mamba_v10 import (
    CausalGlobalFrequencyMambaV10,
    LowRankGlobalFrequencyAttention,
)
from cnvqg.models.factory import build_model


def _small(**overrides) -> CausalGlobalFrequencyMambaV10:
    kwargs = dict(
        variant="student",
        channels=32,
        noise_dim=16,
        temporal_layers=1,
        use_mamba=False,
        auxiliary_vq=False,
        reconstruction_mode="direct_scalar_mask",
        phase_residual_scale=0.0,
        frequency_attention_dim=16,
        frequency_attention_heads=4,
        enforce_parameter_cap=False,
    )
    kwargs.update(overrides)
    return CausalGlobalFrequencyMambaV10(**kwargs)


def test_factory_and_default_student_parameter_cap() -> None:
    model = build_model(
        {
            "architecture": "cnvqg_v10",
            "variant": "student",
            "use_mamba": True,
            "auxiliary_vq": False,
            "reconstruction_mode": "direct_scalar_mask",
            "phase_residual_scale": 0.0,
        }
    )
    assert isinstance(model, CausalGlobalFrequencyMambaV10)
    assert model.parameter_count() <= 1_100_000


def test_frequency_attention_mixes_distant_bins_without_mixing_frames() -> None:
    torch.manual_seed(10)
    module = LowRankGlobalFrequencyAttention(16, 8, 2).eval()
    original = torch.randn(2, 16, 17, 5)
    changed = original.clone()
    changed[:, 0, 0, 3] += 4.0
    with torch.no_grad():
        first = module(original)
        second = module(changed)
    assert torch.equal(first[..., :3], second[..., :3])
    assert torch.equal(first[..., 4:], second[..., 4:])
    assert not torch.equal(first[:, :, -1, 3], second[:, :, -1, 3])


def test_future_audio_cannot_change_finalized_output() -> None:
    torch.manual_seed(11)
    model = _small().eval()
    noisy = torch.randn(1, 1, 3200)
    changed = noisy.clone()
    changed[..., 2080:] = torch.randn_like(changed[..., 2080:])
    with torch.no_grad():
        first = model(noisy)
        second = model(changed)
    assert torch.allclose(
        first.mamba_features[..., :12], second.mamba_features[..., :12], atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(
        first.magnitude_mask[..., :12], second.magnitude_mask[..., :12], atol=1e-6, rtol=0.0
    )


def test_arbitrary_chunks_match_whole_utterance() -> None:
    torch.manual_seed(12)
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
        tail, _ = model.flush(state)
        chunks.append(tail)
    actual = torch.cat(chunks, dim=-1)
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_invalid_attention_shape_fails_loudly() -> None:
    with pytest.raises(ValueError, match="divisible"):
        LowRankGlobalFrequencyAttention(16, attention_dim=10, heads=4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for BF16 Mamba")
def test_bf16_mamba_forward_backward_is_finite() -> None:
    model = _small(use_mamba=True).cuda().train()
    noisy = torch.randn(1, 1, 1600, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(noisy)
        loss = output.enhanced.float().square().mean()
    loss.backward()
    assert torch.isfinite(output.enhanced).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
