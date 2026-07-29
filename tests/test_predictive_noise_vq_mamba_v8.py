from __future__ import annotations

import copy

import pytest
import torch

from cnvqg.models import PredictiveNoiseVQMambaV8
from cnvqg.models.factory import build_model


def network(**kwargs) -> PredictiveNoiseVQMambaV8:
    return PredictiveNoiseVQMambaV8(
        variant="student",
        channels=64,
        noise_dim=32,
        temporal_layers=1,
        use_mamba=False,
        **kwargs,
    ).eval()


def test_factory_and_parameter_caps() -> None:
    assert isinstance(
        build_model({"architecture": "predictive_noise_vq_mamba_v8", "use_mamba": False}),
        PredictiveNoiseVQMambaV8,
    )
    student = PredictiveNoiseVQMambaV8(variant="student", use_mamba=False)
    teacher = PredictiveNoiseVQMambaV8(variant="teacher", use_mamba=False)
    assert student.parameter_count() <= 1_100_000
    assert teacher.parameter_count() <= 2_700_000
    assert student.algorithmic_latency_samples == 320


def test_identity_initialisation_and_exact_mixture_consistency() -> None:
    model = network()
    waveform = torch.randn(1, 1, 1753)
    spectrum, length = model._analysis(waveform.squeeze(1), pad_end=True)
    expected = model._synthesis(spectrum, length).unsqueeze(1)
    with torch.no_grad():
        output = model(waveform)
    torch.testing.assert_close(output.enhanced, expected, atol=5e-6, rtol=5e-6)
    torch.testing.assert_close(
        output.speech_mask, torch.ones_like(output.speech_mask), atol=5e-7, rtol=0
    )
    torch.testing.assert_close(
        output.noise_mask, torch.zeros_like(output.noise_mask), atol=5e-7, rtol=0
    )
    assert output.mixture_residual.abs().max() < 1e-6


def test_hybrid_magnitude_residual_is_identity_initialised_and_additive() -> None:
    model = network(
        auxiliary_vq=False,
        reconstruction_mode="hybrid_magnitude_residual",
    )
    waveform = torch.randn(1, 1, 1753)
    spectrum, length = model._analysis(waveform.squeeze(1), pad_end=True)
    expected = model._synthesis(spectrum, length).unsqueeze(1)
    with torch.no_grad():
        identity = model(waveform)
    torch.testing.assert_close(identity.enhanced, expected, atol=5e-6, rtol=5e-6)
    assert identity.magnitude_residual.abs().max() == 0

    # Activate only the additive branch: it must add magnitude independently
    # of the multiplicative ratio and preserve exact mixture accounting.
    model.decoder.mask_head.bias.data[1] = 1.0
    with torch.no_grad():
        additive = model(waveform)
    assert additive.magnitude_residual.mean() > 0
    assert torch.all(additive.estimated_magnitude >= identity.estimated_magnitude)
    assert additive.mixture_residual.abs().max() < 1e-6


def test_v83_factory_alias_and_parameter_caps() -> None:
    student = build_model(
        {
            "architecture": "predictive_noise_vq_mamba_v83",
            "variant": "student",
            "use_mamba": False,
            "reconstruction_mode": "hybrid_magnitude_residual",
        }
    )
    teacher = PredictiveNoiseVQMambaV8(
        variant="teacher",
        use_mamba=False,
        reconstruction_mode="hybrid_magnitude_residual",
    )
    assert student.parameter_count() <= 1_100_000
    assert teacher.parameter_count() <= 2_700_000


def test_scale_preserving_detail_is_identity_initialised_and_trainable() -> None:
    model = network(auxiliary_vq=False, scale_preserving_detail=True).train()
    waveform = torch.randn(1, 1, 1280)
    output = model(waveform)
    spectrum, length = model._analysis(waveform.squeeze(1), pad_end=True)
    identity = model._synthesis(spectrum, length).unsqueeze(1)
    torch.testing.assert_close(output.enhanced, identity, atol=5e-6, rtol=5e-6)
    output.enhanced.square().mean().backward()
    assert model.decoder.scale_head is not None
    gradient = model.decoder.scale_head.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_direct_scalar_mask_is_identity_initialised_and_has_finite_gradient() -> None:
    model = network(
        auxiliary_vq=False,
        reconstruction_mode="direct_scalar_mask",
    ).train()
    waveform = torch.randn(1, 1, 1280)
    output = model(waveform)
    spectrum, length = model._analysis(waveform.squeeze(1), pad_end=True)
    identity = model._synthesis(spectrum, length).unsqueeze(1)
    torch.testing.assert_close(output.enhanced, identity, atol=5e-6, rtol=5e-6)
    torch.testing.assert_close(
        output.magnitude_mask,
        torch.ones_like(output.magnitude_mask),
        atol=1e-7,
        rtol=0,
    )
    output.magnitude_mask.mean().backward()
    gradient = model.decoder.mask_head.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_codebook_is_not_in_enhancement_path() -> None:
    torch.manual_seed(8)
    model = network()
    clone = copy.deepcopy(model)
    clone.noise_vq.codebook.normal_(100.0, 20.0)
    waveform = torch.randn(1, 1, 960)
    with torch.no_grad():
        first = model(waveform).enhanced
        second = clone(waveform).enhanced
    assert torch.equal(first, second)


def test_prototype_prediction_is_multi_horizon_and_trainable() -> None:
    model = network().train()
    output = model(torch.randn(2, 1, 1600))
    assert output.prototype_logits.shape[2:] == (3, 32)
    assert torch.isfinite(output.prototype_prediction_loss)
    output.vq.loss.backward()
    assert model.prototype_predictor.weight.grad is not None
    assert torch.isfinite(model.prototype_predictor.weight.grad).all()


def test_future_suffix_cannot_change_finalized_prefix() -> None:
    model = network(auxiliary_vq=False)
    model.decoder.mask_head.weight.data.normal_(0.0, 0.02)
    original = torch.randn(1, 1, 1920)
    changed = original.clone()
    boundary = 1280
    changed[..., boundary:] = torch.randn_like(changed[..., boundary:]) * 3.0
    with torch.no_grad():
        first = model(original).enhanced
        second = model(changed).enhanced
    finalized = boundary - model.algorithmic_latency_samples
    torch.testing.assert_close(
        first[..., :finalized], second[..., :finalized], atol=1e-6, rtol=1e-6
    )


def test_streaming_matches_whole_file_for_arbitrary_chunks() -> None:
    model = network(auxiliary_vq=False)
    model.decoder.mask_head.weight.data.normal_(0.0, 0.02)
    waveform = torch.randn(1, 1, 1753)
    whole = model(waveform).enhanced
    state = model.init_stream_state(1, "cpu", torch.float32)
    pieces = []
    offset = 0
    for size in (73, 247, 1, 640, 511, 281):
        piece, state = model.forward_chunk(waveform[..., offset : offset + size], state)
        pieces.append(piece)
        offset += size
    tail, _ = model.flush(state)
    pieces.append(tail)
    torch.testing.assert_close(torch.cat(pieces, -1), whole, atol=1e-4, rtol=1e-4)


def test_backward_gradients_are_finite() -> None:
    model = network().train()
    output = model(torch.randn(1, 1, 1280))
    (output.enhanced.square().mean() + output.vq.loss).backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 test requires CUDA")
def test_bf16_forward_backward_is_finite() -> None:
    model = PredictiveNoiseVQMambaV8(
        variant="student", channels=64, noise_dim=32, use_mamba=True
    ).cuda().train()
    waveform = torch.randn(1, 1, 1280, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(waveform)
        loss = output.enhanced.square().mean() + output.vq.loss
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
