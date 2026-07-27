from __future__ import annotations

import pytest
import torch

from cnvqg.models import CausalMultiScaleMambaV7


def network(**kwargs) -> CausalMultiScaleMambaV7:
    return CausalMultiScaleMambaV7(
        variant="student",
        full_channels=32,
        middle_channels=48,
        bottleneck_channels=64,
        temporal_dim=32,
        frequency_dim=24,
        use_mamba=False,
        **kwargs,
    ).eval()


def test_multiscale_encoder_retains_three_frequency_scales() -> None:
    model = network()
    spectrum, _ = model._analysis(torch.randn(1, 960), pad_end=True)
    magnitude = spectrum.abs().clamp_min(1e-7)
    unit = spectrum / magnitude
    inputs = torch.stack((magnitude.pow(model.magnitude_power), unit.real, unit.imag), 1)
    latent, middle, full = model.encoder(inputs)
    assert full.shape[-2] == 257
    assert middle.shape[-2] == 128
    assert latent.shape[-2] == 64


@pytest.mark.parametrize("representation", ("complex_ratio", "polar_residual"))
def test_zero_initialised_heads_preserve_noisy_reconstruction(
    representation: str,
) -> None:
    model = network(mask_representation=representation)
    waveform = torch.randn(1, 1, 1753)
    spectrum, length = model._analysis(waveform.squeeze(1), pad_end=True)
    expected = model._synthesis(spectrum, length).unsqueeze(1)
    with torch.no_grad():
        output = model(waveform)
    torch.testing.assert_close(output.enhanced, expected, atol=5e-6, rtol=5e-6)
    torch.testing.assert_close(
        output.phase_confidence,
        torch.full_like(output.phase_confidence, 0.75),
        atol=1e-6,
        rtol=1e-6,
    )


def test_polar_controls_are_bounded() -> None:
    model = network(mask_representation="polar_residual", adaptive_phase=False)
    model.decoder.mask_head.bias.data[:] = torch.tensor([100.0, -100.0])
    with torch.no_grad():
        output = model(torch.randn(1, 1, 960))
    assert output.magnitude_mask.max() <= 4.0 + 1e-5
    phase_residual = torch.atan2(
        torch.sin(output.predicted_phase), torch.cos(output.predicted_phase)
    )
    assert torch.isfinite(phase_residual).all()


def test_future_suffix_cannot_change_finalized_prefix() -> None:
    model = network()
    torch.manual_seed(7)
    model.decoder.mask_head.weight.data.normal_(0.0, 0.02)
    model.decoder.phase_head.weight.data.normal_(0.0, 0.02)
    model.decoder.phase_gate_head.weight.data.normal_(0.0, 0.02)
    original = torch.randn(1, 1, 1920)
    changed = original.clone()
    boundary = 1280
    changed[..., boundary:] = 3.0 * torch.randn_like(changed[..., boundary:])
    with torch.no_grad():
        first = model(original).enhanced
        second = model(changed).enhanced
    finalized = boundary - model.algorithmic_latency_samples
    torch.testing.assert_close(
        first[..., :finalized], second[..., :finalized], atol=1e-6, rtol=1e-6
    )


def test_streaming_matches_whole_utterance() -> None:
    model = network()
    model.decoder.mask_head.weight.data.normal_(0.0, 0.02)
    waveform = torch.randn(1, 1, 1753)
    whole = model(waveform).enhanced
    state = model.init_stream_state(1, "cpu", torch.float32)
    pieces = []
    offset = 0
    for size in (73, 247, 1, 640, 511, 281):
        output, state = model.forward_chunk(waveform[..., offset : offset + size], state)
        pieces.append(output)
        offset += size
    tail, _ = model.flush(state)
    pieces.append(tail)
    torch.testing.assert_close(torch.cat(pieces, -1), whole, atol=1e-4, rtol=1e-4)


def test_optional_vq_is_enhancement_independent() -> None:
    model = network(auxiliary_vq=True)
    model.decoder.mask_head.weight.data.normal_(0.0, 0.02)
    waveform = torch.randn(1, 1, 960)
    with torch.no_grad():
        original = model(waveform).enhanced
        for parameter in model.noise_encoder.parameters():
            parameter.normal_(20.0, 5.0)
        model.noise_vq.codebook.normal_(100.0, 20.0)
        changed = model(waveform).enhanced
    assert torch.equal(original, changed)


def test_parameter_caps() -> None:
    assert CausalMultiScaleMambaV7(
        variant="student", use_mamba=False
    ).parameter_count() <= 1_100_000
    assert CausalMultiScaleMambaV7(
        variant="teacher", use_mamba=False
    ).parameter_count() <= 2_700_000


def test_backward_gradients_are_finite() -> None:
    model = network().train()
    output = model(torch.randn(1, 1, 960))
    output.enhanced.square().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
