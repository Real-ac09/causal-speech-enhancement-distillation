from __future__ import annotations

import pytest
import torch

from cnvqg.models import CausalComplexMambaV6


def network(**kwargs) -> CausalComplexMambaV6:
    return CausalComplexMambaV6(
        variant="student",
        channels=64,
        temporal_dim=32,
        temporal_layers=1,
        use_mamba=False,
        **kwargs,
    ).eval()


@pytest.mark.parametrize(
    "representation", ("complex_ratio", "magnitude_only", "polar_residual")
)
def test_all_mask_representations_start_at_identity(representation: str) -> None:
    model = network(mask_representation=representation)
    waveform = torch.randn(1, 1, 1753)
    spectrum, length = model._analysis(waveform.squeeze(1), pad_end=True)
    expected = model._synthesis(spectrum, length).unsqueeze(1)
    with torch.no_grad():
        output = model(waveform)
    # Polar reconstruction can introduce a few ULPs even for a zero phase.
    torch.testing.assert_close(output.enhanced, expected, atol=5e-6, rtol=5e-6)
    torch.testing.assert_close(
        output.magnitude_mask, torch.ones_like(output.magnitude_mask), atol=1e-7, rtol=0
    )


def test_future_suffix_cannot_change_finalized_prefix() -> None:
    model = network()
    torch.manual_seed(6)
    model.decoder.mask_head.weight.data.normal_(0.0, 0.02)
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


def test_streaming_matches_whole_utterance_for_arbitrary_chunks() -> None:
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


def test_auxiliary_branch_cannot_change_enhancement() -> None:
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
    assert CausalComplexMambaV6(
        variant="student", use_mamba=False
    ).parameter_count() <= 1_100_000
    assert CausalComplexMambaV6(
        variant="teacher", use_mamba=False
    ).parameter_count() <= 2_700_000


def test_full_band_variant_preserves_shape_identity_and_cap() -> None:
    model = network(
        mask_representation="polar_residual",
        use_full_band=True,
        frequency_dim=24,
    )
    waveform = torch.randn(1, 1, 1753)
    with torch.no_grad():
        output = model(waveform)
    assert output.enhanced.shape == waveform.shape
    torch.testing.assert_close(
        output.magnitude_mask,
        torch.ones_like(output.magnitude_mask),
        atol=1e-7,
        rtol=0,
    )
    assert CausalComplexMambaV6(
        variant="teacher",
        use_mamba=False,
        use_full_band=True,
        frequency_dim=144,
    ).parameter_count() <= 2_700_000


def test_backward_gradients_are_finite() -> None:
    model = network().train()
    output = model(torch.randn(1, 1, 960))
    (output.enhanced.square().mean() + output.vq.loss).backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
