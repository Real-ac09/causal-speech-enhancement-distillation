from __future__ import annotations

import pytest
import torch

from cnvqg.models import CausalAuxVQMambaV52


def network() -> CausalAuxVQMambaV52:
    model = CausalAuxVQMambaV52(variant="student", use_mamba=False).eval()
    torch.manual_seed(52)
    model.decoder.magnitude_head.weight.data.normal_(0.0, 0.02)
    model.decoder.phase_head.weight.data.normal_(0.0, 0.02)
    return model


def test_encoder_reduces_frequency_once() -> None:
    model = network()
    spectrum, _ = model._analysis(torch.randn(1, 960), pad_end=True)
    inputs = torch.stack(
        (spectrum.abs().pow(model.magnitude_power), spectrum.angle().cos(), spectrum.angle().sin()),
        dim=1,
    )
    latent, full = model.encoder(inputs)
    assert full.shape[-2] == model.n_fft // 2 + 1
    assert latent.shape[-2] == (full.shape[-2] // 2)


def test_blocks_are_untied() -> None:
    model = network()
    assert model.blocks[0] is not model.blocks[1]
    assert model.blocks[0].time_mamba is not model.blocks[1].time_mamba


def test_future_suffix_cannot_change_finalized_prefix() -> None:
    model = network()
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


def test_parameter_caps() -> None:
    assert CausalAuxVQMambaV52(variant="student", use_mamba=False).parameter_count() <= 1_100_000
    assert CausalAuxVQMambaV52(variant="teacher", use_mamba=False).parameter_count() <= 2_700_000


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 requires CUDA")
def test_bf16_gradients_are_finite() -> None:
    model = CausalAuxVQMambaV52(variant="teacher", use_mamba=True).cuda().train()
    waveform = torch.randn(1, 1, 960, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(waveform)
        loss = output.enhanced.square().mean() + output.vq.loss
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
