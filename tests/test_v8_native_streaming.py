from __future__ import annotations

import pytest
import torch

from cnvqg.models import PredictiveNoiseVQMambaV8, V8NativeStreamer


def _model(*, use_mamba: bool = False) -> PredictiveNoiseVQMambaV8:
    model = PredictiveNoiseVQMambaV8(
        variant="student",
        channels=64,
        noise_dim=32,
        temporal_layers=1,
        auxiliary_vq=False,
        use_mamba=use_mamba,
        phase_residual_scale=0.0,
        reconstruction_mode="direct_scalar_mask",
    )
    model.decoder.mask_head.weight.data.normal_(0.0, 0.02)
    model.decoder.mask_head.bias.data.normal_(0.0, 0.02)
    return model.eval()


@pytest.mark.parametrize(
    "length,chunks",
    [
        (73, (1, 17, 55)),
        (320, (160, 160)),
        (1753, (73, 247, 1, 640, 511, 281)),
        (3200, (160,) * 20),
    ],
)
def test_native_stream_matches_offline_for_arbitrary_chunks(
    length: int, chunks: tuple[int, ...]
) -> None:
    torch.manual_seed(81)
    model = _model()
    waveform = torch.randn(1, 1, length)
    expected = model(waveform).enhanced

    streamer = V8NativeStreamer(model)
    state = streamer.init_state(1, waveform.device, waveform.dtype)
    pieces = []
    offset = 0
    for size in chunks:
        piece, state = streamer.process_chunk(waveform[..., offset : offset + size], state)
        pieces.append(piece)
        offset += size
    assert offset == length
    tail, state = streamer.flush(state)
    pieces.append(tail)

    actual = torch.cat(pieces, dim=-1)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert state.emitted_samples == length


def test_native_state_size_is_bounded_over_long_stream() -> None:
    torch.manual_seed(82)
    model = _model()
    streamer = V8NativeStreamer(model)
    state = streamer.init_state(1, "cpu", torch.float32)
    sizes = []
    with torch.inference_mode():
        for _ in range(100):
            _, state = streamer.process_chunk(torch.randn(1, 1, model.hop_length), state)
            sizes.append(state.tensor_elements())
            assert state.input_buffer.shape[-1] < model.win_length
    assert len(set(sizes[2:])) == 1


def test_flush_is_idempotent_and_processing_after_flush_is_rejected() -> None:
    model = _model()
    streamer = V8NativeStreamer(model)
    state = streamer.init_state(1, "cpu", torch.float32)
    _, state = streamer.process_chunk(torch.randn(1, 1, 401), state)
    tail, state = streamer.flush(state)
    assert tail.shape[-1] > 0
    second, state = streamer.flush(state)
    assert second.shape[-1] == 0
    with pytest.raises(RuntimeError, match="after flush"):
        streamer.process_chunk(torch.randn(1, 1, 160), state)


def test_mamba_native_stream_matches_offline() -> None:
    pytest.importorskip("mamba_ssm")
    torch.manual_seed(83)
    model = _model(use_mamba=True)
    waveform = torch.randn(1, 1, 960)
    expected = model(waveform).enhanced
    streamer = V8NativeStreamer(model)
    state = streamer.init_state(1, "cpu", torch.float32)
    pieces = []
    for chunk in waveform.split(model.hop_length, dim=-1):
        piece, state = streamer.process_chunk(chunk, state)
        pieces.append(piece)
    tail, _ = streamer.flush(state)
    actual = torch.cat((*pieces, tail), dim=-1)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
