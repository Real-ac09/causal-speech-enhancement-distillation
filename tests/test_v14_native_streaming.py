from __future__ import annotations

from pathlib import Path

import torch

from cnvqg.models import (
    CausalConfidenceGateV14,
    PredictiveNoiseVQMambaV8,
    V14NativeStreamer,
)


def _model(tmp_path: Path) -> CausalConfidenceGateV14:
    config = {
        "architecture": "causal_temporal_core_v12",
        "variant": "student",
        "sample_rate": 16_000,
        "n_fft": 64,
        "win_length": 32,
        "hop_length": 16,
        "magnitude_power": 0.3,
        "channels": 8,
        "noise_dim": 4,
        "temporal_layers": 1,
        "auxiliary_vq": False,
        "use_mamba": False,
        "temporal_core": "gru",
        "temporal_hidden_dim": 8,
        "time_kernel_size": 1,
        "phase_residual_scale": 0.0,
        "reconstruction_mode": "direct_scalar_mask",
        "enforce_parameter_cap": False,
    }
    architecture = config.pop("architecture")
    backbone = PredictiveNoiseVQMambaV8(**config)
    config["architecture"] = architecture
    checkpoint = tmp_path / "backbone.pt"
    torch.save(
        {
            "config": {"model": config},
            "model_state_dict": backbone.state_dict(),
        },
        checkpoint,
    )
    model = CausalConfidenceGateV14(
        backbone_checkpoint=str(checkpoint),
        gate_hidden_dim=8,
    )
    for parameter in model.confidence_gate.parameters():
        parameter.data.normal_(mean=0.0, std=0.05)
    return model.eval()


def test_native_gate_stream_matches_offline_for_arbitrary_chunks(
    tmp_path: Path,
) -> None:
    torch.manual_seed(15_040)
    model = _model(tmp_path)
    waveform = torch.randn(1, 1, 1_753)
    expected = model(waveform).enhanced
    streamer = V14NativeStreamer(model)
    state = streamer.init_state(1, waveform.device, waveform.dtype)
    pieces = []
    offset = 0
    for size in (73, 247, 1, 640, 511, 281):
        piece, state = streamer.process_chunk(
            waveform[..., offset : offset + size],
            state,
        )
        pieces.append(piece)
        offset += size
    tail, _ = streamer.flush(state)
    actual = torch.cat((*pieces, tail), dim=-1)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_native_gate_state_is_bounded(tmp_path: Path) -> None:
    model = _model(tmp_path)
    streamer = V14NativeStreamer(model)
    state = streamer.init_state(1, "cpu", torch.float32)
    sizes = []
    with torch.inference_mode():
        for _ in range(100):
            _, state = streamer.process_chunk(
                torch.randn(1, 1, model.hop_length),
                state,
            )
            sizes.append(state.tensor_elements())
            assert state.input_buffer.shape[-1] < model.win_length
    assert len(set(sizes[2:])) == 1
    assert V14NativeStreamer.GATE_STATE_KEY in state.gru
