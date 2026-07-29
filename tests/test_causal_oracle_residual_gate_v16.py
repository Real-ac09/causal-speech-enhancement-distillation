from __future__ import annotations

from pathlib import Path

import torch

from cnvqg.models import (
    CausalOracleResidualGateV16,
    PredictiveNoiseVQMambaV8,
    V16NativeStreamer,
)


def _model(tmp_path: Path) -> CausalOracleResidualGateV16:
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
    return CausalOracleResidualGateV16(
        backbone_checkpoint=str(checkpoint),
        gate_hidden_dim=8,
        gate_initial_strength=0.75,
    )


def test_waveform_blend_has_exact_endpoints() -> None:
    noisy = torch.randn(2, 1, 257)
    enhanced = torch.randn_like(noisy)
    zeros = torch.zeros_like(noisy)
    ones = torch.ones_like(noisy)
    assert torch.equal(
        CausalOracleResidualGateV16.blend_waveforms(
            noisy,
            enhanced,
            zeros,
        ),
        noisy,
    )
    torch.testing.assert_close(
        CausalOracleResidualGateV16.blend_waveforms(
            noisy,
            enhanced,
            ones,
        ),
        enhanced,
        atol=2e-7,
        rtol=2e-7,
    )


def test_wrapper_freezes_backbone_and_routes_gate_gradients(
    tmp_path: Path,
) -> None:
    model = _model(tmp_path).train()
    noisy = torch.randn(2, 1, 320)
    output = model(noisy)
    output.enhanced.abs().mean().backward()

    assert model.backbone.training is False
    assert all(
        parameter.grad is None
        for parameter in model.backbone.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.confidence_gate.parameters()
    )
    assert model.gate_parameter_count() < 10_000
    assert output.enhanced.shape == noisy.shape
    assert output.sample_strength.shape == noisy.shape
    assert output.gate_strength.shape[-1] == 1


def test_wrapper_waveform_prefix_is_future_independent(
    tmp_path: Path,
) -> None:
    model = _model(tmp_path).eval()
    first = torch.randn(1, 1, 640)
    second = first.clone()
    second[..., 320:] = torch.randn_like(second[..., 320:])

    with torch.inference_mode():
        first_output = model(first)
        second_output = model(second)

    assert torch.equal(
        first_output.gate_strength[:, :19],
        second_output.gate_strength[:, :19],
    )
    assert torch.equal(
        first_output.enhanced[..., :304],
        second_output.enhanced[..., :304],
    )


def test_native_stream_matches_offline_for_arbitrary_chunks(
    tmp_path: Path,
) -> None:
    torch.manual_seed(16_001)
    model = _model(tmp_path).eval()
    for parameter in model.confidence_gate.parameters():
        parameter.data.normal_(mean=0.0, std=0.05)
    waveform = torch.randn(1, 1, 1_753)
    expected = model(waveform).enhanced
    streamer = V16NativeStreamer(model)
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
    tail, state = streamer.flush(state)
    actual = torch.cat((*pieces, tail), dim=-1)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert V16NativeStreamer.LAST_STRENGTH_KEY in state.gru


def test_native_stream_state_is_bounded(tmp_path: Path) -> None:
    model = _model(tmp_path).eval()
    streamer = V16NativeStreamer(model)
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
