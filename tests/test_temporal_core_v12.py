from __future__ import annotations

import pytest
import torch

from cnvqg.models import PredictiveNoiseVQMambaV8, V8NativeStreamer
from cnvqg.models.factory import build_model
from cnvqg.models.streaming_hybrid_v2 import CausalConv2d


def _model(**overrides: object) -> PredictiveNoiseVQMambaV8:
    config: dict[str, object] = {
        "variant": "student",
        "channels": 64,
        "noise_dim": 32,
        "temporal_layers": 1,
        "auxiliary_vq": False,
        "use_mamba": False,
        "temporal_core": "gru",
        "phase_residual_scale": 0.0,
        "reconstruction_mode": "direct_scalar_mask",
    }
    config.update(overrides)
    model = PredictiveNoiseVQMambaV8(**config)
    model.decoder.mask_head.weight.data.normal_(0.0, 0.02)
    return model.eval()


def test_factory_alias_builds_temporal_core_experiment() -> None:
    model = build_model(
        {
            "architecture": "causal_temporal_core_v12",
            "variant": "student",
            "use_mamba": False,
            "temporal_core": "gru",
        }
    )
    assert isinstance(model, PredictiveNoiseVQMambaV8)
    assert model.temporal_core == "gru"


def test_matched_gru_is_within_two_percent_of_mamba_parameter_count() -> None:
    control = PredictiveNoiseVQMambaV8(
        variant="student", auxiliary_vq=False, reconstruction_mode="direct_scalar_mask"
    )
    candidate = PredictiveNoiseVQMambaV8(
        variant="student",
        auxiliary_vq=False,
        reconstruction_mode="direct_scalar_mask",
        temporal_core="gru",
    )
    relative_difference = abs(
        candidate.parameter_count() - control.parameter_count()
    ) / control.parameter_count()
    assert relative_difference < 0.02


def test_time_kernel_one_removes_all_tf_convolution_history() -> None:
    model = _model(time_kernel_size=1)
    causal_layers = [module for module in model.modules() if isinstance(module, CausalConv2d)]
    assert causal_layers
    assert all(layer.time_padding == 0 for layer in causal_layers)


@pytest.mark.parametrize(
    "hidden_dim,time_kernel_size",
    [(64, 3), (32, 1)],
)
def test_gru_native_stream_matches_whole_utterance(
    hidden_dim: int, time_kernel_size: int
) -> None:
    torch.manual_seed(1201)
    model = _model(
        temporal_hidden_dim=hidden_dim,
        time_kernel_size=time_kernel_size,
    )
    waveform = torch.randn(1, 1, 1753)
    expected = model(waveform).enhanced
    streamer = V8NativeStreamer(model)
    state = streamer.init_state(1, "cpu", torch.float32)
    pieces = []
    for chunk in waveform.split(137, dim=-1):
        piece, state = streamer.process_chunk(chunk, state)
        pieces.append(piece)
    tail, _ = streamer.flush(state)
    actual = torch.cat((*pieces, tail), dim=-1)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("time_kernel_size", [1, 3])
def test_gru_variants_are_causal(time_kernel_size: int) -> None:
    torch.manual_seed(1202)
    model = _model(time_kernel_size=time_kernel_size)
    original = torch.randn(1, 1, 1920)
    changed = original.clone()
    boundary = 1280
    changed[..., boundary:] = torch.randn_like(changed[..., boundary:])
    first = model(original).enhanced
    second = model(changed).enhanced
    finalized = boundary - model.algorithmic_latency_samples
    torch.testing.assert_close(
        first[..., :finalized], second[..., :finalized], atol=1e-6, rtol=1e-6
    )


def test_projected_gru_reduces_persistent_streaming_state() -> None:
    matched = _model(temporal_hidden_dim=64, time_kernel_size=3)
    projected = _model(temporal_hidden_dim=32, time_kernel_size=1)
    chunk = torch.randn(1, 1, 640)

    matched_streamer = V8NativeStreamer(matched)
    matched_state = matched_streamer.init_state(1, "cpu", torch.float32)
    _, matched_state = matched_streamer.process_chunk(chunk, matched_state)

    projected_streamer = V8NativeStreamer(projected)
    projected_state = projected_streamer.init_state(1, "cpu", torch.float32)
    _, projected_state = projected_streamer.process_chunk(chunk, projected_state)

    assert projected_state.tensor_elements() < matched_state.tensor_elements()


def test_projected_gru_training_gradients_are_finite() -> None:
    model = _model(temporal_hidden_dim=32, time_kernel_size=1).train()
    output = model(torch.randn(1, 1, 960))
    loss = output.enhanced.square().mean()
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.bottleneck.temporal.parameters()
        if parameter.requires_grad
    ]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
