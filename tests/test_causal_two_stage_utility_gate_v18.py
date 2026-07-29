from __future__ import annotations

from pathlib import Path

import torch

from cnvqg.models import (
    CausalStatisticsUtilitySafetySelector,
    CausalTwoStageUtilityGateV18,
    CausalTwoStageUtilitySafetySelector,
    PredictiveNoiseVQMambaV8,
    V18TwoStageNativeStreamer,
)


def test_zero_residual_heads_preserve_recipe7b_policy() -> None:
    torch.manual_seed(18_000)
    recipe7 = CausalStatisticsUtilitySafetySelector(
        noise_dim=4,
        hidden_dim=8,
    )
    recipe8 = CausalTwoStageUtilitySafetySelector(
        noise_dim=4,
        hidden_dim=8,
    )
    missing, unexpected = recipe8.load_state_dict(
        recipe7.state_dict(),
        strict=False,
    )
    assert set(missing) == {
        "full_route_head.weight",
        "full_route_head.bias",
        "reduced_policy_head.weight",
        "reduced_policy_head.bias",
    }
    assert unexpected == []
    hidden = torch.randn(2, 7, 8)
    old = recipe7.details_from_hidden(hidden)
    new = recipe8.details_from_hidden(hidden)
    torch.testing.assert_close(new[2], old[2], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(new[0], old[0], atol=1e-6, rtol=1e-6)


def test_two_stage_policy_is_normalised_and_differentiable() -> None:
    selector = CausalTwoStageUtilitySafetySelector(
        noise_dim=4,
        hidden_dim=8,
    )
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    strength, _, probability, *_ = selector.details_from_hidden(hidden)
    torch.testing.assert_close(
        probability.sum(dim=-1),
        torch.ones(2, 5),
    )
    assert bool(((strength >= 0.0) & (strength <= 1.0)).all())
    strength.mean().backward()
    assert selector.full_route_head.weight.grad is not None
    assert selector.reduced_policy_head.weight.grad is not None


def _model(tmp_path: Path) -> CausalTwoStageUtilityGateV18:
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
    model = CausalTwoStageUtilityGateV18(
        backbone_checkpoint=str(checkpoint),
        gate_hidden_dim=8,
        gate_parameter_cap=10_000,
    )
    for parameter in model.confidence_gate.parameters():
        if parameter.requires_grad:
            parameter.data.normal_(mean=0.0, std=0.02)
    return model.eval()


def test_recipe8_is_causal(tmp_path: Path) -> None:
    model = _model(tmp_path)
    prefix = torch.randn(1, 1, 800)
    first = model(torch.cat((prefix, torch.zeros_like(prefix)), dim=-1))
    second = model(torch.cat((prefix, torch.randn_like(prefix)), dim=-1))
    torch.testing.assert_close(
        first.enhanced[..., : prefix.shape[-1]],
        second.enhanced[..., : prefix.shape[-1]],
        atol=2e-5,
        rtol=2e-5,
    )


def test_recipe8_native_stream_matches_offline(tmp_path: Path) -> None:
    torch.manual_seed(18_001)
    model = _model(tmp_path)
    waveform = torch.randn(1, 1, 1_753)
    expected = model(waveform).enhanced
    streamer = V18TwoStageNativeStreamer(model)
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
    assert state.tensor_elements() < 20_000
