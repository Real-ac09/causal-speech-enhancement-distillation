from __future__ import annotations

import torch

from cnvqg.models.causal_scale_aware_mamba_v11 import (
    CausalScaleAwareMambaV11,
    ContextualScaleAdapter,
)
from cnvqg.models.factory import build_model
from cnvqg.models.predictive_noise_vq_mamba_v8 import PredictiveNoiseVQMambaV8


def _small(**overrides) -> CausalScaleAwareMambaV11:
    config = {
        "variant": "student",
        "channels": 32,
        "noise_dim": 16,
        "temporal_layers": 1,
        "use_mamba": False,
        "auxiliary_vq": False,
        "reconstruction_mode": "direct_scalar_mask",
        "phase_residual_scale": 0.0,
        "scale_adapter_hidden_channels": 8,
        "enforce_parameter_cap": False,
    }
    config.update(overrides)
    return CausalScaleAwareMambaV11(**config)


def _load_compatible(target: torch.nn.Module, source: torch.nn.Module) -> None:
    state = target.state_dict()
    compatible = {
        key: value
        for key, value in source.state_dict().items()
        if key in state and state[key].shape == value.shape
    }
    target.load_state_dict(compatible, strict=False)


def test_factory_and_student_parameter_cap() -> None:
    model = build_model(
        {
            "architecture": "cnvqg_v11",
            "variant": "student",
            "auxiliary_vq": False,
            "reconstruction_mode": "direct_scalar_mask",
            "phase_residual_scale": 0.0,
        }
    )
    assert isinstance(model, CausalScaleAwareMambaV11)
    assert model.parameter_count() <= 1_100_000


def test_zero_initialised_adapter_preserves_v8_checkpoint_function() -> None:
    torch.manual_seed(111)
    control = PredictiveNoiseVQMambaV8(
        channels=32,
        noise_dim=16,
        temporal_layers=1,
        use_mamba=False,
        auxiliary_vq=False,
        reconstruction_mode="direct_scalar_mask",
        phase_residual_scale=0.0,
        enforce_parameter_cap=False,
    ).eval()
    candidate = _small().eval()
    _load_compatible(candidate, control)
    waveform = torch.randn(1, 1, 2400)
    with torch.inference_mode():
        expected = control(waveform).enhanced
        actual = candidate(waveform).enhanced
    assert torch.equal(actual, expected)


def test_frequency_coordinate_permits_bin_specific_correction_without_time_mixing() -> None:
    torch.manual_seed(112)
    adapter = ContextualScaleAdapter(8, 4, use_frequency_coordinate=True).eval()
    adapter.output.weight.data.normal_(0.0, 0.1)
    decoded = torch.zeros(1, 8, 17, 5)
    magnitude = torch.ones(1, 17, 5)
    with torch.inference_mode():
        correction = adapter(decoded, magnitude, 0.3)
    assert not torch.equal(correction[:, :, 0], correction[:, :, -1])
    changed = magnitude.clone()
    changed[..., 3] *= 2.0
    with torch.inference_mode():
        changed_output = adapter(decoded, changed, 0.3)
    assert torch.equal(correction[..., :3], changed_output[..., :3])
    assert torch.equal(correction[..., 4:], changed_output[..., 4:])


def test_arbitrary_chunks_match_whole_utterance() -> None:
    torch.manual_seed(113)
    model = _small().eval()
    noisy = torch.randn(1, 1, 2371)
    with torch.inference_mode():
        expected = model(noisy).enhanced
        state = model.init_stream_state(1, noisy.device, noisy.dtype)
        chunks = []
        offset = 0
        for size in (13, 401, 7, 963, 211, 776):
            output, state = model.forward_chunk(noisy[..., offset : offset + size], state)
            chunks.append(output)
            offset += size
        tail, _ = model.flush(state)
        chunks.append(tail)
    actual = torch.cat(chunks, -1)
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)
