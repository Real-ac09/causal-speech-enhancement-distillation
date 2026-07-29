from __future__ import annotations

import torch

from cnvqg.models.continuous_adaptive_tf_mamba_v42 import (
    ContinuousAdaptiveTFMambaV42,
)
from cnvqg.models.factory import build_model


def test_v42_factory_uses_continuous_three_depth_defaults() -> None:
    model = build_model(
        {
            "architecture": "continuous_adaptive_tf_mamba_v42",
            "variant": "tiny",
            "use_mamba": False,
        }
    )
    assert isinstance(model, ContinuousAdaptiveTFMambaV42)
    assert model.max_iterations == 3
    assert not model.use_noise_codebook
    assert model.condition_dynamics
    assert model.cell.condition_dynamics
    assert model.adaptive_iterations


def test_v42_forward_uses_continuous_states_and_normalized_mixture() -> None:
    model = ContinuousAdaptiveTFMambaV42(
        variant="tiny", use_mamba=False, noise_segment_frames=8
    ).eval()
    noisy = torch.randn(2, 1, 8192)
    with torch.no_grad():
        output = model(noisy)
    assert output.enhanced.shape == noisy.shape
    assert output.halting_probabilities.shape == (2, 3)
    assert torch.all(output.code_indices == -1)
    torch.testing.assert_close(
        output.halting_probabilities.sum(-1),
        torch.ones(2),
        atol=1e-6,
        rtol=1e-6,
    )
