from __future__ import annotations

import pytest
import torch

from cnvqg.losses import EnhancementLoss


def _criterion(**kwargs) -> EnhancementLoss:
    return EnhancementLoss(
        waveform_l1_weight=0.0,
        si_sdr_weight=0.0,
        stft_weight=0.0,
        vq_weight=0.0,
        gate_supervision_weight=1.0,
        **kwargs,
    )


def test_gate_supervision_broadcasts_item_targets() -> None:
    prediction = torch.tensor(
        [[[0.2], [0.3]], [[0.7], [0.8]]],
        requires_grad=True,
    )
    output = _criterion(gate_supervision_loss="l1")(
        enhanced=torch.zeros(2, 1, 32),
        clean=torch.zeros(2, 1, 32),
        vq_loss=torch.tensor(0.0),
        gate_strength=prediction,
        gate_target_strength=torch.tensor([0.25, 0.75]),
    )
    assert output.gate_supervision.item() == pytest.approx(0.05)
    output.total.backward()
    assert prediction.grad is not None


def test_gate_supervision_requires_target() -> None:
    with pytest.raises(ValueError, match="requires predicted and target"):
        _criterion()(
            enhanced=torch.zeros(1, 1, 32),
            clean=torch.zeros(1, 1, 32),
            vq_loss=torch.tensor(0.0),
            gate_strength=torch.full((1, 2, 1), 0.5),
        )


def test_zero_gate_supervision_weight_preserves_legacy_call() -> None:
    output = EnhancementLoss(
        waveform_l1_weight=1.0,
        si_sdr_weight=0.0,
        stft_weight=0.0,
        vq_weight=0.0,
    )(
        enhanced=torch.zeros(1, 1, 32),
        clean=torch.ones(1, 1, 32),
        vq_loss=torch.tensor(0.0),
    )
    assert output.gate_supervision.item() == 0.0
