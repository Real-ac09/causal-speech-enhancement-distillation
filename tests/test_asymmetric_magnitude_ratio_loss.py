from __future__ import annotations

import pytest
import torch

from cnvqg.losses import EnhancementLoss


def _criterion(weight: float) -> EnhancementLoss:
    return EnhancementLoss(
        waveform_l1_weight=0.0,
        si_sdr_weight=0.0,
        stft_weight=0.0,
        vq_weight=0.0,
        complex_stft_weight=0.0,
        magnitude_ratio_weight=1.0,
        magnitude_ratio_loss="l1",
        magnitude_ratio_underestimation_weight=weight,
        tf_detail_n_fft=8,
        tf_detail_hop_length=4,
        tf_detail_win_length=8,
        tf_detail_center=False,
    )


def _ratio_loss(criterion: EnhancementLoss, mask_value: float) -> torch.Tensor:
    torch.manual_seed(110)
    waveform = torch.randn(1, 1, 64)
    bins, frames = 5, 15
    mask = torch.full((1, bins, frames), mask_value)
    output = criterion(
        enhanced=waveform,
        clean=waveform,
        vq_loss=waveform.new_tensor(0.0),
        noisy=waveform,
        estimated_magnitude=torch.ones(1, bins, frames),
        magnitude_mask=mask,
        estimated_phase=torch.zeros(1, bins, frames),
    )
    return output.magnitude_ratio


def test_default_weight_preserves_symmetric_l1_ratio_loss() -> None:
    criterion = _criterion(1.0)
    under = _ratio_loss(criterion, 0.5)
    over = _ratio_loss(criterion, 1.5)
    assert torch.allclose(under, over)


def test_underestimation_weight_targets_only_speech_removal() -> None:
    criterion = _criterion(2.0)
    under = _ratio_loss(criterion, 0.5)
    over = _ratio_loss(criterion, 1.5)
    assert torch.allclose(under, 2.0 * over)


def test_underestimation_weight_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _criterion(0.9)
