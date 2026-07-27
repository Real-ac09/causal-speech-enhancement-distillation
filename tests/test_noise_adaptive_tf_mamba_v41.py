from __future__ import annotations

import torch

from cnvqg.losses import EnhancementLoss
from cnvqg.models.factory import build_model
from cnvqg.models.noise_adaptive_tf_mamba_v41 import NoiseAdaptiveTFMambaV41


def test_v41_factory_size_order_and_parameter_budget() -> None:
    sizes = []
    for variant in ("tiny", "base", "large", "xl"):
        model = build_model(
            {
                "architecture": "noise_adaptive_tf_mamba_v41",
                "variant": variant,
                "use_mamba": False,
            }
        )
        sizes.append(sum(parameter.numel() for parameter in model.parameters()))
    assert sizes == sorted(sizes)
    assert sizes[-1] < 1_200_000


def test_v41_forward_has_segment_codes_and_dual_phase_outputs() -> None:
    model = NoiseAdaptiveTFMambaV41(
        variant="tiny",
        use_mamba=False,
        max_iterations=2,
        noise_segment_frames=8,
    ).eval()
    noisy = torch.randn(2, 1, 8192)
    with torch.no_grad():
        output = model(noisy)
    assert output.enhanced.shape == noisy.shape
    assert output.code_indices.shape[1] > 1
    assert output.noise_state.shape[:2] == output.code_indices.shape
    assert output.phase_candidate.shape == output.estimated_phase.shape
    assert output.phase_confidence.shape == output.estimated_phase.shape
    torch.testing.assert_close(
        output.halting_probabilities.sum(-1), torch.ones(2), atol=1e-6, rtol=1e-6
    )


def test_v41_phase_structure_loss_reaches_both_decoder_branches() -> None:
    model = NoiseAdaptiveTFMambaV41(
        variant="tiny", use_mamba=False, max_iterations=2
    ).train()
    noisy = torch.randn(1, 1, 4096)
    clean = torch.randn_like(noisy)
    output = model(noisy)
    criterion = EnhancementLoss(
        stft_fft_sizes=(256,),
        stft_hop_sizes=(64,),
        stft_win_lengths=(256,),
        magnitude_weight=1.0,
        phase_weight=0.2,
        group_delay_weight=0.05,
        instantaneous_frequency_weight=0.05,
        phase_confidence_weight=0.05,
    )
    loss = criterion(
        enhanced=output.enhanced,
        clean=clean,
        vq_loss=output.vq.loss,
        noisy=noisy,
        estimated_magnitude=output.estimated_magnitude,
        estimated_phase=output.estimated_phase,
        phase_candidate=output.phase_candidate,
        phase_confidence=output.phase_confidence,
        expected_iterations=output.expected_iterations,
    )
    loss.total.backward()
    assert torch.isfinite(loss.total)
    assert model.magnitude_head.weight.grad is not None
    assert model.phase_head.weight.grad is not None
    assert model.decoder.up_one[0].weight.grad is not None
