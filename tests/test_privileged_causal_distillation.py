from __future__ import annotations

from types import SimpleNamespace

import torch

from cnvqg.training import PrivilegedCausalDistillationLoss


def test_privileged_distillation_is_finite_and_updates_only_student() -> None:
    torch.manual_seed(46)
    noisy = torch.randn(2, 1, 960)
    clean = 0.8 * noisy
    student_wave = noisy.clone().requires_grad_(True)
    teacher_wave = (clean + 0.02 * torch.randn_like(clean)).requires_grad_(True)
    loss = PrivilegedCausalDistillationLoss(
        waveform_weight=0.02,
        compressed_complex_weight=0.03,
        log_magnitude_weight=0.05,
    )
    output = loss(
        SimpleNamespace(enhanced=student_wave),
        SimpleNamespace(enhanced=teacher_wave),
        clean=clean,
        noisy=noisy,
    )
    output.total.backward()
    assert torch.isfinite(output.total)
    assert student_wave.grad is not None and torch.isfinite(student_wave.grad).all()
    assert teacher_wave.grad is None
    assert 0.0 <= output.teacher_bin_fraction.item() <= 1.0
    assert 0.0 <= output.teacher_sample_fraction.item() <= 1.0


def test_teacher_worse_than_noisy_receives_lower_confidence() -> None:
    torch.manual_seed(47)
    clean = torch.randn(1, 1, 960) * 0.1
    noisy = clean + 0.01 * torch.randn_like(clean)
    student = SimpleNamespace(enhanced=noisy.clone())
    good = SimpleNamespace(enhanced=clean.clone())
    bad = SimpleNamespace(enhanced=clean + torch.randn_like(clean))
    loss = PrivilegedCausalDistillationLoss(log_magnitude_weight=0.05)
    good_output = loss(student, good, clean=clean, noisy=noisy)
    bad_output = loss(student, bad, clean=clean, noisy=noisy)
    assert good_output.teacher_bin_fraction > bad_output.teacher_bin_fraction
