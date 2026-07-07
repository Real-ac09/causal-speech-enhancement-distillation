from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def to_numpy_1d(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()

    x = np.asarray(x, dtype=np.float32)
    return np.squeeze(x)


def align_numpy(
    estimate: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    estimate = to_numpy_1d(estimate)
    target = to_numpy_1d(target)

    min_len = min(len(estimate), len(target))

    return estimate[:min_len], target[:min_len]


def si_sdr_np(
    estimate: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-8,
) -> float:
    estimate, target = align_numpy(estimate, target)

    estimate = estimate - np.mean(estimate)
    target = target - np.mean(target)

    target_energy = np.sum(target ** 2) + eps
    projection = np.sum(estimate * target) * target / target_energy
    noise = estimate - projection

    ratio = np.sum(projection ** 2) / (np.sum(noise ** 2) + eps)

    return float(10.0 * np.log10(ratio + eps))


def safe_pesq(
    clean: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
) -> Optional[float]:
    try:
        from pesq import pesq

        estimate, clean = align_numpy(estimate, clean)

        mode = "wb" if sample_rate == 16000 else "nb"
        return float(pesq(sample_rate, clean, estimate, mode))
    except Exception:
        return None


def safe_stoi(
    clean: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
    extended: bool = False,
) -> Optional[float]:
    try:
        from pystoi.stoi import stoi

        estimate, clean = align_numpy(estimate, clean)

        return float(stoi(clean, estimate, sample_rate, extended=extended))
    except Exception:
        return None


def compute_speech_metrics(
    noisy: np.ndarray,
    enhanced: np.ndarray,
    clean: np.ndarray,
    sample_rate: int,
) -> Dict[str, Optional[float]]:
    noisy, clean = align_numpy(noisy, clean)
    enhanced, clean = align_numpy(enhanced, clean)

    noisy_si_sdr = si_sdr_np(noisy, clean)
    enhanced_si_sdr = si_sdr_np(enhanced, clean)

    return {
        "noisy_si_sdr": noisy_si_sdr,
        "enhanced_si_sdr": enhanced_si_sdr,
        "si_sdr_improvement": enhanced_si_sdr - noisy_si_sdr,
        "noisy_pesq": safe_pesq(clean, noisy, sample_rate),
        "enhanced_pesq": safe_pesq(clean, enhanced, sample_rate),
        "noisy_stoi": safe_stoi(clean, noisy, sample_rate, extended=False),
        "enhanced_stoi": safe_stoi(clean, enhanced, sample_rate, extended=False),
        "noisy_estoi": safe_stoi(clean, noisy, sample_rate, extended=True),
        "enhanced_estoi": safe_stoi(clean, enhanced, sample_rate, extended=True),
    }
