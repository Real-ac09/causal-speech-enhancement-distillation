from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn


def si_sdr_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    estimate = estimate.squeeze(1)
    target = target.squeeze(1)

    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    projection = torch.sum(estimate * target, dim=-1, keepdim=True) * target / target_energy
    noise = estimate - projection

    ratio = torch.sum(projection ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps)
    si_sdr = 10.0 * torch.log10(ratio + eps)

    return -si_sdr.mean()


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes: Sequence[int] = (512, 1024, 2048),
        hop_sizes: Sequence[int] = (128, 256, 512),
        win_lengths: Sequence[int] = (512, 1024, 2048),
        spectral_convergence_weight: float = 1.0,
        log_mag_weight: float = 1.0,
        mag_l1_weight: float = 0.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()

        if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths)):
            raise ValueError("fft_sizes, hop_sizes and win_lengths must have the same length.")

        self.fft_sizes = tuple(int(x) for x in fft_sizes)
        self.hop_sizes = tuple(int(x) for x in hop_sizes)
        self.win_lengths = tuple(int(x) for x in win_lengths)

        self.spectral_convergence_weight = float(spectral_convergence_weight)
        self.log_mag_weight = float(log_mag_weight)
        self.mag_l1_weight = float(mag_l1_weight)
        self.eps = float(eps)

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        estimate = estimate.squeeze(1)
        target = target.squeeze(1)

        total_loss = estimate.new_tensor(0.0)

        for n_fft, hop_length, win_length in zip(
            self.fft_sizes,
            self.hop_sizes,
            self.win_lengths,
        ):
            window = torch.hann_window(win_length, device=estimate.device)

            estimate_stft = torch.stft(
                estimate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                return_complex=True,
            )

            target_stft = torch.stft(
                target,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                return_complex=True,
            )

            estimate_mag = estimate_stft.abs().clamp_min(self.eps)
            target_mag = target_stft.abs().clamp_min(self.eps)

            diff = (target_mag - estimate_mag).reshape(target_mag.shape[0], -1)
            target_flat = target_mag.reshape(target_mag.shape[0], -1)

            spectral_convergence = (
                torch.linalg.vector_norm(diff, ord=2, dim=1)
                / (torch.linalg.vector_norm(target_flat, ord=2, dim=1) + self.eps)
            ).mean()

            log_mag = F.l1_loss(torch.log(estimate_mag), torch.log(target_mag))
            mag_l1 = F.l1_loss(estimate_mag, target_mag)

            total_loss = total_loss + (
                self.spectral_convergence_weight * spectral_convergence
                + self.log_mag_weight * log_mag
                + self.mag_l1_weight * mag_l1
            )

        return total_loss / len(self.fft_sizes)


def hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq / 700.0)


def mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def create_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: Optional[float],
    device: torch.device,
) -> torch.Tensor:
    if f_max is None:
        f_max = float(sample_rate) / 2.0

    min_mel = hz_to_mel(torch.tensor(float(f_min), device=device))
    max_mel = hz_to_mel(torch.tensor(float(f_max), device=device))

    mel_points = torch.linspace(min_mel, max_mel, n_mels + 2, device=device)
    hz_points = mel_to_hz(mel_points)

    bin_freqs = torch.linspace(0.0, float(sample_rate) / 2.0, n_fft // 2 + 1, device=device)

    filters = []
    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        lower = (bin_freqs - left) / (center - left + 1e-8)
        upper = (right - bin_freqs) / (right - center + 1e-8)

        filt = torch.clamp(torch.minimum(lower, upper), min=0.0)
        filters.append(filt)

    return torch.stack(filters, dim=0)


class MelSpectralLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 80,
        f_min: float = 50.0,
        f_max: Optional[float] = 7600.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()

        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.n_mels = int(n_mels)
        self.f_min = float(f_min)
        self.f_max = f_max
        self.eps = float(eps)

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        estimate = estimate.squeeze(1)
        target = target.squeeze(1)

        window = torch.hann_window(self.win_length, device=estimate.device)

        estimate_stft = torch.stft(
            estimate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )

        target_stft = torch.stft(
            target,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )

        estimate_mag = estimate_stft.abs().clamp_min(self.eps)
        target_mag = target_stft.abs().clamp_min(self.eps)

        mel_fb = create_mel_filterbank(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            device=estimate.device,
        )

        estimate_mel = torch.einsum("mf,bft->bmt", mel_fb, estimate_mag).clamp_min(self.eps)
        target_mel = torch.einsum("mf,bft->bmt", mel_fb, target_mag).clamp_min(self.eps)

        return F.l1_loss(torch.log(estimate_mel), torch.log(target_mel))



class ComplexSTFTConsistencyLoss(nn.Module):
    """
    Explicit complex STFT consistency loss.

    This penalises the enhanced waveform if its complex STFT differs from the
    clean waveform STFT. It is useful for complex-mask models because it gives
    direct supervision to real/imaginary structure, not only waveform or
    magnitude/mel structure.
    """

    def __init__(
        self,
        fft_sizes: Sequence[int] = (512, 1024),
        hop_sizes: Sequence[int] = (128, 256),
        win_lengths: Sequence[int] = (512, 1024),
        complex_weight: float = 1.0,
        log_mag_weight: float = 0.5,
        compression_power: float = 1.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()

        if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths)):
            raise ValueError("fft_sizes, hop_sizes and win_lengths must have the same length.")

        self.fft_sizes = tuple(int(x) for x in fft_sizes)
        self.hop_sizes = tuple(int(x) for x in hop_sizes)
        self.win_lengths = tuple(int(x) for x in win_lengths)
        self.complex_weight = float(complex_weight)
        self.log_mag_weight = float(log_mag_weight)
        self.compression_power = float(compression_power)
        self.eps = float(eps)

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        estimate = estimate.squeeze(1)
        target = target.squeeze(1)

        total_loss = estimate.new_tensor(0.0)

        for n_fft, hop_length, win_length in zip(
            self.fft_sizes,
            self.hop_sizes,
            self.win_lengths,
        ):
            window = torch.hann_window(win_length, device=estimate.device)

            estimate_stft = torch.stft(
                estimate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                return_complex=True,
            )

            target_stft = torch.stft(
                target,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                return_complex=True,
            )

            estimate_mag = estimate_stft.abs().clamp_min(self.eps)
            target_mag = target_stft.abs().clamp_min(self.eps)

            estimate_complex = estimate_stft
            target_complex = target_stft
            if self.compression_power != 1.0:
                estimate_complex = estimate_mag.pow(self.compression_power) * (
                    estimate_stft / estimate_mag
                )
                target_complex = target_mag.pow(self.compression_power) * (
                    target_stft / target_mag
                )
            # Normalised compressed-complex error keeps this loss scale stable.
            complex_error = torch.abs(estimate_complex - target_complex).mean()
            target_scale = target_complex.abs().mean().detach().clamp_min(self.eps)
            normalised_complex_error = complex_error / target_scale

            log_mag_error = F.l1_loss(torch.log(estimate_mag), torch.log(target_mag))

            total_loss = total_loss + (
                self.complex_weight * normalised_complex_error
                + self.log_mag_weight * log_mag_error
            )

        return total_loss / len(self.fft_sizes)

@dataclass
class LossOutput:
    total: torch.Tensor
    waveform_l1: torch.Tensor
    si_sdr: torch.Tensor
    stft: torch.Tensor
    vq: torch.Tensor
    mel: torch.Tensor
    complex_stft: torch.Tensor
    noise_prediction: torch.Tensor
    noise_spectrum: torch.Tensor
    magnitude: torch.Tensor
    magnitude_log: torch.Tensor
    magnitude_ratio: torch.Tensor
    phase: torch.Tensor
    group_delay: torch.Tensor
    instantaneous_frequency: torch.Tensor
    phase_confidence: torch.Tensor
    compute: torch.Tensor
    gate_identity: torch.Tensor
    gate_smoothness: torch.Tensor
    gate_supervision: torch.Tensor
    gate_classification: torch.Tensor
    gate_ordinal: torch.Tensor
    gate_strength_regression: torch.Tensor
    gate_separation: torch.Tensor
    gate_utility: torch.Tensor
    gate_violation: torch.Tensor
    gate_feasibility: torch.Tensor
    gate_policy: torch.Tensor
    gate_metric_delta: torch.Tensor

    def as_dict(self) -> Dict[str, float]:
        return {
            "loss_total": float(self.total.detach().cpu()),
            "loss_waveform_l1": float(self.waveform_l1.detach().cpu()),
            "loss_si_sdr": float(self.si_sdr.detach().cpu()),
            "loss_stft": float(self.stft.detach().cpu()),
            "loss_vq": float(self.vq.detach().cpu()),
            "loss_mel": float(self.mel.detach().cpu()),
            "loss_complex_stft": float(self.complex_stft.detach().cpu()),
            "loss_noise_prediction": float(self.noise_prediction.detach().cpu()),
            "loss_noise_spectrum": float(self.noise_spectrum.detach().cpu()),
            "loss_magnitude": float(self.magnitude.detach().cpu()),
            "loss_magnitude_log": float(self.magnitude_log.detach().cpu()),
            "loss_magnitude_ratio": float(self.magnitude_ratio.detach().cpu()),
            "loss_phase": float(self.phase.detach().cpu()),
            "loss_group_delay": float(self.group_delay.detach().cpu()),
            "loss_instantaneous_frequency": float(
                self.instantaneous_frequency.detach().cpu()
            ),
            "loss_phase_confidence": float(self.phase_confidence.detach().cpu()),
            "loss_compute": float(self.compute.detach().cpu()),
            "loss_gate_identity": float(self.gate_identity.detach().cpu()),
            "loss_gate_smoothness": float(self.gate_smoothness.detach().cpu()),
            "loss_gate_supervision": float(
                self.gate_supervision.detach().cpu()
            ),
            "loss_gate_classification": float(
                self.gate_classification.detach().cpu()
            ),
            "loss_gate_ordinal": float(self.gate_ordinal.detach().cpu()),
            "loss_gate_strength_regression": float(
                self.gate_strength_regression.detach().cpu()
            ),
            "loss_gate_separation": float(
                self.gate_separation.detach().cpu()
            ),
            "loss_gate_utility": float(self.gate_utility.detach().cpu()),
            "loss_gate_violation": float(
                self.gate_violation.detach().cpu()
            ),
            "loss_gate_feasibility": float(
                self.gate_feasibility.detach().cpu()
            ),
            "loss_gate_policy": float(self.gate_policy.detach().cpu()),
            "loss_gate_metric_delta": float(
                self.gate_metric_delta.detach().cpu()
            ),
        }


class EnhancementLoss(nn.Module):
    def __init__(
        self,
        waveform_l1_weight: float = 1.0,
        si_sdr_weight: float = 0.5,
        stft_weight: float = 1.0,
        vq_weight: float = 0.25,
        stft_fft_sizes: Sequence[int] = (512, 1024, 2048),
        stft_hop_sizes: Sequence[int] = (128, 256, 512),
        stft_win_lengths: Sequence[int] = (512, 1024, 2048),
        stft_spectral_convergence_weight: float = 1.0,
        stft_log_mag_weight: float = 1.0,
        stft_mag_l1_weight: float = 0.0,
        mel_weight: float = 0.0,
        mel_sample_rate: int = 16000,
        mel_n_fft: int = 512,
        mel_hop_length: int = 160,
        mel_win_length: int = 400,
        mel_n_mels: int = 80,
        mel_f_min: float = 50.0,
        mel_f_max: Optional[float] = 7600.0,
        complex_stft_weight: float = 0.0,
        complex_stft_fft_sizes: Sequence[int] = (512, 1024),
        complex_stft_hop_sizes: Sequence[int] = (128, 256),
        complex_stft_win_lengths: Sequence[int] = (512, 1024),
        complex_stft_complex_weight: float = 1.0,
        complex_stft_log_mag_weight: float = 0.5,
        complex_stft_compression_power: float = 1.0,
        noise_prediction_weight: float = 0.0,
        noise_spectrum_weight: float = 0.0,
        noise_spectrum_compression_power: float = 0.3,
        noise_prediction_n_fft: int = 512,
        noise_prediction_hop_length: int = 128,
        noise_prediction_win_length: int = 512,
        magnitude_weight: float = 0.0,
        magnitude_log_weight: float = 0.0,
        magnitude_ratio_weight: float = 0.0,
        magnitude_ratio_cap: float = 1.0,
        magnitude_ratio_loss: str = "smooth_l1",
        magnitude_ratio_charbonnier_eps: float = 1e-3,
        magnitude_ratio_underestimation_weight: float = 1.0,
        phase_weight: float = 0.0,
        group_delay_weight: float = 0.0,
        instantaneous_frequency_weight: float = 0.0,
        phase_confidence_weight: float = 0.0,
        phase_confidence_temperature: float = 1.0,
        compute_weight: float = 0.0,
        tf_detail_n_fft: int = 512,
        tf_detail_hop_length: int = 128,
        tf_detail_win_length: int = 512,
        tf_detail_center: bool = True,
        tf_detail_magnitude_power: float = 0.3,
        waveform_charbonnier_eps: float = 0.0,
        magnitude_equal_loudness: bool = False,
        gate_identity_weight: float = 0.0,
        gate_identity_snr_threshold_db: float = 15.0,
        gate_identity_snr_temperature_db: float = 2.0,
        gate_smoothness_weight: float = 0.0,
        gate_supervision_weight: float = 0.0,
        gate_supervision_loss: str = "smooth_l1",
        gate_supervision_beta: float = 0.05,
        gate_classification_weight: float = 0.0,
        gate_class_weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0),
        gate_ordinal_weight: float = 0.0,
        gate_strength_regression_weight: float = 0.0,
        gate_strength_regression_beta: float = 0.1,
        gate_separation_weight: float = 0.0,
        gate_separation_scale: float = 0.5,
        gate_separation_minimum_target_distance: float = 0.5,
        gate_utility_weight: float = 0.0,
        gate_violation_weight: float = 0.0,
        gate_feasibility_weight: float = 0.0,
        gate_policy_weight: float = 0.0,
        gate_metric_delta_weight: float = 0.0,
        gate_utility_beta: float = 0.5,
        gate_violation_beta: float = 0.25,
        gate_metric_delta_beta: float = 0.5,
        gate_utility_clip: float = 10.0,
        gate_violation_log_clip: float = 5.0,
        gate_metric_delta_clip: float = 10.0,
        gate_recipe5_burn_in_fraction: float = 0.0,
        gate_metric_delta_scales: Sequence[float] = (
            0.1,
            1.0,
            0.01,
            0.02,
        ),
    ) -> None:
        super().__init__()

        self.waveform_l1_weight = float(waveform_l1_weight)
        self.si_sdr_weight = float(si_sdr_weight)
        self.stft_weight = float(stft_weight)
        self.vq_weight = float(vq_weight)
        self.mel_weight = float(mel_weight)
        self.complex_stft_weight = float(complex_stft_weight)
        self.noise_prediction_weight = float(noise_prediction_weight)
        self.noise_spectrum_weight = float(noise_spectrum_weight)
        self.noise_spectrum_compression_power = float(noise_spectrum_compression_power)
        if not 0.0 < self.noise_spectrum_compression_power <= 1.0:
            raise ValueError("noise_spectrum_compression_power must be in (0, 1]")
        self.noise_prediction_n_fft = int(noise_prediction_n_fft)
        self.noise_prediction_hop_length = int(noise_prediction_hop_length)
        self.noise_prediction_win_length = int(noise_prediction_win_length)
        self.magnitude_weight = float(magnitude_weight)
        self.magnitude_log_weight = float(magnitude_log_weight)
        self.magnitude_ratio_weight = float(magnitude_ratio_weight)
        self.magnitude_ratio_cap = float(magnitude_ratio_cap)
        self.magnitude_ratio_loss = str(magnitude_ratio_loss).lower()
        self.magnitude_ratio_charbonnier_eps = float(magnitude_ratio_charbonnier_eps)
        self.magnitude_ratio_underestimation_weight = float(
            magnitude_ratio_underestimation_weight
        )
        if self.magnitude_ratio_cap <= 0.0:
            raise ValueError("magnitude_ratio_cap must be positive")
        if self.magnitude_ratio_loss not in {"smooth_l1", "l1", "charbonnier"}:
            raise ValueError(
                "magnitude_ratio_loss must be 'smooth_l1', 'l1', or 'charbonnier'"
            )
        if self.magnitude_ratio_charbonnier_eps <= 0.0:
            raise ValueError("magnitude_ratio_charbonnier_eps must be positive")
        if self.magnitude_ratio_underestimation_weight < 1.0:
            raise ValueError(
                "magnitude_ratio_underestimation_weight must be at least 1"
            )
        self.phase_weight = float(phase_weight)
        self.group_delay_weight = float(group_delay_weight)
        self.instantaneous_frequency_weight = float(
            instantaneous_frequency_weight
        )
        self.phase_confidence_weight = float(phase_confidence_weight)
        self.phase_confidence_temperature = float(phase_confidence_temperature)
        if self.phase_confidence_temperature <= 0.0:
            raise ValueError("phase_confidence_temperature must be positive")
        self.compute_weight = float(compute_weight)
        self.tf_detail_n_fft = int(tf_detail_n_fft)
        self.tf_detail_hop_length = int(tf_detail_hop_length)
        self.tf_detail_win_length = int(tf_detail_win_length)
        self.tf_detail_center = bool(tf_detail_center)
        self.tf_detail_magnitude_power = float(tf_detail_magnitude_power)
        self.waveform_charbonnier_eps = float(waveform_charbonnier_eps)
        self.magnitude_equal_loudness = bool(magnitude_equal_loudness)
        self.gate_identity_weight = float(gate_identity_weight)
        self.gate_identity_snr_threshold_db = float(
            gate_identity_snr_threshold_db
        )
        self.gate_identity_snr_temperature_db = float(
            gate_identity_snr_temperature_db
        )
        self.gate_smoothness_weight = float(gate_smoothness_weight)
        self.gate_supervision_weight = float(gate_supervision_weight)
        self.gate_supervision_loss = str(gate_supervision_loss).lower()
        self.gate_supervision_beta = float(gate_supervision_beta)
        self.gate_classification_weight = float(gate_classification_weight)
        class_weights = torch.as_tensor(
            tuple(gate_class_weights),
            dtype=torch.float32,
        )
        if class_weights.ndim != 1 or class_weights.numel() < 2:
            raise ValueError(
                "gate_class_weights must contain at least two values"
            )
        if not bool(torch.all(class_weights > 0.0)):
            raise ValueError("gate_class_weights must be strictly positive")
        self.register_buffer("gate_class_weights", class_weights)
        self.gate_ordinal_weight = float(gate_ordinal_weight)
        self.gate_strength_regression_weight = float(
            gate_strength_regression_weight
        )
        self.gate_strength_regression_beta = float(
            gate_strength_regression_beta
        )
        self.gate_separation_weight = float(gate_separation_weight)
        self.gate_separation_scale = float(gate_separation_scale)
        self.gate_separation_minimum_target_distance = float(
            gate_separation_minimum_target_distance
        )
        self.gate_utility_weight = float(gate_utility_weight)
        self.gate_violation_weight = float(gate_violation_weight)
        self.gate_feasibility_weight = float(gate_feasibility_weight)
        self.gate_policy_weight = float(gate_policy_weight)
        self.gate_metric_delta_weight = float(gate_metric_delta_weight)
        self.gate_utility_beta = float(gate_utility_beta)
        self.gate_violation_beta = float(gate_violation_beta)
        self.gate_metric_delta_beta = float(gate_metric_delta_beta)
        self.gate_utility_clip = float(gate_utility_clip)
        self.gate_violation_log_clip = float(gate_violation_log_clip)
        self.gate_metric_delta_clip = float(gate_metric_delta_clip)
        self.gate_recipe5_burn_in_fraction = float(
            gate_recipe5_burn_in_fraction
        )
        metric_delta_scales = torch.as_tensor(
            tuple(gate_metric_delta_scales),
            dtype=torch.float32,
        )
        if metric_delta_scales.shape != (4,) or not bool(
            torch.all(metric_delta_scales > 0.0)
        ):
            raise ValueError(
                "gate_metric_delta_scales must contain four positive values"
            )
        self.register_buffer(
            "gate_metric_delta_scales",
            metric_delta_scales,
        )
        if (
            self.gate_identity_weight < 0.0
            or self.gate_smoothness_weight < 0.0
            or self.gate_supervision_weight < 0.0
            or self.gate_classification_weight < 0.0
            or self.gate_ordinal_weight < 0.0
            or self.gate_strength_regression_weight < 0.0
            or self.gate_separation_weight < 0.0
            or self.gate_utility_weight < 0.0
            or self.gate_violation_weight < 0.0
            or self.gate_feasibility_weight < 0.0
            or self.gate_policy_weight < 0.0
            or self.gate_metric_delta_weight < 0.0
        ):
            raise ValueError("gate loss weights must be non-negative")
        if self.gate_identity_snr_temperature_db <= 0.0:
            raise ValueError("gate_identity_snr_temperature_db must be positive")
        if self.gate_supervision_loss not in {"l1", "mse", "smooth_l1"}:
            raise ValueError(
                "gate_supervision_loss must be l1, mse, or smooth_l1"
            )
        if self.gate_supervision_beta <= 0.0:
            raise ValueError("gate_supervision_beta must be positive")
        if self.gate_strength_regression_beta <= 0.0:
            raise ValueError(
                "gate_strength_regression_beta must be positive"
            )
        if self.gate_separation_scale < 0.0:
            raise ValueError("gate_separation_scale must be non-negative")
        if not 0.0 <= self.gate_separation_minimum_target_distance <= 1.0:
            raise ValueError(
                "gate_separation_minimum_target_distance must be in [0, 1]"
            )
        if (
            self.gate_utility_beta <= 0.0
            or self.gate_violation_beta <= 0.0
            or self.gate_metric_delta_beta <= 0.0
        ):
            raise ValueError("Recipe-5 smooth-L1 betas must be positive")
        if (
            self.gate_utility_clip <= 0.0
            or self.gate_violation_log_clip <= 0.0
            or self.gate_metric_delta_clip <= 0.0
        ):
            raise ValueError("Recipe-5 target clips must be positive")
        if not 0.0 <= self.gate_recipe5_burn_in_fraction < 1.0:
            raise ValueError(
                "gate_recipe5_burn_in_fraction must be in [0, 1)"
            )

        self.stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=stft_fft_sizes,
            hop_sizes=stft_hop_sizes,
            win_lengths=stft_win_lengths,
            spectral_convergence_weight=stft_spectral_convergence_weight,
            log_mag_weight=stft_log_mag_weight,
            mag_l1_weight=stft_mag_l1_weight,
        )

        self.mel_loss = MelSpectralLoss(
            sample_rate=mel_sample_rate,
            n_fft=mel_n_fft,
            hop_length=mel_hop_length,
            win_length=mel_win_length,
            n_mels=mel_n_mels,
            f_min=mel_f_min,
            f_max=mel_f_max,
        )

        self.complex_stft_loss = ComplexSTFTConsistencyLoss(
            fft_sizes=complex_stft_fft_sizes,
            hop_sizes=complex_stft_hop_sizes,
            win_lengths=complex_stft_win_lengths,
            complex_weight=complex_stft_complex_weight,
            log_mag_weight=complex_stft_log_mag_weight,
            compression_power=complex_stft_compression_power,
        )

    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        vq_loss: torch.Tensor,
        noisy: Optional[torch.Tensor] = None,
        noise_prediction: Optional[torch.Tensor] = None,
        estimated_noise_magnitude: Optional[torch.Tensor] = None,
        estimated_magnitude: Optional[torch.Tensor] = None,
        magnitude_mask: Optional[torch.Tensor] = None,
        estimated_phase: Optional[torch.Tensor] = None,
        phase_candidate: Optional[torch.Tensor] = None,
        phase_confidence: Optional[torch.Tensor] = None,
        expected_iterations: Optional[torch.Tensor] = None,
        gate_strength: Optional[torch.Tensor] = None,
        gate_target_strength: Optional[torch.Tensor] = None,
        gate_logits: Optional[torch.Tensor] = None,
        gate_ordinal_logits: Optional[torch.Tensor] = None,
        gate_target_class: Optional[torch.Tensor] = None,
        gate_frame_mask: Optional[torch.Tensor] = None,
        gate_utility: Optional[torch.Tensor] = None,
        gate_log_violation: Optional[torch.Tensor] = None,
        gate_feasibility_logits: Optional[torch.Tensor] = None,
        gate_metric_deltas: Optional[torch.Tensor] = None,
        gate_target_utility: Optional[torch.Tensor] = None,
        gate_target_violation: Optional[torch.Tensor] = None,
        gate_target_feasible: Optional[torch.Tensor] = None,
        gate_target_policy: Optional[torch.Tensor] = None,
        gate_target_metric_deltas: Optional[torch.Tensor] = None,
        gate_target_metric_mask: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        min_len = min(enhanced.shape[-1], clean.shape[-1])
        enhanced = enhanced[..., :min_len]
        clean = clean[..., :min_len]

        if self.waveform_charbonnier_eps > 0.0:
            waveform_l1 = torch.sqrt(
                (enhanced - clean).square() + self.waveform_charbonnier_eps**2
            ).mean()
        else:
            waveform_l1 = F.l1_loss(enhanced, clean)
        si_sdr = si_sdr_loss(enhanced, clean)
        if self.stft_weight > 0.0:
            stft = self.stft_loss(enhanced, clean)
        else:
            stft = enhanced.new_tensor(0.0)
        vq = vq_loss

        if self.mel_weight > 0.0:
            mel = self.mel_loss(enhanced, clean)
        else:
            mel = enhanced.new_tensor(0.0)

        if self.complex_stft_weight > 0.0:
            complex_stft = self.complex_stft_loss(enhanced, clean)
        else:
            complex_stft = enhanced.new_tensor(0.0)

        target_noise = None
        if self.noise_prediction_weight > 0.0 or self.noise_spectrum_weight > 0.0:
            if noisy is None:
                raise ValueError("Noise objectives require the noisy input")
            if self.noise_prediction_weight > 0.0 and noise_prediction is None:
                raise ValueError(
                    "noise_prediction_weight > 0 requires noisy and noise_prediction"
                )
            noisy = noisy[..., :min_len].squeeze(1)
            clean_wave = clean.squeeze(1)
            noise_wave = noisy - clean_wave
            window = torch.hann_window(
                self.noise_prediction_win_length,
                device=noise_wave.device,
                dtype=noise_wave.dtype,
            )
            target_noise = torch.stft(
                noise_wave,
                n_fft=self.noise_prediction_n_fft,
                hop_length=self.noise_prediction_hop_length,
                win_length=self.noise_prediction_win_length,
                window=window,
                center=False,
                return_complex=True,
            ).abs()
            if self.noise_prediction_weight > 0.0:
                prediction = F.interpolate(
                    noise_prediction,
                    size=target_noise.shape[-1],
                    mode="linear",
                    align_corners=False,
                )
                noise_prediction_loss = F.l1_loss(
                    torch.log1p(prediction),
                    torch.log1p(target_noise),
                )
            else:
                noise_prediction_loss = enhanced.new_tensor(0.0)
        else:
            noise_prediction_loss = enhanced.new_tensor(0.0)

        if self.noise_spectrum_weight > 0.0:
            if estimated_noise_magnitude is None or target_noise is None:
                raise ValueError(
                    "noise_spectrum_weight > 0 requires estimated_noise_magnitude"
                )
            bins = min(estimated_noise_magnitude.shape[-2], target_noise.shape[-2])
            frames = min(estimated_noise_magnitude.shape[-1], target_noise.shape[-1])
            predicted_noise = estimated_noise_magnitude[..., :bins, :frames].clamp_min(1e-7)
            target_noise_aligned = target_noise[..., :bins, :frames].clamp_min(1e-7)
            power = self.noise_spectrum_compression_power
            noise_spectrum_loss = F.l1_loss(
                predicted_noise.pow(power), target_noise_aligned.pow(power)
            )
        else:
            noise_spectrum_loss = enhanced.new_tensor(0.0)

        needs_tf_detail = any(
            weight > 0.0
            for weight in (
                self.magnitude_weight,
                self.magnitude_log_weight,
                self.magnitude_ratio_weight,
                self.phase_weight,
                self.group_delay_weight,
                self.instantaneous_frequency_weight,
                self.phase_confidence_weight,
            )
        )
        if needs_tf_detail:
            if estimated_magnitude is None or estimated_phase is None or noisy is None:
                raise ValueError("TF detail losses require magnitude, phase, and noisy input")
            detail_window = torch.hann_window(
                self.tf_detail_win_length, device=clean.device, dtype=clean.dtype
            )
            clean_detail = torch.stft(
                clean.squeeze(1),
                n_fft=self.tf_detail_n_fft,
                hop_length=self.tf_detail_hop_length,
                win_length=self.tf_detail_win_length,
                window=detail_window,
                center=self.tf_detail_center,
                return_complex=True,
            )
            noisy_detail = torch.stft(
                noisy.squeeze(1),
                n_fft=self.tf_detail_n_fft,
                hop_length=self.tf_detail_hop_length,
                win_length=self.tf_detail_win_length,
                window=detail_window,
                center=self.tf_detail_center,
                return_complex=True,
            )
            bins = min(clean_detail.shape[-2], estimated_magnitude.shape[-2])
            frames = min(clean_detail.shape[-1], estimated_magnitude.shape[-1])
            clean_detail = clean_detail[..., :bins, :frames]
            noisy_detail = noisy_detail[..., :bins, :frames]
            predicted_magnitude = estimated_magnitude[..., :bins, :frames]
            predicted_phase = estimated_phase[..., :bins, :frames]
            clean_magnitude = clean_detail.abs().clamp_min(1e-7)
            magnitude_error = (
                predicted_magnitude.clamp_min(1e-7).pow(self.tf_detail_magnitude_power)
                - clean_magnitude.pow(self.tf_detail_magnitude_power)
            ).abs()
            if self.magnitude_equal_loudness:
                # Smooth A-weighting-like emphasis, bounded to avoid silencing
                # low-frequency supervision. The mean-one normalization keeps
                # the configured loss weight interpretable.
                frequencies = torch.linspace(
                    0.0, 0.5 * 16000.0, bins, device=clean.device
                ).clamp_min(20.0)
                f2 = frequencies.square()
                numerator = (12200.0**2) * f2.square()
                denominator = (
                    (f2 + 20.6**2)
                    * (f2 + 12200.0**2)
                    * torch.sqrt((f2 + 107.7**2) * (f2 + 737.9**2))
                )
                weights = (numerator / denominator.clamp_min(1e-12)).clamp_min(0.1)
                weights = weights / weights.mean()
                magnitude_error = magnitude_error * weights[None, :, None]
            magnitude_loss = magnitude_error.mean()
            magnitude_log_loss = F.l1_loss(
                torch.log1p(predicted_magnitude.clamp_min(0.0)),
                torch.log1p(clean_magnitude),
            )
            noisy_magnitude = noisy_detail.abs().clamp_min(1e-7)
            if magnitude_mask is not None:
                predicted_ratio = magnitude_mask[..., :bins, :frames].float().clamp_min(0.0)
            else:
                # Compatibility fallback for models that do not expose their
                # mask. New ratio-supervised models should pass it directly.
                predicted_ratio = predicted_magnitude.clamp_min(0.0) / noisy_magnitude
            target_ratio = (clean_magnitude / noisy_magnitude).clamp(
                min=0.0, max=self.magnitude_ratio_cap
            )
            # Emphasise bins carrying clean speech while retaining a small
            # floor so noise-only regions still learn attenuation.
            ratio_importance = clean_magnitude.pow(self.tf_detail_magnitude_power)
            ratio_importance = ratio_importance / ratio_importance.mean(
                dim=(-2, -1), keepdim=True
            ).clamp_min(1e-7)
            ratio_importance = ratio_importance.clamp(min=0.1, max=5.0)
            ratio_error = predicted_ratio - target_ratio
            if self.magnitude_ratio_loss == "smooth_l1":
                ratio_penalty = F.smooth_l1_loss(
                    predicted_ratio, target_ratio, reduction="none"
                )
            elif self.magnitude_ratio_loss == "l1":
                ratio_penalty = ratio_error.abs()
            else:
                eps = self.magnitude_ratio_charbonnier_eps
                ratio_penalty = torch.sqrt(ratio_error.square() + eps**2) - eps
            if self.magnitude_ratio_underestimation_weight != 1.0:
                ratio_penalty = ratio_penalty * torch.where(
                    ratio_error < 0.0,
                    self.magnitude_ratio_underestimation_weight,
                    1.0,
                )
            magnitude_ratio_loss = (ratio_penalty * ratio_importance).mean()
            clean_phase = torch.angle(clean_detail)
            phase_error = torch.atan2(
                torch.sin(predicted_phase - clean_phase),
                torch.cos(predicted_phase - clean_phase),
            )
            phase_importance = clean_magnitude / clean_magnitude.mean(
                dim=(-2, -1), keepdim=True
            ).clamp_min(1e-7)
            phase_importance = phase_importance.clamp(max=5.0)
            phase_loss = ((1.0 - torch.cos(phase_error)) * phase_importance).mean()

            if self.group_delay_weight > 0.0:
                predicted_group_delay = torch.atan2(
                    torch.sin(predicted_phase[:, 1:] - predicted_phase[:, :-1]),
                    torch.cos(predicted_phase[:, 1:] - predicted_phase[:, :-1]),
                )
                clean_group_delay = torch.atan2(
                    torch.sin(clean_phase[:, 1:] - clean_phase[:, :-1]),
                    torch.cos(clean_phase[:, 1:] - clean_phase[:, :-1]),
                )
                group_delay_error = torch.atan2(
                    torch.sin(predicted_group_delay - clean_group_delay),
                    torch.cos(predicted_group_delay - clean_group_delay),
                )
                group_importance = 0.5 * (
                    phase_importance[:, 1:] + phase_importance[:, :-1]
                )
                group_delay_loss = (
                    (1.0 - torch.cos(group_delay_error)) * group_importance
                ).mean()
            else:
                group_delay_loss = enhanced.new_tensor(0.0)

            if self.instantaneous_frequency_weight > 0.0:
                predicted_if = torch.atan2(
                    torch.sin(predicted_phase[..., 1:] - predicted_phase[..., :-1]),
                    torch.cos(predicted_phase[..., 1:] - predicted_phase[..., :-1]),
                )
                clean_if = torch.atan2(
                    torch.sin(clean_phase[..., 1:] - clean_phase[..., :-1]),
                    torch.cos(clean_phase[..., 1:] - clean_phase[..., :-1]),
                )
                if_error = torch.atan2(
                    torch.sin(predicted_if - clean_if),
                    torch.cos(predicted_if - clean_if),
                )
                if_importance = 0.5 * (
                    phase_importance[..., 1:] + phase_importance[..., :-1]
                )
                instantaneous_frequency_loss = (
                    (1.0 - torch.cos(if_error)) * if_importance
                ).mean()
            else:
                instantaneous_frequency_loss = enhanced.new_tensor(0.0)

            if self.phase_confidence_weight > 0.0:
                if phase_confidence is None:
                    raise ValueError("phase_confidence_weight requires phase confidence")
                confidence = phase_confidence[..., :bins, :frames].clamp(1e-5, 1.0 - 1e-5)
                if phase_candidate is not None:
                    candidate = phase_candidate[..., :bins, :frames]
                    candidate_error = torch.atan2(
                        torch.sin(candidate - clean_phase),
                        torch.cos(candidate - clean_phase),
                    )
                    confidence_target = torch.exp(
                        -candidate_error.detach().abs()
                        / self.phase_confidence_temperature
                    )
                else:
                    noisy_magnitude = noisy_detail.abs().clamp_min(1e-7)
                    confidence_target = (
                        clean_magnitude / (clean_magnitude + noisy_magnitude)
                    ).mul(2.0).clamp(0.0, 1.0)
                confidence_loss = F.binary_cross_entropy(confidence, confidence_target)
            else:
                confidence_loss = enhanced.new_tensor(0.0)
        else:
            magnitude_loss = enhanced.new_tensor(0.0)
            magnitude_log_loss = enhanced.new_tensor(0.0)
            magnitude_ratio_loss = enhanced.new_tensor(0.0)
            phase_loss = enhanced.new_tensor(0.0)
            group_delay_loss = enhanced.new_tensor(0.0)
            instantaneous_frequency_loss = enhanced.new_tensor(0.0)
            confidence_loss = enhanced.new_tensor(0.0)

        if self.compute_weight > 0.0:
            if expected_iterations is None:
                raise ValueError("compute_weight requires expected_iterations")
            compute_loss = expected_iterations.float().mean()
        else:
            compute_loss = enhanced.new_tensor(0.0)

        if self.gate_identity_weight > 0.0:
            if gate_strength is None or noisy is None:
                raise ValueError(
                    "gate_identity_weight requires gate_strength and noisy input"
                )
            aligned_noisy = noisy[..., :min_len]
            residual_noise = aligned_noisy - clean
            clean_energy = clean.square().mean(dim=(-2, -1)).clamp_min(1e-8)
            noise_energy = residual_noise.square().mean(dim=(-2, -1)).clamp_min(
                1e-8
            )
            snr_db = 10.0 * torch.log10(clean_energy / noise_energy)
            identity_weight = torch.sigmoid(
                (
                    snr_db - self.gate_identity_snr_threshold_db
                )
                / self.gate_identity_snr_temperature_db
            )
            # A zero residual strength makes the final waveform equal to the
            # mixture, which is the required identity behaviour for clean or
            # very-high-SNR inputs.
            per_item_identity = gate_strength.float().square().flatten(
                start_dim=1
            ).mean(dim=1)
            gate_identity_loss = (
                identity_weight * per_item_identity
            ).mean()
        else:
            gate_identity_loss = enhanced.new_tensor(0.0)

        if self.gate_smoothness_weight > 0.0:
            if gate_strength is None:
                raise ValueError(
                    "gate_smoothness_weight requires gate_strength"
                )
            if gate_strength.shape[1] > 1:
                differences = (
                    gate_strength[:, 1:] - gate_strength[:, :-1]
                ).abs()
                if gate_frame_mask is not None:
                    frames = min(
                        gate_strength.shape[1],
                        gate_frame_mask.shape[1],
                    )
                    differences = (
                        gate_strength[:, 1:frames]
                        - gate_strength[:, : frames - 1]
                    ).abs()
                    adjacent_mask = (
                        gate_frame_mask[:, 1:frames]
                        & gate_frame_mask[:, : frames - 1]
                    ).to(
                        device=differences.device,
                        dtype=differences.dtype,
                    )
                    while adjacent_mask.ndim < differences.ndim:
                        adjacent_mask = adjacent_mask.unsqueeze(-1)
                    gate_smoothness_loss = (
                        differences * adjacent_mask
                    ).sum() / adjacent_mask.sum().clamp_min(1.0)
                else:
                    gate_smoothness_loss = differences.mean()
            else:
                gate_smoothness_loss = enhanced.new_tensor(0.0)
        else:
            gate_smoothness_loss = enhanced.new_tensor(0.0)

        if self.gate_supervision_weight > 0.0:
            if gate_strength is None or gate_target_strength is None:
                raise ValueError(
                    "gate_supervision_weight requires predicted and target "
                    "gate strengths"
                )
            target_strength = gate_target_strength.to(
                device=gate_strength.device,
                dtype=gate_strength.dtype,
            )
            while target_strength.ndim < gate_strength.ndim:
                target_strength = target_strength.unsqueeze(-1)
            try:
                target_strength = target_strength.expand_as(gate_strength)
            except RuntimeError as error:
                raise ValueError(
                    "Gate target strength cannot broadcast to prediction: "
                    f"{tuple(gate_target_strength.shape)} versus "
                    f"{tuple(gate_strength.shape)}"
                ) from error
            if self.gate_supervision_loss == "l1":
                gate_supervision_loss = F.l1_loss(
                    gate_strength,
                    target_strength,
                )
            elif self.gate_supervision_loss == "mse":
                gate_supervision_loss = F.mse_loss(
                    gate_strength,
                    target_strength,
                )
            else:
                gate_supervision_loss = F.smooth_l1_loss(
                    gate_strength,
                    target_strength,
                    beta=self.gate_supervision_beta,
                )
        else:
            gate_supervision_loss = enhanced.new_tensor(0.0)

        if self.gate_classification_weight > 0.0:
            if gate_logits is None or gate_target_class is None:
                raise ValueError(
                    "gate_classification_weight requires gate logits and "
                    "target classes"
                )
            if gate_logits.ndim != 3:
                raise ValueError(
                    "Gate logits must be [batch, frames, classes], got "
                    f"{tuple(gate_logits.shape)}"
                )
            batch_size, frames, classes = gate_logits.shape
            if classes != self.gate_class_weights.numel():
                raise ValueError(
                    f"Gate logits have {classes} classes but "
                    f"{self.gate_class_weights.numel()} weights were configured"
                )
            target_class = gate_target_class.to(
                device=gate_logits.device,
                dtype=torch.long,
            ).flatten()
            if target_class.numel() != batch_size:
                raise ValueError(
                    "Gate target class must contain one label per item"
                )
            if bool(
                torch.any(target_class < 0)
                or torch.any(target_class >= classes)
            ):
                raise ValueError("Gate target class is out of range")
            expanded_target = target_class[:, None].expand(batch_size, frames)
            frame_loss = F.cross_entropy(
                gate_logits.float().reshape(-1, classes),
                expanded_target.reshape(-1),
                reduction="none",
            ).reshape(batch_size, frames)
            if gate_frame_mask is None:
                frame_mask = torch.ones_like(frame_loss)
            else:
                aligned_frames = min(frames, gate_frame_mask.shape[1])
                frame_loss = frame_loss[:, :aligned_frames]
                frame_mask = gate_frame_mask[:, :aligned_frames].to(
                    device=frame_loss.device,
                    dtype=frame_loss.dtype,
                )
            valid_per_item = frame_mask.sum(dim=1)
            if bool(torch.any(valid_per_item <= 0.0)):
                raise ValueError(
                    "Every item must contain at least one valid gate frame"
                )
            per_item_loss = (
                frame_loss * frame_mask
            ).sum(dim=1) / valid_per_item
            item_weights = self.gate_class_weights.to(
                device=target_class.device,
                dtype=per_item_loss.dtype,
            )[target_class]
            gate_classification_loss = (
                per_item_loss * item_weights
            ).sum() / item_weights.sum().clamp_min(1e-8)
        else:
            gate_classification_loss = enhanced.new_tensor(0.0)

        needs_recipe3_targets = any(
            weight > 0.0
            for weight in (
                self.gate_ordinal_weight,
                self.gate_strength_regression_weight,
                self.gate_separation_weight,
            )
        )
        if needs_recipe3_targets:
            if (
                gate_strength is None
                or gate_target_class is None
                or gate_target_strength is None
            ):
                raise ValueError(
                    "Recipe-3 gate losses require strength predictions, "
                    "target classes, and target strengths"
                )
            recipe3_target_class = gate_target_class.to(
                device=gate_strength.device,
                dtype=torch.long,
            ).flatten()
            recipe3_target_strength = gate_target_strength.to(
                device=gate_strength.device,
                dtype=torch.float32,
            ).flatten()
            recipe3_frames = gate_strength.shape[1]
            if gate_frame_mask is None:
                recipe3_mask = torch.ones(
                    gate_strength.shape[0],
                    recipe3_frames,
                    device=gate_strength.device,
                    dtype=torch.float32,
                )
            else:
                recipe3_frames = min(
                    recipe3_frames,
                    gate_frame_mask.shape[1],
                )
                recipe3_mask = gate_frame_mask[:, :recipe3_frames].to(
                    device=gate_strength.device,
                    dtype=torch.float32,
                )
            recipe3_valid = recipe3_mask.sum(dim=1).clamp_min(1.0)
            predicted_item_strength = (
                gate_strength[:, :recipe3_frames, 0].float()
                * recipe3_mask
            ).sum(dim=1) / recipe3_valid

        if self.gate_ordinal_weight > 0.0:
            if gate_ordinal_logits is None:
                raise ValueError(
                    "gate_ordinal_weight requires cumulative ordinal logits"
                )
            ordinal_frames = min(
                gate_ordinal_logits.shape[1],
                recipe3_frames,
            )
            thresholds = gate_ordinal_logits.shape[-1]
            ordinal_targets = (
                recipe3_target_class[:, None]
                > torch.arange(
                    thresholds,
                    device=gate_ordinal_logits.device,
                )[None, :]
            ).float()
            ordinal_targets = ordinal_targets[:, None, :].expand(
                -1,
                ordinal_frames,
                -1,
            )
            ordinal_frame_loss = F.binary_cross_entropy_with_logits(
                gate_ordinal_logits[:, :ordinal_frames].float(),
                ordinal_targets,
                reduction="none",
            ).mean(dim=-1)
            ordinal_mask = recipe3_mask[:, :ordinal_frames]
            ordinal_per_item = (
                ordinal_frame_loss * ordinal_mask
            ).sum(dim=1) / ordinal_mask.sum(dim=1).clamp_min(1.0)
            gate_ordinal_loss = ordinal_per_item.mean()
        else:
            gate_ordinal_loss = enhanced.new_tensor(0.0)

        if self.gate_strength_regression_weight > 0.0:
            regression_frame_loss = F.smooth_l1_loss(
                gate_strength[:, :recipe3_frames, 0].float(),
                recipe3_target_strength[:, None].expand(
                    -1,
                    recipe3_frames,
                ),
                beta=self.gate_strength_regression_beta,
                reduction="none",
            )
            gate_strength_regression_loss = (
                (
                    regression_frame_loss * recipe3_mask
                ).sum(dim=1)
                / recipe3_valid
            ).mean()
        else:
            gate_strength_regression_loss = enhanced.new_tensor(0.0)

        if self.gate_separation_weight > 0.0:
            target_difference = (
                recipe3_target_strength[:, None]
                - recipe3_target_strength[None, :]
            )
            prediction_difference = (
                predicted_item_strength[:, None]
                - predicted_item_strength[None, :]
            )
            pair_mask = torch.triu(
                torch.ones_like(target_difference, dtype=torch.bool),
                diagonal=1,
            ) & (
                target_difference.abs()
                >= self.gate_separation_minimum_target_distance
            )
            if bool(pair_mask.any()):
                directed_prediction = (
                    target_difference.sign() * prediction_difference
                )
                required_margin = (
                    self.gate_separation_scale
                    * target_difference.abs()
                )
                gate_separation_loss = F.relu(
                    required_margin - directed_prediction
                )[pair_mask].mean()
            else:
                gate_separation_loss = enhanced.new_tensor(0.0)
        else:
            gate_separation_loss = enhanced.new_tensor(0.0)

        needs_recipe5_targets = any(
            weight > 0.0
            for weight in (
                self.gate_utility_weight,
                self.gate_violation_weight,
                self.gate_feasibility_weight,
                self.gate_policy_weight,
                self.gate_metric_delta_weight,
            )
        )
        if needs_recipe5_targets:
            required = {
                "gate_logits": gate_logits,
                "gate_utility": gate_utility,
                "gate_log_violation": gate_log_violation,
                "gate_feasibility_logits": gate_feasibility_logits,
                "gate_metric_deltas": gate_metric_deltas,
                "gate_target_utility": gate_target_utility,
                "gate_target_violation": gate_target_violation,
                "gate_target_feasible": gate_target_feasible,
                "gate_target_policy": gate_target_policy,
                "gate_target_metric_deltas": gate_target_metric_deltas,
                "gate_target_metric_mask": gate_target_metric_mask,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    f"Recipe-5 losses require: {', '.join(missing)}"
                )
            recipe5_frames = min(
                gate_logits.shape[1],
                gate_utility.shape[1],
                gate_log_violation.shape[1],
                gate_feasibility_logits.shape[1],
                gate_metric_deltas.shape[1],
            )
            if gate_frame_mask is None:
                recipe5_mask = torch.ones(
                    gate_logits.shape[0],
                    recipe5_frames,
                    device=gate_logits.device,
                    dtype=torch.float32,
                )
            else:
                recipe5_frames = min(
                    recipe5_frames,
                    gate_frame_mask.shape[1],
                )
                recipe5_mask = gate_frame_mask[:, :recipe5_frames].to(
                    device=gate_logits.device,
                    dtype=torch.float32,
                )
            if self.gate_recipe5_burn_in_fraction > 0.0:
                valid_lengths = recipe5_mask.sum(dim=1)
                burn_in_frames = torch.floor(
                    valid_lengths * self.gate_recipe5_burn_in_fraction
                ).to(torch.long)
                positions = torch.arange(
                    recipe5_frames,
                    device=recipe5_mask.device,
                )[None, :]
                recipe5_mask = recipe5_mask * (
                    positions >= burn_in_frames[:, None]
                ).to(recipe5_mask.dtype)
            recipe5_valid = recipe5_mask.sum(dim=1).clamp_min(1.0)

            def masked_recipe5(frame_loss: torch.Tensor) -> torch.Tensor:
                while frame_loss.ndim > 2:
                    frame_loss = frame_loss.mean(dim=-1)
                return (
                    (frame_loss * recipe5_mask).sum(dim=1)
                    / recipe5_valid
                ).mean()

            target_utility = gate_target_utility.to(
                gate_logits.device,
                torch.float32,
            ).clamp(-self.gate_utility_clip, self.gate_utility_clip)
            utility_loss = F.smooth_l1_loss(
                gate_utility[:, :recipe5_frames].float(),
                target_utility[:, None, :].expand(
                    -1, recipe5_frames, -1
                ),
                beta=self.gate_utility_beta,
                reduction="none",
            )
            gate_utility_loss = masked_recipe5(utility_loss)

            target_log_violation = torch.log1p(
                gate_target_violation.to(
                    gate_logits.device,
                    torch.float32,
                ).clamp_min(0.0)
            ).clamp_max(self.gate_violation_log_clip)
            violation_loss = F.smooth_l1_loss(
                gate_log_violation[:, :recipe5_frames].float(),
                target_log_violation[:, None, :].expand(
                    -1, recipe5_frames, -1
                ),
                beta=self.gate_violation_beta,
                reduction="none",
            )
            gate_violation_loss = masked_recipe5(violation_loss)

            target_feasible = gate_target_feasible.to(
                gate_logits.device,
                torch.float32,
            )
            feasibility_loss = F.binary_cross_entropy_with_logits(
                gate_feasibility_logits[:, :recipe5_frames].float(),
                target_feasible[:, None, :].expand(
                    -1, recipe5_frames, -1
                ),
                reduction="none",
            )
            gate_feasibility_loss = masked_recipe5(feasibility_loss)

            target_policy = gate_target_policy.to(
                gate_logits.device,
                torch.float32,
            )
            policy_loss = -(
                target_policy[:, None, :]
                * F.log_softmax(
                    gate_logits[:, :recipe5_frames].float(),
                    dim=-1,
                )
            ).sum(dim=-1)
            gate_policy_loss = masked_recipe5(policy_loss)

            target_metric = gate_target_metric_deltas.to(
                gate_logits.device,
                torch.float32,
            ) / self.gate_metric_delta_scales.to(
                gate_logits.device
            ).view(1, 1, -1)
            target_metric = target_metric.clamp(
                -self.gate_metric_delta_clip,
                self.gate_metric_delta_clip,
            )
            metric_loss = F.smooth_l1_loss(
                gate_metric_deltas[:, :recipe5_frames].float(),
                target_metric[:, None, :, :].expand(
                    -1, recipe5_frames, -1, -1
                ),
                beta=self.gate_metric_delta_beta,
                reduction="none",
            )
            metric_mask = gate_target_metric_mask.to(
                gate_logits.device,
                torch.float32,
            )[:, None, :, :].expand(-1, recipe5_frames, -1, -1)
            metric_loss = (
                metric_loss * metric_mask
            ).sum(dim=(-1, -2)) / metric_mask.sum(
                dim=(-1, -2)
            ).clamp_min(1.0)
            gate_metric_delta_loss = masked_recipe5(metric_loss)
        else:
            gate_utility_loss = enhanced.new_tensor(0.0)
            gate_violation_loss = enhanced.new_tensor(0.0)
            gate_feasibility_loss = enhanced.new_tensor(0.0)
            gate_policy_loss = enhanced.new_tensor(0.0)
            gate_metric_delta_loss = enhanced.new_tensor(0.0)

        total = (
            self.waveform_l1_weight * waveform_l1
            + self.si_sdr_weight * si_sdr
            + self.stft_weight * stft
            + self.vq_weight * vq
            + self.mel_weight * mel
            + self.complex_stft_weight * complex_stft
            + self.noise_prediction_weight * noise_prediction_loss
            + self.noise_spectrum_weight * noise_spectrum_loss
            + self.magnitude_weight * magnitude_loss
            + self.magnitude_log_weight * magnitude_log_loss
            + self.magnitude_ratio_weight * magnitude_ratio_loss
            + self.phase_weight * phase_loss
            + self.group_delay_weight * group_delay_loss
            + self.instantaneous_frequency_weight * instantaneous_frequency_loss
            + self.phase_confidence_weight * confidence_loss
            + self.compute_weight * compute_loss
            + self.gate_identity_weight * gate_identity_loss
            + self.gate_smoothness_weight * gate_smoothness_loss
            + self.gate_supervision_weight * gate_supervision_loss
            + self.gate_classification_weight * gate_classification_loss
            + self.gate_ordinal_weight * gate_ordinal_loss
            + self.gate_strength_regression_weight
            * gate_strength_regression_loss
            + self.gate_separation_weight * gate_separation_loss
            + self.gate_utility_weight * gate_utility_loss
            + self.gate_violation_weight * gate_violation_loss
            + self.gate_feasibility_weight * gate_feasibility_loss
            + self.gate_policy_weight * gate_policy_loss
            + self.gate_metric_delta_weight * gate_metric_delta_loss
        )

        return LossOutput(
            total=total,
            waveform_l1=waveform_l1,
            si_sdr=si_sdr,
            stft=stft,
            vq=vq,
            mel=mel,
            complex_stft=complex_stft,
            noise_prediction=noise_prediction_loss,
            noise_spectrum=noise_spectrum_loss,
            magnitude=magnitude_loss,
            magnitude_log=magnitude_log_loss,
            magnitude_ratio=magnitude_ratio_loss,
            phase=phase_loss,
            group_delay=group_delay_loss,
            instantaneous_frequency=instantaneous_frequency_loss,
            phase_confidence=confidence_loss,
            compute=compute_loss,
            gate_identity=gate_identity_loss,
            gate_smoothness=gate_smoothness_loss,
            gate_supervision=gate_supervision_loss,
            gate_classification=gate_classification_loss,
            gate_ordinal=gate_ordinal_loss,
            gate_strength_regression=gate_strength_regression_loss,
            gate_separation=gate_separation_loss,
            gate_utility=gate_utility_loss,
            gate_violation=gate_violation_loss,
            gate_feasibility=gate_feasibility_loss,
            gate_policy=gate_policy_loss,
            gate_metric_delta=gate_metric_delta_loss,
        )
