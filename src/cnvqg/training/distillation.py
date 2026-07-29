from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from cnvqg.models.streaming_hybrid_v2 import StreamingHybridV2Output
from cnvqg.models.causal_aux_vq_mamba_v5 import CausalAuxVQMambaV5Output


@dataclass
class DistillationLossOutput:
    total: torch.Tensor
    waveform: torch.Tensor
    spectrum: torch.Tensor
    gain: torch.Tensor
    phase: torch.Tensor
    speech_features: torch.Tensor
    noise_features: torch.Tensor
    noise_prediction: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach().cpu())
            for name in self.__dataclass_fields__
        }


def _cosine_feature_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    frames = min(student.shape[1], teacher.shape[1])
    student = F.normalize(student[:, :frames], dim=-1)
    teacher = F.normalize(teacher[:, :frames], dim=-1)
    return (1.0 - (student * teacher).sum(dim=-1)).mean()


class HybridV2DistillationLoss(nn.Module):
    def __init__(
        self,
        student_speech_dim: int,
        teacher_speech_dim: int,
        student_noise_dim: int,
        teacher_noise_dim: int,
        waveform_weight: float = 0.30,
        spectrum_weight: float = 0.30,
        gain_weight: float = 0.10,
        phase_weight: float = 0.05,
        speech_feature_weight: float = 0.10,
        noise_feature_weight: float = 0.10,
        noise_prediction_weight: float = 0.05,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
    ) -> None:
        super().__init__()
        self.weights = {
            "waveform": float(waveform_weight),
            "spectrum": float(spectrum_weight),
            "gain": float(gain_weight),
            "phase": float(phase_weight),
            "speech_features": float(speech_feature_weight),
            "noise_features": float(noise_feature_weight),
            "noise_prediction": float(noise_prediction_weight),
        }
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.speech_adapters = nn.ModuleList(
            nn.Linear(student_speech_dim, teacher_speech_dim, bias=False)
            for _ in range(3)
        )
        self.noise_adapter = nn.Linear(
            student_noise_dim,
            teacher_noise_dim,
            bias=False,
        )

    def _spectrum(self, waveform: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(
            self.win_length,
            device=waveform.device,
            dtype=waveform.dtype,
        )
        return torch.stft(
            waveform.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=False,
            return_complex=True,
        ).abs()

    def forward(
        self,
        student: StreamingHybridV2Output,
        teacher: StreamingHybridV2Output,
        weight_scale: float = 1.0,
    ) -> DistillationLossOutput:
        teacher_wave = teacher.enhanced.detach()
        waveform = F.l1_loss(student.enhanced, teacher_wave)
        student_spectrum = self._spectrum(student.enhanced).clamp_min(1e-7)
        teacher_spectrum = self._spectrum(teacher_wave).clamp_min(1e-7)
        spectrum = F.l1_loss(torch.log(student_spectrum), torch.log(teacher_spectrum))

        target_size = student.gain.shape[-2:]
        teacher_gain = F.interpolate(
            teacher.gain.detach().unsqueeze(1),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        teacher_phase = F.interpolate(
            teacher.phase_delta.detach().unsqueeze(1),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        gain = F.l1_loss(torch.log(student.gain), torch.log(teacher_gain))
        magnitude_weight = teacher_spectrum / teacher_spectrum.mean(
            dim=(-2, -1), keepdim=True
        ).clamp_min(1e-7)
        magnitude_weight = F.interpolate(
            magnitude_weight.unsqueeze(1),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1).clamp(max=5.0)
        phase = ((student.phase_delta - teacher_phase).abs() * magnitude_weight).mean()

        speech_losses = []
        for adapter, student_feature, teacher_feature in zip(
            self.speech_adapters,
            student.speech_features,
            teacher.speech_features,
        ):
            speech_losses.append(
                _cosine_feature_loss(adapter(student_feature), teacher_feature.detach())
            )
        speech_features = torch.stack(speech_losses).mean()
        noise_features = _cosine_feature_loss(
            self.noise_adapter(student.quantized_noise),
            teacher.quantized_noise.detach(),
        )

        teacher_noise_prediction = F.interpolate(
            teacher.noise_prediction.detach(),
            size=student.noise_prediction.shape[-1],
            mode="linear",
            align_corners=False,
        )
        if teacher_noise_prediction.shape[1] != student.noise_prediction.shape[1]:
            raise ValueError("Teacher and student noise predictors use different FFT bins")
        noise_prediction = F.l1_loss(
            torch.log1p(student.noise_prediction),
            torch.log1p(teacher_noise_prediction),
        )

        components = {
            "waveform": waveform,
            "spectrum": spectrum,
            "gain": gain,
            "phase": phase,
            "speech_features": speech_features,
            "noise_features": noise_features,
            "noise_prediction": noise_prediction,
        }
        total = sum(self.weights[name] * value for name, value in components.items())
        total = total * float(weight_scale)
        return DistillationLossOutput(total=total, **components)


@dataclass
class V5DistillationLossOutput:
    total: torch.Tensor
    spectrum: torch.Tensor
    features: torch.Tensor
    magnitude_mask: torch.Tensor
    phase: torch.Tensor
    noise_vq: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {name: float(getattr(self, name).detach().cpu()) for name in self.__dataclass_fields__}


class V5DistillationLoss(nn.Module):
    """Teacher-target half of V5 distillation; clean supervision supplies 50%."""

    def __init__(self, student_channels: int, teacher_channels: int,
                 student_noise_dim: int, teacher_noise_dim: int,
                 spectrum_weight: float = 0.20, feature_weight: float = 0.15,
                 magnitude_mask_weight: float = 0.05, phase_weight: float = 0.05,
                 noise_vq_weight: float = 0.05, temperature: float = 2.0) -> None:
        super().__init__()
        self.weights = {"spectrum": spectrum_weight, "features": feature_weight,
                        "magnitude_mask": magnitude_mask_weight, "phase": phase_weight,
                        "noise_vq": noise_vq_weight}
        self.temperature = float(temperature)
        self.encoder_adapter = nn.Conv2d(student_channels, teacher_channels, 1, bias=False)
        self.mamba_adapter = nn.Conv2d(student_channels, teacher_channels, 1, bias=False)
        self.noise_adapter = nn.Linear(student_noise_dim, teacher_noise_dim, bias=False)

    @staticmethod
    def _match(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        size = (min(a.shape[-2], b.shape[-2]), min(a.shape[-1], b.shape[-1]))
        return (F.interpolate(a, size=size, mode="bilinear", align_corners=False),
                F.interpolate(b, size=size, mode="bilinear", align_corners=False))

    def forward(self, student: CausalAuxVQMambaV5Output,
                teacher: CausalAuxVQMambaV5Output,
                weight_scale: float = 1.0) -> V5DistillationLossOutput:
        sm, tm = self._match(student.estimated_magnitude.unsqueeze(1),
                             teacher.estimated_magnitude.detach().unsqueeze(1))
        spectrum = F.l1_loss(sm.clamp_min(1e-7).pow(0.3), tm.clamp_min(1e-7).pow(0.3))
        se, te = self._match(self.encoder_adapter(student.encoder_features),
                             teacher.encoder_features.detach())
        sx, tx = self._match(self.mamba_adapter(student.mamba_features),
                             teacher.mamba_features.detach())
        features = 0.5 * (F.smooth_l1_loss(se, te) + F.smooth_l1_loss(sx, tx))
        smask, tmask = self._match(student.magnitude_mask.unsqueeze(1),
                                   teacher.magnitude_mask.detach().unsqueeze(1))
        magnitude_mask = F.l1_loss(smask, tmask)
        sp, tp = self._match(student.predicted_phase.unsqueeze(1),
                             teacher.predicted_phase.detach().unsqueeze(1))
        phase = (1.0 - torch.cos(sp - tp)).mean()
        frames = min(student.continuous_noise_state.shape[1],
                     teacher.continuous_noise_state.shape[1])
        noise_features = F.smooth_l1_loss(
            self.noise_adapter(student.continuous_noise_state[:, :frames]),
            teacher.continuous_noise_state[:, :frames].detach())
        temperature = self.temperature
        student_log = student.code_posterior[:, :frames].clamp_min(1e-8).log()
        teacher_log = teacher.code_posterior[:, :frames].detach().clamp_min(1e-8).log()
        posterior = F.kl_div(
            F.log_softmax(student_log / temperature, dim=-1),
            F.softmax(teacher_log / temperature, dim=-1),
            reduction="none",
        ).sum(dim=-1).mean() * temperature**2
        noise_vq = 0.5 * (noise_features + posterior)
        values = {"spectrum": spectrum, "features": features,
                  "magnitude_mask": magnitude_mask, "phase": phase, "noise_vq": noise_vq}
        total = sum(float(self.weights[name]) * value for name, value in values.items())
        return V5DistillationLossOutput(total=total * float(weight_scale), **values)


@dataclass
class PrivilegedCausalDistillationLossOutput:
    total: torch.Tensor
    waveform: torch.Tensor
    compressed_complex: torch.Tensor
    log_magnitude: torch.Tensor
    teacher_bin_fraction: torch.Tensor
    teacher_sample_fraction: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach().cpu())
            for name in self.__dataclass_fields__
        }


class PrivilegedCausalDistillationLoss(nn.Module):
    """Copy a noncausal teacher only where it improves on the noisy input.

    Teacher and student internals are intentionally not matched: their frame
    rates, normalization, and temporal contracts differ. All targets are
    projected onto the student's causal STFT grid and confidence-gated using
    clean speech available only during training.
    """

    def __init__(
        self,
        waveform_weight: float = 0.05,
        compressed_complex_weight: float = 0.10,
        log_magnitude_weight: float = 0.10,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 320,
        compression_power: float = 0.3,
        confidence_temperature: float = 0.05,
        confidence_floor: float = 0.05,
    ) -> None:
        super().__init__()
        if not 0.0 <= confidence_floor <= 1.0:
            raise ValueError("confidence_floor must be between zero and one")
        self.waveform_weight = float(waveform_weight)
        self.compressed_complex_weight = float(compressed_complex_weight)
        self.log_magnitude_weight = float(log_magnitude_weight)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.compression_power = float(compression_power)
        self.confidence_temperature = float(confidence_temperature)
        self.confidence_floor = float(confidence_floor)
        self.register_buffer(
            "window", torch.hann_window(self.win_length, periodic=True), persistent=False
        )

    def _spectrum(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            waveform.squeeze(1).float(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(waveform),
            center=False,
            return_complex=True,
        )

    def _compressed(self, spectrum: torch.Tensor) -> torch.Tensor:
        magnitude = spectrum.abs().clamp_min(1e-7)
        return spectrum / magnitude * magnitude.pow(self.compression_power)

    def forward(
        self,
        student: CausalAuxVQMambaV5Output,
        teacher,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        weight_scale: float = 1.0,
    ) -> PrivilegedCausalDistillationLossOutput:
        student_wave = student.enhanced.float()
        teacher_wave = teacher.enhanced.detach().float()
        clean_wave = clean.float()
        noisy_wave = noisy.float()
        length = min(
            student_wave.shape[-1], teacher_wave.shape[-1],
            clean_wave.shape[-1], noisy_wave.shape[-1],
        )
        student_wave = student_wave[..., :length]
        teacher_wave = teacher_wave[..., :length]
        clean_wave = clean_wave[..., :length]
        noisy_wave = noisy_wave[..., :length]

        teacher_sample_error = (teacher_wave - clean_wave).abs()
        noisy_sample_error = (noisy_wave - clean_wave).abs()
        sample_confidence = torch.sigmoid(
            (noisy_sample_error - teacher_sample_error)
            / max(self.confidence_temperature, 1e-6)
        ).detach()
        sample_confidence = self.confidence_floor + (
            1.0 - self.confidence_floor
        ) * sample_confidence
        waveform = (
            sample_confidence
            * torch.sqrt((student_wave - teacher_wave).square() + 1e-6)
        ).mean()

        student_spectrum = self._spectrum(student_wave)
        teacher_spectrum = self._spectrum(teacher_wave)
        clean_spectrum = self._spectrum(clean_wave)
        noisy_spectrum = self._spectrum(noisy_wave)
        teacher_error = (
            teacher_spectrum.abs().clamp_min(1e-7).pow(self.compression_power)
            - clean_spectrum.abs().clamp_min(1e-7).pow(self.compression_power)
        ).abs()
        noisy_error = (
            noisy_spectrum.abs().clamp_min(1e-7).pow(self.compression_power)
            - clean_spectrum.abs().clamp_min(1e-7).pow(self.compression_power)
        ).abs()
        bin_confidence = torch.sigmoid(
            (noisy_error - teacher_error)
            / max(self.confidence_temperature, 1e-6)
        ).detach()
        bin_confidence = self.confidence_floor + (
            1.0 - self.confidence_floor
        ) * bin_confidence

        compressed_complex = (
            bin_confidence
            * (self._compressed(student_spectrum) - self._compressed(teacher_spectrum)).abs()
        ).mean()
        student_log = torch.log(student_spectrum.abs().clamp_min(1e-7))
        teacher_log = torch.log(teacher_spectrum.abs().clamp_min(1e-7))
        log_magnitude = (bin_confidence * (student_log - teacher_log).abs()).mean()
        total = (
            self.waveform_weight * waveform
            + self.compressed_complex_weight * compressed_complex
            + self.log_magnitude_weight * log_magnitude
        ) * float(weight_scale)
        return PrivilegedCausalDistillationLossOutput(
            total=total,
            waveform=waveform,
            compressed_complex=compressed_complex,
            log_magnitude=log_magnitude,
            teacher_bin_fraction=(bin_confidence > 0.5).float().mean(),
            teacher_sample_fraction=(sample_confidence > 0.5).float().mean(),
        )
