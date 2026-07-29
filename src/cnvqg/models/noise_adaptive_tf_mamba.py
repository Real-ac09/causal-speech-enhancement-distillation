from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .complex_cnvqg_model import TemporalStack
from .noise_vq import VQOutput
from .streaming_hybrid_v2 import EMANoiseVectorQuantizer


def _groups(channels: int) -> int:
    for value in (8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


@dataclass
class NoiseAdaptiveTFOutput:
    enhanced: torch.Tensor
    estimated_magnitude: torch.Tensor
    estimated_phase: torch.Tensor
    phase_confidence: torch.Tensor
    magnitude_mask: torch.Tensor
    noise_state: torch.Tensor
    code_indices: torch.Tensor
    expected_iterations: torch.Tensor
    halting_probabilities: torch.Tensor
    noise_prediction: torch.Tensor
    vq: VQOutput
    vq_gate: torch.Tensor


class TFEncoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, stride=(2, 1), padding=1),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, stride=(2, 1), padding=1),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class RecurrentDualAxisMambaCell(nn.Module):
    """One parameter-shared refinement cell applied repeatedly."""

    def __init__(
        self,
        channels: int,
        noise_dim: int,
        use_mamba: bool,
        mamba_d_state: int,
        mamba_d_conv: int,
        mamba_expand: int,
        condition_dynamics: bool,
    ) -> None:
        super().__init__()
        self.condition_dynamics = bool(condition_dynamics)
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
        )
        temporal_kwargs = dict(
            dim=channels,
            layers=1,
            use_mamba=use_mamba,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
        )
        self.time_model = TemporalStack(**temporal_kwargs)
        self.frequency_model = TemporalStack(**temporal_kwargs)
        self.time_scale = nn.Parameter(torch.tensor(0.1))
        self.frequency_scale = nn.Parameter(torch.tensor(0.1))
        # Always instantiate the adapter so baseline checkpoints remain exactly
        # load-compatible with later code-conditioned ablations.
        self.condition = nn.Linear(noise_dim, channels * 2)

    def forward(self, features: torch.Tensor, noise_state: torch.Tensor) -> torch.Tensor:
        batch, channels, bins, frames = features.shape
        residual = features + self.local(features)

        time_sequence = residual.permute(0, 2, 3, 1).reshape(batch * bins, frames, channels)
        time_sequence = self.time_model(time_sequence)
        time_features = time_sequence.view(batch, bins, frames, channels).permute(0, 3, 1, 2)
        residual = residual + torch.tanh(self.time_scale) * time_features

        frequency_sequence = residual.permute(0, 3, 2, 1).reshape(
            batch * frames, bins, channels
        )
        frequency_sequence = self.frequency_model(frequency_sequence)
        frequency_features = frequency_sequence.view(
            batch, frames, bins, channels
        ).permute(0, 3, 2, 1)
        residual = residual + torch.tanh(self.frequency_scale) * frequency_features

        if self.condition_dynamics:
            scale, shift = self.condition(noise_state).chunk(2, dim=-1)
            residual = residual * (1.0 + 0.1 * torch.tanh(scale[:, :, None, None]))
            residual = residual + 0.1 * shift[:, :, None, None]
        return residual


class NoiseAdaptiveTFMamba(nn.Module):
    """Compact magnitude/phase enhancer with optional noise-adaptive recurrence."""

    PRESETS = {
        "tiny": {"channels": 48, "noise_dim": 32, "codebook_size": 32},
        "base": {"channels": 72, "noise_dim": 48, "codebook_size": 64},
        "large": {"channels": 96, "noise_dim": 64, "codebook_size": 96},
        "xl": {"channels": 144, "noise_dim": 96, "codebook_size": 128},
    }

    def __init__(
        self,
        variant: str = "base",
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        center: bool = True,
        magnitude_power: float = 0.3,
        mask_max: float = 2.0,
        max_iterations: int = 4,
        use_mamba: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        use_noise_codebook: bool = True,
        condition_dynamics: bool = True,
        adaptive_iterations: bool = True,
        phase_confidence: bool = True,
        channels: int | None = None,
        noise_dim: int | None = None,
        codebook_size: int | None = None,
        vq_decay: float = 0.99,
        vq_residual_weight: float = 1.0,
        vq_gate_learnable: bool = False,
        vq_gate_initial: float = 0.01,
        vq_update_codebook: bool = True,
    ) -> None:
        super().__init__()
        if variant not in self.PRESETS:
            raise ValueError(f"Unknown v4 variant: {variant}")
        preset = self.PRESETS[variant]
        self.variant = variant
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.center = bool(center)
        self.magnitude_power = float(magnitude_power)
        self.mask_max = float(mask_max)
        self.max_iterations = int(max_iterations)
        self.use_noise_codebook = bool(use_noise_codebook)
        self.condition_dynamics = bool(condition_dynamics)
        self.adaptive_iterations = bool(adaptive_iterations)
        self.use_phase_confidence = bool(phase_confidence)
        self.channels = int(preset["channels"] if channels is None else channels)
        self.noise_dim = int(preset["noise_dim"] if noise_dim is None else noise_dim)
        self.codebook_size = int(
            preset["codebook_size"] if codebook_size is None else codebook_size
        )
        self.vq_residual_weight = float(vq_residual_weight)
        if not 0.0 <= self.vq_residual_weight <= 1.0:
            raise ValueError("vq_residual_weight must be between 0 and 1")
        self.vq_gate_learnable = bool(vq_gate_learnable)
        self.vq_gate_initial = float(vq_gate_initial)
        if not 0.0 < self.vq_gate_initial < 1.0:
            raise ValueError("vq_gate_initial must be strictly between 0 and 1")
        if self.vq_gate_learnable and not self.use_noise_codebook:
            raise ValueError("A learnable VQ gate requires use_noise_codebook=true")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")

        self.encoder = TFEncoder(self.channels)
        self.noise_encoder = nn.Sequential(
            nn.Linear(self.channels, self.noise_dim),
            nn.SiLU(),
        )
        self.noise_vq = EMANoiseVectorQuantizer(
            self.codebook_size,
            self.noise_dim,
            decay=vq_decay,
            commitment_weight=0.25,
        )
        self.noise_vq.update_codebook = bool(vq_update_codebook)
        if self.vq_gate_learnable:
            initial_logit = math.log(self.vq_gate_initial / (1.0 - self.vq_gate_initial))
            self.vq_gate_logits = nn.Parameter(
                torch.full((self.noise_dim,), initial_logit)
            )
        else:
            self.register_parameter("vq_gate_logits", None)
        self.cell = RecurrentDualAxisMambaCell(
            self.channels,
            self.noise_dim,
            use_mamba,
            mamba_d_state,
            mamba_d_conv,
            mamba_expand,
            condition_dynamics,
        )
        self.halting_head = nn.Linear(self.channels + self.noise_dim, 1)
        output_channels = 4 if phase_confidence else 3
        self.decoder = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 3, padding=1),
            nn.GroupNorm(_groups(self.channels), self.channels),
            nn.SiLU(),
            nn.Conv2d(self.channels, output_channels, 1),
        )
        self.noise_predictor = nn.Sequential(
            nn.Linear(self.noise_dim, self.noise_dim),
            nn.SiLU(),
            nn.Linear(self.noise_dim, self.n_fft // 2 + 1),
            nn.Softplus(),
        )

    @property
    def temporal(self) -> TemporalStack:
        return self.cell.time_model

    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(
            self.win_length, device=waveform.device, dtype=waveform.dtype
        )
        return torch.stft(
            waveform.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            return_complex=True,
        )

    def _empty_vq(self, noise: torch.Tensor) -> VQOutput:
        zero = noise.new_tensor(0.0)
        return VQOutput(
            quantized=noise,
            indices=torch.full(
                noise.shape[:2], -1, dtype=torch.long, device=noise.device
            ),
            loss=zero,
            commitment_loss=zero,
            codebook_loss=zero,
            perplexity=zero,
        )

    def _vq_gate(self, reference: torch.Tensor) -> torch.Tensor:
        if not self.use_noise_codebook:
            return reference.new_zeros(self.noise_dim)
        if self.vq_gate_logits is not None:
            return torch.sigmoid(self.vq_gate_logits).to(reference.dtype)
        return reference.new_full((self.noise_dim,), self.vq_residual_weight)

    def _blend_vq(
        self,
        continuous: torch.Tensor,
        quantized: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.vq_gate_logits is None and self.vq_residual_weight == 1.0:
            return quantized, self._vq_gate(continuous)
        gate = self._vq_gate(continuous)
        view_shape = (1,) * (continuous.ndim - 1) + (self.noise_dim,)
        blended = continuous + gate.view(view_shape) * (quantized - continuous)
        return blended, gate

    def forward(self, noisy: torch.Tensor) -> NoiseAdaptiveTFOutput:
        if noisy.ndim != 3 or noisy.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, T], got {tuple(noisy.shape)}")
        length = noisy.shape[-1]
        spectrum = self._stft(noisy.float())
        magnitude = spectrum.abs().clamp_min(1e-7)
        phase = torch.angle(spectrum)
        compressed = magnitude.pow(self.magnitude_power)
        inputs = torch.stack((compressed, torch.cos(phase), torch.sin(phase)), dim=1)
        features = self.encoder(inputs)

        continuous_noise = self.noise_encoder(features.mean(dim=(-2, -1))).unsqueeze(1)
        if self.use_noise_codebook:
            vq = self.noise_vq(continuous_noise)
            gated_noise, vq_gate = self._blend_vq(continuous_noise, vq.quantized)
        else:
            vq = self._empty_vq(continuous_noise)
            gated_noise = continuous_noise
            vq_gate = self._vq_gate(continuous_noise)
        noise_state = gated_noise.squeeze(1)

        remaining = features.new_ones(features.shape[0])
        accumulated = torch.zeros_like(features)
        halt_probabilities = []
        expected_iterations = features.new_zeros(features.shape[0])
        current = features
        for index in range(self.max_iterations):
            current = self.cell(current, noise_state)
            if self.adaptive_iterations and index < self.max_iterations - 1:
                pooled = current.mean(dim=(-2, -1))
                halt = torch.sigmoid(
                    self.halting_head(torch.cat((pooled, noise_state), dim=-1)).squeeze(-1)
                )
                probability = remaining * halt
                remaining = remaining * (1.0 - halt)
            elif index == self.max_iterations - 1:
                probability = remaining
                remaining = torch.zeros_like(remaining)
            else:
                probability = torch.zeros_like(remaining)
            halt_probabilities.append(probability)
            accumulated = accumulated + probability[:, None, None, None] * current
            expected_iterations = expected_iterations + probability * float(index + 1)

        decoded = self.decoder(accumulated)
        decoded = F.interpolate(
            decoded,
            size=(magnitude.shape[-2], magnitude.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        magnitude_mask = self.mask_max * torch.sigmoid(decoded[:, 0])
        estimated_magnitude = magnitude * magnitude_mask
        phase_residual = torch.atan2(decoded[:, 1], decoded[:, 2] + 1e-8)
        if self.use_phase_confidence:
            confidence = torch.sigmoid(decoded[:, 3])
        else:
            confidence = torch.ones_like(phase_residual)
        estimated_phase = phase + confidence * phase_residual
        enhanced_spectrum = torch.polar(estimated_magnitude, estimated_phase)
        window = torch.hann_window(
            self.win_length, device=noisy.device, dtype=enhanced_spectrum.real.dtype
        )
        enhanced = torch.istft(
            enhanced_spectrum,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            length=length,
        ).unsqueeze(1)
        noise_prediction = self.noise_predictor(noise_state).unsqueeze(-1)
        return NoiseAdaptiveTFOutput(
            enhanced=enhanced,
            estimated_magnitude=estimated_magnitude,
            estimated_phase=estimated_phase,
            phase_confidence=confidence,
            magnitude_mask=magnitude_mask,
            noise_state=noise_state,
            code_indices=vq.indices,
            expected_iterations=expected_iterations,
            halting_probabilities=torch.stack(halt_probabilities, dim=-1),
            noise_prediction=noise_prediction,
            vq=vq,
            vq_gate=vq_gate,
        )


__all__ = ["NoiseAdaptiveTFMamba", "NoiseAdaptiveTFOutput"]
