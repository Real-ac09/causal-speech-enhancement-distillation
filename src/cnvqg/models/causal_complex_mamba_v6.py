from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .causal_aux_vq_mamba_v5 import (
    AuxiliaryVQOutput,
    CausalAuxVQMambaV5,
    CausalAuxVQMambaV5Output,
)
from .causal_aux_vq_mamba_v51 import FrameGroupNorm, _match_frequency
from .complex_cnvqg_model import TemporalStack
from .streaming_hybrid_v2 import CausalConv2d, EMANoiseVectorQuantizer


def _norm_act(channels: int) -> nn.Sequential:
    return nn.Sequential(FrameGroupNorm(channels), nn.SiLU())


class CausalResidualUnitV6(nn.Module):
    """Cheap local time-frequency processing with no future-time statistics."""

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CausalConv2d(
                channels,
                channels,
                kernel_size=(5, 3),
                dilation=(1, dilation),
                groups=channels,
            ),
            nn.Conv2d(channels, channels * 2, 1),
            nn.GLU(dim=1),
            FrameGroupNorm(channels),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + torch.tanh(self.scale) * self.net(features)


class CausalComplexEncoderV6(nn.Module):
    """One frequency reduction and a full-resolution reconstruction skip."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        full_channels = channels // 2
        self.stem = nn.Sequential(
            CausalConv2d(3, full_channels, (5, 3)),
            _norm_act(full_channels),
            CausalResidualUnitV6(full_channels),
        )
        self.down = nn.Sequential(
            nn.Conv2d(
                full_channels,
                channels,
                kernel_size=(4, 1),
                stride=(2, 1),
                padding=(1, 0),
            ),
            _norm_act(channels),
            CausalResidualUnitV6(channels),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        full = self.stem(features)
        return self.down(full), full


class FullBandFrequencyBlockV6(nn.Module):
    """Frame-local bidirectional frequency context with a bounded residual."""

    def __init__(self, channels: int, frequency_dim: int) -> None:
        super().__init__()
        if frequency_dim <= 0 or frequency_dim % 2:
            raise ValueError("frequency_dim must be a positive even integer")
        self.frequency_in = nn.Linear(channels, frequency_dim)
        self.norm = nn.LayerNorm(frequency_dim)
        self.frequency_gru = nn.GRU(
            frequency_dim,
            frequency_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.frequency_out = nn.Linear(frequency_dim, channels)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, bins, frames = features.shape
        sequence = features.permute(0, 3, 2, 1).reshape(
            batch * frames, bins, channels
        )
        projected = self.norm(self.frequency_in(sequence))
        self.frequency_gru.flatten_parameters()
        frequency, _ = self.frequency_gru(projected)
        frequency = self.frequency_out(frequency)
        frequency = frequency.view(batch, frames, bins, channels).permute(0, 3, 2, 1)
        return features + torch.tanh(self.scale) * frequency


class CausalTemporalCoreV6(nn.Module):
    """Local TF convolutions plus one projected causal temporal Mamba stack."""

    def __init__(
        self,
        channels: int,
        temporal_dim: int,
        temporal_layers: int,
        use_mamba: bool,
        d_state: int,
        d_conv: int,
        expand: int,
        use_full_band: bool = False,
        frequency_dim: int = 144,
    ) -> None:
        super().__init__()
        self.pre = nn.Sequential(
            CausalResidualUnitV6(channels, dilation=1),
            CausalResidualUnitV6(channels, dilation=2),
        )
        self.full_band = (
            FullBandFrequencyBlockV6(channels, frequency_dim)
            if use_full_band
            else nn.Identity()
        )
        self.temporal_in = nn.Linear(channels, temporal_dim)
        self.temporal = TemporalStack(
            dim=temporal_dim,
            layers=temporal_layers,
            use_mamba=use_mamba,
            mamba_d_state=d_state,
            mamba_d_conv=d_conv,
            mamba_expand=expand,
        )
        self.temporal_out = nn.Linear(temporal_dim, channels)
        self.temporal_scale = nn.Parameter(torch.tensor(0.1))
        self.post = CausalResidualUnitV6(channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        current = self.pre(features)
        current = self.full_band(current)
        batch, channels, bins, frames = current.shape
        sequence = current.permute(0, 2, 3, 1).reshape(batch * bins, frames, channels)
        sequence = self.temporal_out(self.temporal(self.temporal_in(sequence)))
        temporal = sequence.view(batch, bins, frames, channels).permute(0, 3, 1, 2)
        current = current + torch.tanh(self.temporal_scale) * temporal
        return self.post(current)


class ComplexMaskDecoderV6(nn.Module):
    """Identity-initialised complex mask; phase is corrected only implicitly."""

    def __init__(
        self,
        channels: int,
        representation: Literal["complex_ratio", "magnitude_only", "polar_residual"],
        residual_limit: float = 1.0,
        log_magnitude_limit: float = math.log(4.0),
        phase_limit: float = math.pi / 2.0,
    ) -> None:
        super().__init__()
        if representation not in {"complex_ratio", "magnitude_only", "polar_residual"}:
            raise ValueError(f"Unknown V6 mask representation: {representation}")
        if residual_limit <= 0.0:
            raise ValueError("complex_mask_residual_limit must be positive")
        full_channels = channels // 2
        self.representation = representation
        self.residual_limit = float(residual_limit)
        self.log_magnitude_limit = float(log_magnitude_limit)
        self.phase_limit = float(phase_limit)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                channels,
                full_channels,
                kernel_size=(4, 1),
                stride=(2, 1),
                padding=(1, 0),
            ),
            _norm_act(full_channels),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(full_channels * 2, full_channels, 1),
            _norm_act(full_channels),
            CausalResidualUnitV6(full_channels),
            CausalResidualUnitV6(full_channels, dilation=2),
        )
        self.mask_head = nn.Conv2d(full_channels, 2, 1)
        nn.init.zeros_(self.mask_head.weight)
        nn.init.zeros_(self.mask_head.bias)

    def forward(
        self, latent: torch.Tensor, full_skip: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decoded = _match_frequency(self.up(latent), full_skip)
        decoded = self.fuse(torch.cat((decoded, full_skip), dim=1))
        # PyTorch complex tensors do not support BF16 components. Keep the
        # network under autocast, then assemble the tiny two-channel output in
        # FP32; the cast remains differentiable.
        controls = self.mask_head(decoded).float()
        if self.representation == "complex_ratio":
            residual = self.residual_limit * torch.tanh(controls)
            mask = torch.complex(1.0 + residual[:, 0], residual[:, 1])
        else:
            magnitude = torch.exp(
                self.log_magnitude_limit * torch.tanh(controls[:, 0])
            )
            if self.representation == "magnitude_only":
                phase = torch.zeros_like(magnitude)
            else:
                phase = self.phase_limit * torch.tanh(controls[:, 1])
            mask = torch.polar(magnitude, phase)
        return mask, decoded


class CausalComplexMambaV6(CausalAuxVQMambaV5):
    """Simple causal enhancement baseline for the V6 architecture search.

    Enhancement is driven solely by continuous convolutional/Mamba features.
    Optional VQ remains a training-only auxiliary branch and cannot affect the
    enhanced spectrum.
    """

    PRESETS = {
        "student": {
            "channels": 224,
            "temporal_dim": 128,
            "noise_dim": 64,
            "cap": 1_100_000,
        },
        "teacher": {
            "channels": 304,
            "temporal_dim": 192,
            "noise_dim": 96,
            "cap": 2_700_000,
        },
    }

    def __init__(
        self,
        variant: Literal["student", "teacher"] = "teacher",
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 320,
        magnitude_power: float = 0.3,
        channels: int | None = None,
        temporal_dim: int | None = None,
        temporal_layers: int = 2,
        noise_dim: int | None = None,
        auxiliary_vq: bool = False,
        codebook_size: int = 32,
        vq_commitment_weight: float = 0.02,
        vq_usage_weight: float = 0.005,
        use_mamba: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        use_full_band: bool = False,
        frequency_dim: int = 144,
        complex_mask_residual_limit: float = 1.0,
        mask_representation: Literal[
            "complex_ratio", "magnitude_only", "polar_residual"
        ] = "complex_ratio",
        log_magnitude_limit: float = math.log(4.0),
        phase_limit: float = math.pi / 2.0,
        enforce_parameter_cap: bool = True,
    ) -> None:
        # Reuse the proven V5 causal analysis/synthesis and streaming methods,
        # but deliberately avoid constructing the V5 enhancement graph.
        nn.Module.__init__(self)
        if variant not in self.PRESETS:
            raise ValueError(f"Unknown V6 variant: {variant}")
        if n_fft < win_length or hop_length > win_length:
            raise ValueError("V6 requires n_fft >= win_length >= hop_length")
        preset = self.PRESETS[variant]
        self.variant = variant
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.magnitude_power = float(magnitude_power)
        self.channels = int(channels or preset["channels"])
        self.temporal_dim = int(temporal_dim or preset["temporal_dim"])
        self.noise_dim = int(noise_dim or preset["noise_dim"])
        self.auxiliary_vq = bool(auxiliary_vq)
        self.vq_usage_weight = float(vq_usage_weight)
        self.parameter_cap = int(preset["cap"])
        self.algorithmic_latency_samples = self.win_length
        self.phase_residual_scale = 1.0
        self.magnitude_residual_scale = 1.0

        self.register_buffer(
            "analysis_window",
            torch.hann_window(self.win_length, periodic=True),
            persistent=False,
        )
        self.encoder = CausalComplexEncoderV6(self.channels)
        self.core = CausalTemporalCoreV6(
            self.channels,
            self.temporal_dim,
            int(temporal_layers),
            use_mamba,
            int(mamba_d_state),
            int(mamba_d_conv),
            int(mamba_expand),
            bool(use_full_band),
            int(frequency_dim),
        )
        self.temporal = self.core.temporal
        self.decoder = ComplexMaskDecoderV6(
            self.channels,
            representation=mask_representation,
            residual_limit=complex_mask_residual_limit,
            log_magnitude_limit=log_magnitude_limit,
            phase_limit=phase_limit,
        )
        if self.auxiliary_vq:
            self.noise_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(self.channels, self.noise_dim),
                nn.SiLU(),
                nn.LayerNorm(self.noise_dim),
            )
            self.noise_vq: EMANoiseVectorQuantizer | None = EMANoiseVectorQuantizer(
                codebook_size=codebook_size,
                code_dim=self.noise_dim,
                update_interval=1,
                commitment_weight=vq_commitment_weight,
            )
        else:
            self.noise_encoder = None
            self.noise_vq = None

        count = self.parameter_count()
        if enforce_parameter_cap and count > self.parameter_cap:
            raise ValueError(
                f"V6 {variant} has {count:,} parameters, exceeding cap "
                f"{self.parameter_cap:,}."
            )

    def _noise_state(self, features: torch.Tensor) -> torch.Tensor:
        pooled = features.mean(dim=-2).transpose(1, 2)
        if self.noise_encoder is not None:
            return self.noise_encoder(pooled)
        # Preserve the common output interface without retaining dead
        # trainable parameters in the no-VQ baseline.
        if pooled.shape[-1] >= self.noise_dim:
            return pooled[..., : self.noise_dim]
        return F.pad(pooled, (0, self.noise_dim - pooled.shape[-1]))

    def _auxiliary_outputs(
        self, continuous_noise: torch.Tensor, frequency_bins: int
    ) -> tuple[AuxiliaryVQOutput, torch.Tensor, torch.Tensor]:
        zero = continuous_noise.new_tensor(0.0)
        if self.noise_vq is None:
            indices = torch.zeros(
                continuous_noise.shape[:2], device=continuous_noise.device, dtype=torch.long
            )
            posterior = continuous_noise.new_ones(*continuous_noise.shape[:2], 1)
            return (
                AuxiliaryVQOutput(zero, zero, zero, zero, zero, zero, zero),
                indices,
                posterior,
            )
        raw = self.noise_vq(continuous_noise)
        probabilities = F.one_hot(raw.indices, self.noise_vq.codebook_size).float().mean((0, 1))
        usage_kl = (
            probabilities
            * (probabilities.clamp_min(1e-8).log() + math.log(self.noise_vq.codebook_size))
        ).sum()
        flat = continuous_noise.reshape(-1, self.noise_dim)
        codebook = self.noise_vq.codebook.to(flat)
        distances = (
            flat.square().sum(1, keepdim=True)
            - 2.0 * flat @ codebook.transpose(0, 1)
            + codebook.square().sum(1)
        )
        posterior = torch.softmax(-distances, dim=-1).view(
            *continuous_noise.shape[:2], self.noise_vq.codebook_size
        )
        return (
            AuxiliaryVQOutput(
                raw.loss + self.vq_usage_weight * usage_kl,
                raw.commitment_loss,
                usage_kl,
                zero,
                raw.perplexity,
                raw.active_fraction,
                raw.dead_fraction,
            ),
            raw.indices,
            posterior,
        )

    def _forward_waveform(
        self, noisy: torch.Tensor, pad_end: bool
    ) -> CausalAuxVQMambaV5Output:
        if noisy.ndim != 3 or noisy.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, T], got {tuple(noisy.shape)}")
        spectrum, original_length = self._analysis(noisy.squeeze(1), pad_end=pad_end)
        if spectrum.shape[-1] == 0:
            raise ValueError("At least one complete analysis window is required")
        magnitude = spectrum.abs().clamp_min(1e-7)
        compressed = magnitude.pow(self.magnitude_power)
        unit = spectrum / magnitude
        inputs = torch.stack((compressed, unit.real, unit.imag), dim=1)
        latent, full_skip = self.encoder(inputs)
        current = self.core(latent)
        complex_mask, decoded = self.decoder(current, full_skip)

        # These evaluation scales enable cheap diagnostics without changing a
        # trained checkpoint. Scaling in polar form keeps both controls clear.
        raw_magnitude_mask = complex_mask.abs().clamp_min(1e-7)
        raw_phase_residual = torch.angle(complex_mask)
        magnitude_mask = torch.exp(
            float(self.magnitude_residual_scale) * raw_magnitude_mask.log()
        )
        phase_residual = float(self.phase_residual_scale) * raw_phase_residual
        estimated_magnitude = magnitude * magnitude_mask
        predicted_phase = torch.angle(spectrum) + phase_residual
        enhanced_spectrum = torch.polar(estimated_magnitude, predicted_phase)
        enhanced = self._synthesis(enhanced_spectrum, original_length).unsqueeze(1)

        continuous_noise = self._noise_state(current)
        vq, code_indices, code_posterior = self._auxiliary_outputs(
            continuous_noise, magnitude.shape[-2]
        )
        zero = enhanced.new_tensor(0.0)
        noise_prediction = enhanced.new_zeros(
            enhanced.shape[0], magnitude.shape[-2], continuous_noise.shape[1]
        )
        return CausalAuxVQMambaV5Output(
            enhanced=enhanced,
            estimated_magnitude=estimated_magnitude,
            predicted_phase=predicted_phase,
            phase_candidate=predicted_phase,
            magnitude_mask=magnitude_mask,
            continuous_noise_state=continuous_noise,
            code_indices=code_indices,
            code_posterior=code_posterior,
            code_perplexity=vq.perplexity,
            vq_adapter_strength=zero,
            noise_prediction=noise_prediction,
            encoder_features=latent,
            mamba_features=current,
            vq=vq,
            phase_confidence=None,
        )


__all__ = [
    "CausalComplexEncoderV6",
    "CausalComplexMambaV6",
    "CausalResidualUnitV6",
    "CausalTemporalCoreV6",
    "ComplexMaskDecoderV6",
    "FullBandFrequencyBlockV6",
]
