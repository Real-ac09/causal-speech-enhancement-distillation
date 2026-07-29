from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .causal_aux_vq_mamba_v5 import AuxiliaryVQOutput, CausalAuxVQMambaV5Output
from .causal_aux_vq_mamba_v51 import (
    CausalAuxVQMambaV51,
    CausalBranchRefiner,
    FrameGroupNorm,
    _match_frequency,
    _norm_act,
)
from .complex_cnvqg_model import TemporalStack
from .streaming_hybrid_v2 import CausalConv2d


class OneStageCausalEncoderV52(nn.Module):
    """Retain a full-resolution skip and reduce frequency exactly once."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        half = channels // 2
        self.stem = nn.Sequential(
            CausalConv2d(3, half, (3, 3)),
            _norm_act(half),
            CausalConv2d(half, half, (3, 3), groups=half),
            nn.Conv2d(half, half, 1),
            _norm_act(half),
        )
        self.down = nn.Sequential(
            nn.Conv2d(half, channels, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
            _norm_act(channels),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        full = self.stem(features)
        return self.down(full), full


class EfficientCausalDualAxisBlockV52(nn.Module):
    """Untied dual-axis block with projected Mamba dimensions.

    Frequency resolution is intentionally higher than V5.1. Projecting each
    axis before Mamba prevents that correction from doubling quadratic channel
    mixing and makes multiple independent blocks affordable.
    """

    def __init__(
        self,
        channels: int,
        axis_dim: int,
        noise_dim: int,
        use_mamba: bool,
        d_state: int,
        d_conv: int,
        expand: int,
    ) -> None:
        super().__init__()
        self.local = nn.Sequential(
            CausalConv2d(channels, channels, (3, 3), groups=channels),
            nn.Conv2d(channels, channels, 1),
            FrameGroupNorm(channels),
            nn.SiLU(),
        )
        stack = dict(
            dim=axis_dim,
            layers=1,
            use_mamba=use_mamba,
            mamba_d_state=d_state,
            mamba_d_conv=d_conv,
            mamba_expand=expand,
        )
        self.time_in = nn.Linear(channels, axis_dim)
        self.time_mamba = TemporalStack(**stack)
        self.time_out = nn.Linear(axis_dim, channels)
        self.frequency_in = nn.Linear(channels, axis_dim)
        self.frequency_mamba = TemporalStack(**stack)
        self.frequency_out = nn.Linear(axis_dim, channels)
        self.condition = nn.Linear(noise_dim, channels * 2)
        self.time_scale = nn.Parameter(torch.tensor(0.1))
        self.frequency_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, features: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        batch, channels, bins, frames = features.shape
        current = features + self.local(features)

        time_sequence = current.permute(0, 2, 3, 1).reshape(batch * bins, frames, channels)
        time_sequence = self.time_out(self.time_mamba(self.time_in(time_sequence)))
        time_features = time_sequence.view(batch, bins, frames, channels).permute(0, 3, 1, 2)
        current = current + torch.tanh(self.time_scale) * time_features

        frequency_sequence = current.permute(0, 3, 2, 1).reshape(
            batch * frames, bins, channels
        )
        projected = self.frequency_in(frequency_sequence)
        forward = self.frequency_mamba(projected)
        backward = self.frequency_mamba(projected.flip(1)).flip(1)
        frequency = self.frequency_out(0.5 * (forward + backward))
        frequency = frequency.view(batch, frames, bins, channels).permute(0, 3, 2, 1)
        current = current + torch.tanh(self.frequency_scale) * frequency

        scale, shift = self.condition(noise).transpose(1, 2).chunk(2, dim=1)
        return current * (1.0 + 0.1 * torch.tanh(scale[:, :, None])) + 0.1 * shift[:, :, None]


class OneStageDetailDecoderV52(nn.Module):
    """One-stage decoder with an explicit full-resolution detail route."""

    def __init__(
        self,
        channels: int,
        magnitude_mode: Literal["bounded_mask", "log_ratio", "compressed_residual"],
        magnitude_power: float,
    ) -> None:
        super().__init__()
        if magnitude_mode not in {"bounded_mask", "log_ratio", "compressed_residual"}:
            raise ValueError(f"Unknown V5.2 magnitude mode: {magnitude_mode}")
        self.magnitude_mode = magnitude_mode
        self.magnitude_power = float(magnitude_power)
        half = channels // 2
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                channels, half, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)
            ),
            _norm_act(half),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, half, 1),
            _norm_act(half),
            CausalConv2d(half, half, (3, 3), groups=half),
            nn.Conv2d(half, half, 1),
            _norm_act(half),
        )
        self.detail = nn.Sequential(
            CausalConv2d(half, half, (3, 3), groups=half),
            nn.Conv2d(half, half, 1),
            _norm_act(half),
        )
        self.detail_scale = nn.Parameter(torch.tensor(0.1))
        self.magnitude_branch = CausalBranchRefiner(half)
        self.phase_branch = CausalBranchRefiner(half)
        self.magnitude_head = nn.Conv2d(half, 1, 1)
        self.phase_head = nn.Conv2d(half, 3, 1)
        self.magnitude_to_phase = nn.Conv2d(half, half, 1, bias=False)
        self.phase_cross_scale = nn.Parameter(torch.tensor(0.0))
        self.mask_scale_logit = nn.Parameter(torch.tensor(0.0))
        self.log_ratio_limit = nn.Parameter(torch.tensor(math.log(4.0)))
        nn.init.zeros_(self.magnitude_head.weight)
        nn.init.zeros_(self.magnitude_head.bias)
        nn.init.zeros_(self.phase_head.weight)
        nn.init.zeros_(self.phase_head.bias)
        with torch.no_grad():
            self.phase_head.bias[1] = 1.0

    def forward(
        self,
        latent: torch.Tensor,
        full_skip: torch.Tensor,
        noisy_magnitude: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        decoded = _match_frequency(self.up(latent), full_skip)
        decoded = self.fuse(torch.cat((decoded, full_skip), dim=1))
        decoded = decoded + torch.tanh(self.detail_scale) * self.detail(full_skip)

        magnitude_features = self.magnitude_branch(decoded)
        control = self.magnitude_head(magnitude_features).squeeze(1)
        if self.magnitude_mode == "bounded_mask":
            maximum = 1.0 + 2.0 * torch.sigmoid(self.mask_scale_logit)
            identity_bias = torch.log(1.0 / (maximum - 1.0))
            mask = maximum * torch.sigmoid(control + identity_bias)
            estimated_magnitude = noisy_magnitude * mask
        elif self.magnitude_mode == "log_ratio":
            limit = self.log_ratio_limit.abs().clamp(0.1, math.log(10.0))
            mask = torch.exp(limit * torch.tanh(control))
            estimated_magnitude = noisy_magnitude * mask
        else:
            compressed = noisy_magnitude.pow(self.magnitude_power)
            ratio = (1.0 + torch.tanh(control)).clamp_min(1e-4)
            estimated_magnitude = (compressed * ratio).clamp_min(1e-7).pow(
                1.0 / self.magnitude_power
            )
            mask = estimated_magnitude / noisy_magnitude.clamp_min(1e-7)

        phase_features = self.phase_branch(decoded)
        phase_features = phase_features + torch.tanh(
            self.phase_cross_scale
        ) * self.magnitude_to_phase(magnitude_features)
        phase_output = self.phase_head(phase_features)
        phase_vector = F.normalize(phase_output[:, :2], dim=1, eps=1e-7)
        phase_confidence = torch.sigmoid(phase_output[:, 2])
        return estimated_magnitude, mask, phase_vector, phase_confidence


class CausalAuxVQMambaV52(CausalAuxVQMambaV51):
    """Parameter-efficient causal V5.2 with one frequency reduction."""

    PRESETS = {
        "student": {"channels": 128, "axis_dim": 80, "noise_dim": 64, "blocks": 2, "passes": 2, "cap": 1_100_000},
        "teacher": {"channels": 192, "axis_dim": 112, "noise_dim": 96, "blocks": 2, "passes": 2, "cap": 2_700_000},
    }

    def __init__(
        self,
        variant: Literal["student", "teacher"] = "teacher",
        magnitude_mode: Literal[
            "bounded_mask", "log_ratio", "compressed_residual"
        ] = "log_ratio",
        axis_dim: int | None = None,
        num_blocks: int | None = None,
        **kwargs,
    ) -> None:
        preset = self.PRESETS[variant]
        channels = int(kwargs.pop("channels", preset["channels"]))
        noise_dim = int(kwargs.pop("noise_dim", preset["noise_dim"]))
        axis_dim = int(axis_dim or preset["axis_dim"])
        num_blocks = int(num_blocks or preset["blocks"])
        enforce = bool(kwargs.pop("enforce_parameter_cap", True))
        use_mamba = bool(kwargs.get("use_mamba", True))
        d_state = int(kwargs.get("mamba_d_state", 16))
        d_conv = int(kwargs.get("mamba_d_conv", 4))
        expand = int(kwargs.get("mamba_expand", 2))
        super().__init__(
            variant=variant,
            magnitude_mode=magnitude_mode,
            channels=channels,
            noise_dim=noise_dim,
            refinement_passes=2,
            enforce_parameter_cap=False,
            **kwargs,
        )
        if num_blocks < 1:
            raise ValueError("V5.2 num_blocks must be positive")
        self.axis_dim = axis_dim
        self.num_blocks = num_blocks
        self.parameter_cap = int(preset["cap"])
        self.encoder = OneStageCausalEncoderV52(channels)
        self.blocks = nn.ModuleList(
            EfficientCausalDualAxisBlockV52(
                channels, axis_dim, noise_dim, use_mamba, d_state, d_conv, expand
            )
            for _ in range(num_blocks)
        )
        self.decoder = OneStageDetailDecoderV52(
            channels, magnitude_mode, self.magnitude_power
        )
        self.temporal = self.blocks[0].time_mamba
        count = self.parameter_count()
        if enforce and count > self.parameter_cap:
            raise ValueError(
                f"V5.2 {variant} has {count:,} parameters, exceeding cap "
                f"{self.parameter_cap:,}."
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
        noisy_phase = torch.angle(spectrum)
        inputs = torch.stack(
            (magnitude.pow(self.magnitude_power), torch.cos(noisy_phase), torch.sin(noisy_phase)),
            dim=1,
        )
        features, full_skip = self.encoder(inputs)
        continuous_noise = self._rolling_noise(features)
        raw_vq = self.noise_vq(continuous_noise)
        probabilities = F.one_hot(raw_vq.indices, self.noise_vq.codebook_size).float().mean((0, 1))
        usage_kl = (
            probabilities
            * (probabilities.clamp_min(1e-8).log() + math.log(self.noise_vq.codebook_size))
        ).sum()
        noise_prediction = self.noise_predictor(raw_vq.quantized).transpose(1, 2)
        flat_noise = continuous_noise.reshape(-1, self.noise_dim)
        codebook = self.noise_vq.codebook.to(flat_noise)
        distances = (
            flat_noise.square().sum(1, keepdim=True)
            - 2.0 * flat_noise @ codebook.transpose(0, 1)
            + codebook.square().sum(1)
        )
        code_posterior = torch.softmax(-distances, dim=-1).view(
            *continuous_noise.shape[:2], self.noise_vq.codebook_size
        )

        current = features
        for block in self.blocks:
            current = block(current, continuous_noise)

        estimated_magnitude, magnitude_mask, phase_vector, phase_confidence = self.decoder(
            current, full_skip, magnitude
        )
        magnitude_scale = float(self.magnitude_residual_scale)
        if magnitude_scale != 1.0:
            log_ratio = torch.log(estimated_magnitude / magnitude).clamp(-20.0, 20.0)
            estimated_magnitude = magnitude * torch.exp(magnitude_scale * log_ratio)
            magnitude_mask = estimated_magnitude / magnitude
        phase_residual = torch.atan2(phase_vector[:, 0], phase_vector[:, 1])
        phase_candidate = noisy_phase + phase_residual
        predicted_phase = noisy_phase + (
            float(self.phase_residual_scale) * phase_confidence * phase_residual
        )
        enhanced = self._synthesis(
            torch.polar(estimated_magnitude, predicted_phase), original_length
        ).unsqueeze(1)
        zero = enhanced.new_tensor(0.0)
        vq = AuxiliaryVQOutput(
            loss=raw_vq.loss + self.vq_usage_weight * usage_kl,
            commitment_loss=raw_vq.commitment_loss,
            usage_kl=usage_kl,
            reconstruction_loss=zero,
            perplexity=raw_vq.perplexity,
            active_fraction=raw_vq.active_fraction,
            dead_fraction=raw_vq.dead_fraction,
        )
        return CausalAuxVQMambaV5Output(
            enhanced=enhanced,
            estimated_magnitude=estimated_magnitude,
            predicted_phase=predicted_phase,
            phase_candidate=phase_candidate,
            magnitude_mask=magnitude_mask,
            continuous_noise_state=continuous_noise,
            code_indices=raw_vq.indices,
            code_posterior=code_posterior,
            code_perplexity=raw_vq.perplexity,
            vq_adapter_strength=zero,
            noise_prediction=noise_prediction,
            encoder_features=features,
            mamba_features=current,
            vq=vq,
            phase_confidence=phase_confidence,
        )


__all__ = [
    "CausalAuxVQMambaV52",
    "EfficientCausalDualAxisBlockV52",
    "OneStageCausalEncoderV52",
    "OneStageDetailDecoderV52",
]
