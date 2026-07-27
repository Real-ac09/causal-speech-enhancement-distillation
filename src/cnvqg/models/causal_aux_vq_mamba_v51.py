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
from .complex_cnvqg_model import TemporalStack
from .streaming_hybrid_v2 import CausalConv2d


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class FrameGroupNorm(nn.Module):
    """GroupNorm over channels/frequency independently for every time frame.

    Ordinary GroupNorm on ``[B, C, F, T]`` includes the complete time axis in
    its statistics and therefore leaks future frames. Frequency context within
    the current frame is permitted by the V5 streaming contract.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, bins, frames = features.shape
        current = features.permute(0, 3, 1, 2).reshape(batch * frames, channels, bins)
        current = self.norm(current)
        return current.view(batch, frames, channels, bins).permute(0, 2, 3, 1)


def _norm_act(channels: int) -> nn.Sequential:
    return nn.Sequential(FrameGroupNorm(channels), nn.SiLU())


class TwoStageCausalEncoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        half = channels // 2
        self.stem = nn.Sequential(CausalConv2d(3, half, (3, 3)), _norm_act(half))
        self.down_one = nn.Sequential(
            nn.Conv2d(half, channels, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
            _norm_act(channels),
        )
        self.down_two = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
            _norm_act(channels),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full = self.stem(features)
        half = self.down_one(full)
        return self.down_two(half), half, full


class CausalDualAxisCellV51(nn.Module):
    def __init__(
        self,
        channels: int,
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
        kwargs = dict(
            dim=channels,
            layers=1,
            use_mamba=use_mamba,
            mamba_d_state=d_state,
            mamba_d_conv=d_conv,
            mamba_expand=expand,
        )
        self.time_mamba = TemporalStack(**kwargs)
        self.frequency_mamba = TemporalStack(**kwargs)
        self.condition = nn.Linear(noise_dim, channels * 2)
        self.time_scale = nn.Parameter(torch.tensor(0.1))
        self.frequency_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, features: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        batch, channels, bins, frames = features.shape
        current = features + self.local(features)
        sequence = current.permute(0, 2, 3, 1).reshape(batch * bins, frames, channels)
        temporal = self.time_mamba(sequence)
        temporal = temporal.view(batch, bins, frames, channels).permute(0, 3, 1, 2)
        current = current + torch.tanh(self.time_scale) * temporal

        sequence = current.permute(0, 3, 2, 1).reshape(batch * frames, bins, channels)
        forward = self.frequency_mamba(sequence)
        backward = self.frequency_mamba(sequence.flip(1)).flip(1)
        frequency = 0.5 * (forward + backward)
        frequency = frequency.view(batch, frames, bins, channels).permute(0, 3, 2, 1)
        current = current + torch.tanh(self.frequency_scale) * frequency

        condition = self.condition(noise).transpose(1, 2)
        scale, shift = condition.chunk(2, dim=1)
        return current * (1.0 + 0.1 * torch.tanh(scale[:, :, None])) + 0.1 * shift[:, :, None]


def _match_frequency(features: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if features.shape[-2:] == reference.shape[-2:]:
        return features
    if features.shape[-1] != reference.shape[-1]:
        raise RuntimeError("V5.1 decoder attempted to resize the causal time axis")
    return F.interpolate(features, size=reference.shape[-2:], mode="nearest")


class CausalBranchRefiner(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CausalConv2d(channels, channels, (3, 3), groups=channels),
            nn.Conv2d(channels, channels, 1),
            FrameGroupNorm(channels),
            nn.SiLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.net(features)


class LearnedCausalDualHeadDecoder(nn.Module):
    def __init__(
        self,
        channels: int,
        magnitude_mode: Literal["bounded_mask", "log_ratio", "compressed_residual"],
        magnitude_power: float,
    ) -> None:
        super().__init__()
        if magnitude_mode not in {"bounded_mask", "log_ratio", "compressed_residual"}:
            raise ValueError(f"Unknown V5.1 magnitude mode: {magnitude_mode}")
        self.magnitude_mode = magnitude_mode
        self.magnitude_power = float(magnitude_power)
        half = channels // 2
        self.up_one = nn.Sequential(
            nn.ConvTranspose2d(
                channels, channels, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)
            ),
            _norm_act(channels),
        )
        self.fuse_one = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            _norm_act(channels),
            CausalConv2d(channels, channels, (3, 3), groups=channels),
            nn.Conv2d(channels, channels, 1),
            _norm_act(channels),
        )
        self.up_two = nn.Sequential(
            nn.ConvTranspose2d(
                channels, half, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)
            ),
            _norm_act(half),
        )
        self.fuse_two = nn.Sequential(
            nn.Conv2d(channels, half, 1),
            _norm_act(half),
            CausalConv2d(half, half, (3, 3), groups=half),
            nn.Conv2d(half, half, 1),
            _norm_act(half),
        )
        self.magnitude_branch = CausalBranchRefiner(half)
        self.phase_branch = CausalBranchRefiner(half)
        self.magnitude_head = nn.Conv2d(half, 1, 1)
        self.phase_head = nn.Conv2d(half, 3, 1)
        self.magnitude_to_phase = nn.Conv2d(half, half, 1, bias=False)
        self.phase_cross_scale = nn.Parameter(torch.tensor(0.05))
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
        half_skip: torch.Tensor,
        full_skip: torch.Tensor,
        noisy_magnitude: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        current = _match_frequency(self.up_one(latent), half_skip)
        current = self.fuse_one(torch.cat((current, half_skip), dim=1))
        current = _match_frequency(self.up_two(current), full_skip)
        decoded = self.fuse_two(torch.cat((current, full_skip), dim=1))

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
            compressed_ratio = (1.0 + torch.tanh(control)).clamp_min(1e-4)
            estimated_magnitude = (compressed * compressed_ratio).clamp_min(1e-7).pow(
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


class CausalAuxVQMambaV51(CausalAuxVQMambaV5):
    """Causal V5.1 tournament model with learned multi-resolution reconstruction."""

    PRESETS = {
        "student": {"channels": 128, "noise_dim": 64, "passes": 2, "cap": 1_100_000},
        "teacher": {"channels": 192, "noise_dim": 96, "passes": 2, "cap": 2_700_000},
    }

    def __init__(
        self,
        variant: Literal["student", "teacher"] = "teacher",
        magnitude_mode: Literal[
            "bounded_mask", "log_ratio", "compressed_residual"
        ] = "log_ratio",
        vq_mode: Literal["disabled", "train_only", "bounded_adapter"] = "train_only",
        phase_residual_scale: float = 1.0,
        **kwargs,
    ) -> None:
        if vq_mode not in {"disabled", "train_only", "bounded_adapter"}:
            raise ValueError(f"Unknown V5.1 VQ mode: {vq_mode}")
        if not 0.0 <= phase_residual_scale <= 1.0:
            raise ValueError("phase_residual_scale must be between zero and one")
        preset = self.PRESETS[variant]
        channels = int(kwargs.pop("channels", preset["channels"]))
        noise_dim = int(kwargs.pop("noise_dim", preset["noise_dim"]))
        passes = int(kwargs.pop("refinement_passes", preset["passes"]))
        enforce = bool(kwargs.pop("enforce_parameter_cap", True))
        super().__init__(
            variant=variant,
            channels=channels,
            noise_dim=noise_dim,
            refinement_passes=passes,
            enforce_parameter_cap=False,
            vq_mode="train_only" if vq_mode == "disabled" else vq_mode,
            **kwargs,
        )
        self.vq_mode = vq_mode
        self.parameter_cap = int(preset["cap"])
        self.magnitude_mode = magnitude_mode
        # Evaluation-time ablations may attenuate the predicted phase update.
        # This is deliberately not a parameter or persistent buffer, so old
        # checkpoints remain bit-for-bit loadable and training defaults to the
        # original full-strength phase path.
        self.phase_residual_scale = float(phase_residual_scale)
        # Likewise, one applies the complete predicted log-magnitude ratio and
        # zero preserves noisy magnitude. Log-domain interpolation works
        # consistently for mask, ratio, and compressed-residual decoders.
        self.magnitude_residual_scale = 1.0
        use_mamba = bool(kwargs.get("use_mamba", True))
        d_state = int(kwargs.get("mamba_d_state", 16))
        d_conv = int(kwargs.get("mamba_d_conv", 4))
        expand = int(kwargs.get("mamba_expand", 2))
        self.encoder = TwoStageCausalEncoder(self.channels)
        self.cell = CausalDualAxisCellV51(
            self.channels, self.noise_dim, use_mamba, d_state, d_conv, expand
        )
        self.decoder = LearnedCausalDualHeadDecoder(
            self.channels, magnitude_mode, self.magnitude_power
        )
        self.temporal = self.cell.time_mamba
        count = self.parameter_count()
        if enforce and count > self.parameter_cap:
            raise ValueError(
                f"V5.1 {variant} has {count:,} parameters, exceeding {self.parameter_cap:,}"
            )

    def set_vq_mode(
        self, mode: Literal["disabled", "train_only", "bounded_adapter"]
    ) -> None:
        if mode not in {"disabled", "train_only", "bounded_adapter"}:
            raise ValueError(f"Unknown V5.1 VQ mode: {mode}")
        self.vq_mode = mode

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
        features, half_skip, full_skip = self.encoder(inputs)
        continuous_noise = self._rolling_noise(features)
        zero = continuous_noise.new_tensor(0.0)
        if self.vq_mode == "disabled":
            raw_vq = None
            usage_kl = zero
            code_indices = torch.full(
                continuous_noise.shape[:2],
                -1,
                dtype=torch.long,
                device=continuous_noise.device,
            )
            code_posterior = continuous_noise.new_zeros(
                *continuous_noise.shape[:2], self.noise_vq.codebook_size
            )
            noise_prediction = self.noise_predictor(continuous_noise).transpose(1, 2)
        else:
            raw_vq = self.noise_vq(continuous_noise)
            probabilities = F.one_hot(
                raw_vq.indices, self.noise_vq.codebook_size
            ).float().mean((0, 1))
            usage_kl = (
                probabilities
                * (
                    probabilities.clamp_min(1e-8).log()
                    + math.log(self.noise_vq.codebook_size)
                )
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
            code_indices = raw_vq.indices

        adapter_strength = continuous_noise.new_tensor(0.0)
        adapter_condition = None
        if self.vq_mode == "bounded_adapter" and raw_vq is not None:
            adapter_condition, adapter_strength = self.adapter(raw_vq.quantized, self.training)
        current = features
        for _ in range(self.refinement_passes):
            current = self.cell(current, continuous_noise)
            if adapter_condition is not None:
                scale, shift = adapter_condition.transpose(1, 2).chunk(2, dim=1)
                current = current * (1.0 + scale[:, :, None]) + shift[:, :, None]

        estimated_magnitude, magnitude_mask, phase_vector, phase_confidence = self.decoder(
            current, half_skip, full_skip, magnitude
        )
        magnitude_scale = float(self.magnitude_residual_scale)
        if magnitude_scale != 1.0:
            log_ratio = (
                estimated_magnitude.clamp_min(1e-7).log()
                - magnitude.clamp_min(1e-7).log()
            )
            estimated_magnitude = magnitude * torch.exp(magnitude_scale * log_ratio)
            magnitude_mask = estimated_magnitude / magnitude.clamp_min(1e-7)
        phase_residual = torch.atan2(phase_vector[:, 0], phase_vector[:, 1])
        predicted_phase = noisy_phase + (
            float(self.phase_residual_scale) * phase_confidence * phase_residual
        )
        enhanced_spectrum = torch.polar(estimated_magnitude, predicted_phase)
        enhanced = self._synthesis(enhanced_spectrum, original_length).unsqueeze(1)
        zero = enhanced.new_tensor(0.0)
        if raw_vq is None:
            vq = AuxiliaryVQOutput(zero, zero, zero, zero, zero, zero, zero)
        else:
            vq = AuxiliaryVQOutput(
                loss=raw_vq.loss + self.vq_usage_weight * usage_kl,
                commitment_loss=raw_vq.commitment_loss,
                usage_kl=usage_kl,
                reconstruction_loss=zero,
                perplexity=raw_vq.perplexity,
                active_fraction=raw_vq.active_fraction,
                dead_fraction=raw_vq.dead_fraction,
            )
        output = CausalAuxVQMambaV5Output(
            enhanced=enhanced,
            estimated_magnitude=estimated_magnitude,
            predicted_phase=predicted_phase,
            phase_candidate=noisy_phase + phase_residual,
            magnitude_mask=magnitude_mask,
            continuous_noise_state=continuous_noise,
            code_indices=code_indices,
            code_posterior=code_posterior,
            code_perplexity=vq.perplexity,
            vq_adapter_strength=adapter_strength,
            noise_prediction=noise_prediction,
            encoder_features=features,
            mamba_features=current,
            vq=vq,
            phase_confidence=phase_confidence,
        )
        return output


__all__ = [
    "CausalAuxVQMambaV51",
    "FrameGroupNorm",
    "LearnedCausalDualHeadDecoder",
    "TwoStageCausalEncoder",
]
