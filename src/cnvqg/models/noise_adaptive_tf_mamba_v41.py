from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .complex_cnvqg_model import TemporalStack
from .noise_adaptive_tf_mamba import NoiseAdaptiveTFMamba, _groups
from .noise_vq import VQOutput


def _norm_act(channels: int) -> nn.Sequential:
    return nn.Sequential(nn.GroupNorm(_groups(channels), channels), nn.SiLU())


class SkipTFEncoder(nn.Module):
    """Frequency pyramid that retains both reconstruction resolutions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        half = channels // 2
        self.stem = nn.Sequential(nn.Conv2d(3, half, 3, padding=1), _norm_act(half))
        self.down_one = nn.Sequential(
            nn.Conv2d(half, channels, 3, stride=(2, 1), padding=1),
            _norm_act(channels),
        )
        self.down_two = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=(2, 1), padding=1),
            _norm_act(channels),
        )

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full = self.stem(features)
        half = self.down_one(full)
        latent = self.down_two(half)
        return latent, half, full


def _match_tf(features: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Crop or interpolate only for odd-bin transposed-convolution mismatches."""

    target = reference.shape[-2:]
    if features.shape[-2:] == target:
        return features
    return F.interpolate(features, size=target, mode="bilinear", align_corners=False)


class LearnedSkipDecoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        half = channels // 2
        self.up_one = nn.Sequential(
            nn.ConvTranspose2d(
                channels,
                channels,
                kernel_size=(4, 3),
                stride=(2, 1),
                padding=(1, 1),
            ),
            _norm_act(channels),
        )
        self.fuse_one = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            _norm_act(channels),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            _norm_act(channels),
        )
        self.up_two = nn.Sequential(
            nn.ConvTranspose2d(
                channels,
                half,
                kernel_size=(4, 3),
                stride=(2, 1),
                padding=(1, 1),
            ),
            _norm_act(half),
        )
        self.fuse_two = nn.Sequential(
            nn.Conv2d(channels, half, 1),
            _norm_act(half),
            nn.Conv2d(half, half, 3, padding=1, groups=half),
            nn.Conv2d(half, half, 1),
            _norm_act(half),
        )

    def forward(
        self,
        latent: torch.Tensor,
        half_skip: torch.Tensor,
        full_skip: torch.Tensor,
    ) -> torch.Tensor:
        features = _match_tf(self.up_one(latent), half_skip)
        features = self.fuse_one(torch.cat((features, half_skip), dim=1))
        features = _match_tf(self.up_two(features), full_skip)
        return self.fuse_two(torch.cat((features, full_skip), dim=1))


class BranchRefiner(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            _norm_act(channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.net(features)


class SegmentConditionedDualAxisMambaCell(nn.Module):
    """Shared dual-axis cell with time-varying segment conditioning."""

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
        self.condition = nn.Linear(noise_dim, channels * 2)

    def forward(
        self, features: torch.Tensor, noise_states: torch.Tensor
    ) -> torch.Tensor:
        batch, channels, bins, frames = features.shape
        residual = features + self.local(features)

        time_sequence = residual.permute(0, 2, 3, 1).reshape(
            batch * bins, frames, channels
        )
        time_sequence = self.time_model(time_sequence)
        time_features = time_sequence.view(batch, bins, frames, channels).permute(
            0, 3, 1, 2
        )
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
            condition = self.condition(noise_states).transpose(1, 2)
            condition = F.interpolate(condition, size=frames, mode="nearest")
            scale, shift = condition.chunk(2, dim=1)
            residual = residual * (
                1.0 + 0.1 * torch.tanh(scale[:, :, None, :])
            )
            residual = residual + 0.1 * shift[:, :, None, :]
        return residual


@dataclass
class NoiseAdaptiveTFV41Output:
    enhanced: torch.Tensor
    estimated_magnitude: torch.Tensor
    estimated_phase: torch.Tensor
    phase_candidate: torch.Tensor
    phase_confidence: torch.Tensor
    magnitude_mask: torch.Tensor
    noise_state: torch.Tensor
    code_indices: torch.Tensor
    expected_iterations: torch.Tensor
    halting_probabilities: torch.Tensor
    noise_prediction: torch.Tensor
    vq: VQOutput
    vq_gate: torch.Tensor


class NoiseAdaptiveTFMambaV41(NoiseAdaptiveTFMamba):
    """V4.1: skip reconstruction, dual heads, and segmentwise noise states."""

    PRESETS = {
        "tiny": {"channels": 48, "noise_dim": 32, "codebook_size": 8},
        "base": {"channels": 72, "noise_dim": 48, "codebook_size": 16},
        "large": {"channels": 96, "noise_dim": 64, "codebook_size": 16},
        "xl": {"channels": 144, "noise_dim": 96, "codebook_size": 16},
    }

    def __init__(self, *args, noise_segment_frames: int = 32, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if noise_segment_frames < 1:
            raise ValueError("noise_segment_frames must be positive")
        self.noise_segment_frames = int(noise_segment_frames)
        self.noise_vq.update_interval = 1

        self.encoder = SkipTFEncoder(self.channels)
        self.cell = SegmentConditionedDualAxisMambaCell(
            channels=self.channels,
            noise_dim=self.noise_dim,
            use_mamba=bool(kwargs.get("use_mamba", True)),
            mamba_d_state=int(kwargs.get("mamba_d_state", 16)),
            mamba_d_conv=int(kwargs.get("mamba_d_conv", 4)),
            mamba_expand=int(kwargs.get("mamba_expand", 2)),
            condition_dynamics=self.condition_dynamics,
        )
        self.decoder = LearnedSkipDecoder(self.channels)
        head_channels = self.channels // 2
        self.magnitude_branch = BranchRefiner(head_channels)
        self.magnitude_head = nn.Conv2d(head_channels, 1, 1)
        self.phase_branch = BranchRefiner(head_channels)
        self.magnitude_to_phase = nn.Conv2d(head_channels, head_channels, 1)
        self.phase_cross_scale = nn.Parameter(torch.tensor(0.0))
        self.phase_head = nn.Conv2d(
            head_channels, 3 if self.use_phase_confidence else 2, 1
        )
        # Start as an identity enhancer: mask_max * sigmoid(0) == 1 for the
        # default mask_max=2, and atan2(0, 1) gives a zero phase residual.
        nn.init.zeros_(self.magnitude_head.weight)
        nn.init.zeros_(self.magnitude_head.bias)
        nn.init.zeros_(self.phase_head.weight)
        nn.init.zeros_(self.phase_head.bias)
        with torch.no_grad():
            self.phase_head.bias[1] = 1.0

    def _segment_noise(self, features: torch.Tensor) -> torch.Tensor:
        sequence = features.mean(dim=-2).transpose(1, 2)
        frames = sequence.shape[1]
        segments = math.ceil(frames / self.noise_segment_frames)
        padded_frames = segments * self.noise_segment_frames
        if padded_frames > frames:
            sequence = F.pad(
                sequence.transpose(1, 2),
                (0, padded_frames - frames),
                mode="replicate",
            ).transpose(1, 2)
        sequence = sequence.view(
            sequence.shape[0], segments, self.noise_segment_frames, self.channels
        ).mean(dim=2)
        return self.noise_encoder(sequence)

    def forward(self, noisy: torch.Tensor) -> NoiseAdaptiveTFV41Output:
        if noisy.ndim != 3 or noisy.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, T], got {tuple(noisy.shape)}")
        length = noisy.shape[-1]
        spectrum = self._stft(noisy.float())
        magnitude = spectrum.abs().clamp_min(1e-7)
        noisy_phase = torch.angle(spectrum)
        compressed = magnitude.pow(self.magnitude_power)
        inputs = torch.stack(
            (compressed, torch.cos(noisy_phase), torch.sin(noisy_phase)), dim=1
        )
        features, half_skip, full_skip = self.encoder(inputs)

        continuous_noise = self._segment_noise(features)
        if self.use_noise_codebook:
            vq = self.noise_vq(continuous_noise)
            noise_states, vq_gate = self._blend_vq(
                continuous_noise, vq.quantized
            )
        else:
            vq = self._empty_vq(continuous_noise)
            noise_states = continuous_noise
            vq_gate = self._vq_gate(continuous_noise)
        pooled_noise = noise_states.mean(dim=1)

        remaining = features.new_ones(features.shape[0])
        accumulated = torch.zeros_like(features)
        halt_probabilities = []
        expected_iterations = features.new_zeros(features.shape[0])
        current = features
        for index in range(self.max_iterations):
            current = self.cell(current, noise_states)
            if self.adaptive_iterations and index < self.max_iterations - 1:
                pooled = current.mean(dim=(-2, -1))
                halt = torch.sigmoid(
                    self.halting_head(
                        torch.cat((pooled, pooled_noise), dim=-1)
                    ).squeeze(-1)
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

        decoded = self.decoder(accumulated, half_skip, full_skip)
        magnitude_features = self.magnitude_branch(decoded)
        magnitude_mask = self.mask_max * torch.sigmoid(
            self.magnitude_head(magnitude_features).squeeze(1)
        )
        estimated_magnitude = magnitude * magnitude_mask

        phase_features = self.phase_branch(decoded)
        phase_features = phase_features + torch.tanh(
            self.phase_cross_scale
        ) * self.magnitude_to_phase(magnitude_features)
        phase_output = self.phase_head(phase_features)
        phase_residual = torch.atan2(
            phase_output[:, 0], phase_output[:, 1] + 1e-8
        )
        phase_candidate = noisy_phase + phase_residual
        if self.use_phase_confidence:
            confidence = torch.sigmoid(phase_output[:, 2])
        else:
            confidence = torch.ones_like(phase_residual)
        estimated_phase = noisy_phase + confidence * phase_residual

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
        noise_prediction = self.noise_predictor(noise_states).transpose(1, 2)
        return NoiseAdaptiveTFV41Output(
            enhanced=enhanced,
            estimated_magnitude=estimated_magnitude,
            estimated_phase=estimated_phase,
            phase_candidate=phase_candidate,
            phase_confidence=confidence,
            magnitude_mask=magnitude_mask,
            noise_state=noise_states,
            code_indices=vq.indices,
            expected_iterations=expected_iterations,
            halting_probabilities=torch.stack(halt_probabilities, dim=-1),
            noise_prediction=noise_prediction,
            vq=vq,
            vq_gate=vq_gate,
        )


__all__ = ["NoiseAdaptiveTFMambaV41", "NoiseAdaptiveTFV41Output"]
