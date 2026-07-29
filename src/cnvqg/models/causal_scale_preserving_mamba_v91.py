from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .causal_prototype_dual_axis_mamba_v9 import (
    CausalPrototypeDualAxisMambaV9,
    CausalPrototypeDualAxisMambaV9Output,
    FrameChannelRMSNorm,
    FrequencyResidualUnitV9,
)


class ScalePreservingFrequencyResidualUnit(nn.Module):
    """Frequency-local residual unit that retains absolute feature scale."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=(3, 1), padding=(1, 0), groups=channels
        )
        self.expand = nn.Conv2d(channels, hidden * 2, 1)
        self.project = nn.Conv2d(hidden, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(x)
        value, gate = self.expand(y).chunk(2, dim=1)
        y = self.project(F.silu(value) * torch.sigmoid(gate))
        return x + torch.tanh(self.scale) * y


class ScalePreservingEncoderV91(nn.Module):
    """Two-scale encoder with an unnormalised high-resolution skip."""

    def __init__(self, detail_channels: int, half_channels: int, core_channels: int) -> None:
        super().__init__()
        self.stem_conv = nn.Conv2d(3, detail_channels, kernel_size=(5, 1), padding=(2, 0))
        self.stem_unit = ScalePreservingFrequencyResidualUnit(detail_channels)
        self.down_one = nn.Sequential(
            nn.Conv2d(
                detail_channels,
                half_channels,
                kernel_size=(3, 1),
                stride=(2, 1),
                padding=(1, 0),
            ),
            FrameChannelRMSNorm(half_channels),
            nn.SiLU(),
            FrequencyResidualUnitV9(half_channels),
        )
        self.down_two = nn.Sequential(
            nn.Conv2d(
                half_channels,
                core_channels,
                kernel_size=(3, 1),
                stride=(2, 1),
                padding=(1, 0),
            ),
            FrameChannelRMSNorm(core_channels),
            nn.SiLU(),
            FrequencyResidualUnitV9(core_channels),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        detail = self.stem_unit(F.silu(self.stem_conv(x)))
        half = self.down_one(detail)
        core = self.down_two(half)
        return core, half, detail


class ExplicitPolarDecoderV91(nn.Module):
    """Scale-preserving magnitude decoder with an isolated phase branch."""

    def __init__(self, detail_channels: int, half_channels: int, core_channels: int) -> None:
        super().__init__()
        self.core_to_half = nn.Conv2d(core_channels, half_channels, 1)
        self.half_fuse = nn.Sequential(
            nn.Conv2d(half_channels * 2, half_channels, 1),
            FrameChannelRMSNorm(half_channels),
            nn.SiLU(),
            FrequencyResidualUnitV9(half_channels),
        )
        self.half_to_detail = nn.Conv2d(half_channels, detail_channels, 1)
        # The final channel is the raw compressed noisy magnitude. No
        # normalisation follows this fusion, so absolute level remains visible.
        self.detail_fuse = nn.Sequential(
            nn.Conv2d(detail_channels * 2 + 1, detail_channels, 1),
            nn.SiLU(),
            ScalePreservingFrequencyResidualUnit(detail_channels),
        )
        self.magnitude_head = nn.Conv2d(detail_channels + 1, 1, 1)
        self.phase_head = nn.Conv2d(detail_channels, 1, 1)
        nn.init.zeros_(self.magnitude_head.weight)
        nn.init.zeros_(self.magnitude_head.bias)
        nn.init.zeros_(self.phase_head.weight)
        nn.init.zeros_(self.phase_head.bias)

    def forward(
        self,
        core: torch.Tensor,
        half_skip: torch.Tensor,
        detail_skip: torch.Tensor,
        compressed_magnitude: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        half = F.interpolate(
            self.core_to_half(core), size=half_skip.shape[-2:], mode="nearest"
        )
        half = self.half_fuse(torch.cat((half, half_skip), dim=1))
        detail = F.interpolate(
            self.half_to_detail(half), size=detail_skip.shape[-2:], mode="nearest"
        )
        magnitude_input = compressed_magnitude[:, None].to(detail.dtype)
        detail = self.detail_fuse(torch.cat((detail, detail_skip, magnitude_input), dim=1))
        magnitude_logits = self.magnitude_head(
            torch.cat((detail, magnitude_input), dim=1)
        ).squeeze(1)
        phase_logits = self.phase_head(detail).squeeze(1)
        return magnitude_logits, phase_logits, detail


class CausalScalePreservingMambaV91(CausalPrototypeDualAxisMambaV9):
    """V9.1: V9 temporal core with a scale-preserving explicit polar decoder."""

    def __init__(
        self,
        *args,
        magnitude_mask_maximum: float = 2.0,
        phase_residual_limit: float = 0.0,
        **kwargs,
    ) -> None:
        enforce_parameter_cap = bool(kwargs.get("enforce_parameter_cap", True))
        # The parent topology is temporary and is replaced below.
        kwargs["enforce_parameter_cap"] = False
        if magnitude_mask_maximum <= 1.0:
            raise ValueError("magnitude_mask_maximum must exceed one for identity initialisation")
        if not 0.0 <= phase_residual_limit <= math.pi:
            raise ValueError("phase_residual_limit must be in [0, pi]")
        super().__init__(*args, **kwargs)
        self.magnitude_mask_maximum = float(magnitude_mask_maximum)
        self.phase_residual_limit = float(phase_residual_limit)
        self.encoder = ScalePreservingEncoderV91(
            self.detail_channels, self.half_channels, self.core_channels
        )
        self.decoder = ExplicitPolarDecoderV91(
            self.detail_channels, self.half_channels, self.core_channels
        )
        if enforce_parameter_cap and self.parameter_count() > self.parameter_cap:
            raise ValueError(
                f"V9.1 {self.variant} has {self.parameter_count():,} parameters, "
                f"exceeding cap {self.parameter_cap:,}"
            )

    def _forward_waveform(
        self, noisy: torch.Tensor, pad_end: bool
    ) -> CausalPrototypeDualAxisMambaV9Output:
        if noisy.ndim != 3 or noisy.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, T], got {tuple(noisy.shape)}")
        spectrum, original_length = self._analysis(noisy.squeeze(1), pad_end=pad_end)
        if spectrum.shape[-1] == 0:
            raise ValueError("At least one complete analysis window is required")
        magnitude = spectrum.abs().clamp_min(1e-7)
        compressed = magnitude.pow(self.magnitude_power)
        unit = spectrum / magnitude
        inputs = torch.stack(
            (compressed * unit.real, compressed * unit.imag, torch.log1p(magnitude)), dim=1
        ).to(self.encoder.stem_conv.weight.dtype)
        core, half_skip, detail_skip = self.encoder(inputs)
        current = core
        for block in self.core:
            current = block(current)
        magnitude_logits, phase_logits, _ = self.decoder(
            current, half_skip, detail_skip, compressed
        )
        maximum = self.magnitude_mask_maximum
        identity_offset = math.log(1.0 / (maximum - 1.0))
        magnitude_mask = maximum * torch.sigmoid(magnitude_logits + identity_offset)
        phase_residual = self.phase_residual_limit * torch.tanh(phase_logits)
        predicted_phase = torch.angle(spectrum) + phase_residual.float()
        estimated_magnitude = magnitude * magnitude_mask.float()
        enhanced_spectrum = torch.polar(estimated_magnitude, predicted_phase)
        enhanced = self._synthesis(enhanced_spectrum, original_length).unsqueeze(1)

        compressed_gain = magnitude_mask.float().clamp_min(1e-7).pow(self.magnitude_power)
        complex_mask = torch.polar(compressed_gain, phase_residual.float())
        vq, indices, posterior, noise = self._empty_vq(
            enhanced, noisy.shape[0], spectrum.shape[-1]
        )
        confidence = torch.ones_like(predicted_phase)
        zero = enhanced.new_tensor(0.0)
        return CausalPrototypeDualAxisMambaV9Output(
            enhanced=enhanced,
            estimated_magnitude=estimated_magnitude,
            predicted_phase=predicted_phase,
            phase_candidate=predicted_phase,
            phase_confidence=confidence,
            magnitude_mask=magnitude_mask,
            complex_mask=complex_mask,
            continuous_noise_state=noise,
            code_indices=indices,
            code_posterior=posterior,
            code_perplexity=zero,
            vq_adapter_strength=zero,
            noise_prediction=None,
            encoder_features=core,
            mamba_features=current,
            vq=vq,
        )


__all__ = [
    "CausalScalePreservingMambaV91",
    "ExplicitPolarDecoderV91",
    "ScalePreservingEncoderV91",
    "ScalePreservingFrequencyResidualUnit",
]
