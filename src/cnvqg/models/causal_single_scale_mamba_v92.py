from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .causal_prototype_dual_axis_mamba_v9 import (
    CausalPrototypeDualAxisMambaV9Output,
    FrameChannelRMSNorm,
    FrequencyResidualUnitV9,
)
from .causal_scale_preserving_mamba_v91 import (
    CausalScalePreservingMambaV91,
    ScalePreservingFrequencyResidualUnit,
)


class SingleReductionEncoderV92(nn.Module):
    """Scale-preserving 257-bin detail path and one 257 -> 129 reduction."""

    def __init__(self, detail_channels: int, core_channels: int) -> None:
        super().__init__()
        self.stem_conv = nn.Conv2d(3, detail_channels, kernel_size=(5, 1), padding=(2, 0))
        self.stem_unit = ScalePreservingFrequencyResidualUnit(detail_channels)
        self.down = nn.Sequential(
            nn.Conv2d(
                detail_channels,
                core_channels,
                kernel_size=(3, 1),
                stride=(2, 1),
                padding=(1, 0),
            ),
            FrameChannelRMSNorm(core_channels),
            nn.SiLU(),
            FrequencyResidualUnitV9(core_channels),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        detail = self.stem_unit(F.silu(self.stem_conv(x)))
        return self.down(detail), detail


class SingleReductionPolarDecoderV92(nn.Module):
    def __init__(self, detail_channels: int, core_channels: int) -> None:
        super().__init__()
        self.core_to_detail = nn.Conv2d(core_channels, detail_channels, 1)
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
        detail_skip: torch.Tensor,
        compressed_magnitude: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        detail = F.interpolate(
            self.core_to_detail(core), size=detail_skip.shape[-2:], mode="nearest"
        )
        magnitude_input = compressed_magnitude[:, None].to(detail.dtype)
        detail = self.detail_fuse(torch.cat((detail, detail_skip, magnitude_input), dim=1))
        magnitude_logits = self.magnitude_head(
            torch.cat((detail, magnitude_input), dim=1)
        ).squeeze(1)
        phase_logits = self.phase_head(detail).squeeze(1)
        return magnitude_logits, phase_logits, detail


class CausalSingleScaleMambaV92(CausalScalePreservingMambaV91):
    """V9.2: one frequency reduction so temporal Mamba retains 129 bins."""

    PRESETS = {
        "student": {
            "detail_channels": 40,
            "half_channels": 80,
            "core_channels": 176,
            "blocks": 2,
            "noise_dim": 64,
            "cap": 1_100_000,
        },
        "teacher": {
            "detail_channels": 64,
            "half_channels": 128,
            "core_channels": 240,
            "blocks": 3,
            "noise_dim": 96,
            "cap": 2_700_000,
        },
    }

    def __init__(self, *args, **kwargs) -> None:
        enforce_parameter_cap = bool(kwargs.get("enforce_parameter_cap", True))
        # V9.1's topology is temporary and is replaced below.
        kwargs["enforce_parameter_cap"] = False
        super().__init__(*args, **kwargs)
        self.encoder = SingleReductionEncoderV92(self.detail_channels, self.core_channels)
        self.decoder = SingleReductionPolarDecoderV92(
            self.detail_channels, self.core_channels
        )
        if enforce_parameter_cap and self.parameter_count() > self.parameter_cap:
            raise ValueError(
                f"V9.2 {self.variant} has {self.parameter_count():,} parameters, "
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
        inputs = self._input_features(magnitude, compressed, unit).to(
            next(self.encoder.parameters()).dtype
        )
        core, detail_skip = self.encoder(inputs)
        current, continuous_noise = self._prepare_core(core)
        magnitude_logits, phase_logits, _ = self.decoder(
            current, detail_skip, compressed
        )
        maximum = self.magnitude_mask_maximum
        identity_offset = math.log(1.0 / (maximum - 1.0))
        magnitude_mask = maximum * torch.sigmoid(magnitude_logits + identity_offset)
        phase_residual = self.phase_residual_limit * torch.tanh(phase_logits)
        predicted_phase = torch.angle(spectrum) + phase_residual.float()
        estimated_magnitude = magnitude * magnitude_mask.float()
        enhanced = self._synthesis(
            torch.polar(estimated_magnitude, predicted_phase), original_length
        ).unsqueeze(1)
        compressed_gain = magnitude_mask.float().clamp_min(1e-7).pow(self.magnitude_power)
        complex_mask = torch.polar(compressed_gain, phase_residual.float())
        vq, indices, posterior, _ = self._empty_vq(
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
            continuous_noise_state=continuous_noise,
            code_indices=indices,
            code_posterior=posterior,
            code_perplexity=zero,
            vq_adapter_strength=zero,
            noise_prediction=None,
            encoder_features=core,
            mamba_features=current,
            vq=vq,
        )

    def _process_core(self, core: torch.Tensor) -> torch.Tensor:
        current = core
        for block in self.core:
            current = block(current)
        return current

    def _input_features(
        self,
        magnitude: torch.Tensor,
        compressed: torch.Tensor,
        unit: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            (compressed * unit.real, compressed * unit.imag, torch.log1p(magnitude)), dim=1
        )

    def _prepare_core(self, core: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extension point for separately gated continuous conditioning."""
        noise = core.new_zeros(core.shape[0], core.shape[-1], self.noise_dim)
        return self._process_core(core), noise


__all__ = [
    "CausalSingleScaleMambaV92",
    "SingleReductionEncoderV92",
    "SingleReductionPolarDecoderV92",
]
