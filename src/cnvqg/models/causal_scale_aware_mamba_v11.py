from __future__ import annotations

import torch
from torch import nn

from .predictive_noise_vq_mamba_v8 import (
    MixtureConsistentComplexDecoder,
    PredictiveNoiseVQMambaV8,
)


class ContextualScaleAdapter(nn.Module):
    """A zero-initialised, frequency-local correction to direct mask logits."""

    def __init__(
        self,
        detail_channels: int,
        hidden_channels: int = 16,
        use_frequency_coordinate: bool = True,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        if not 0.0 < residual_scale <= 1.0:
            raise ValueError("residual_scale must be in (0, 1]")
        self.use_frequency_coordinate = bool(use_frequency_coordinate)
        self.residual_scale = float(residual_scale)
        scale_channels = 4 if self.use_frequency_coordinate else 3
        self.scale_projection = nn.Conv2d(scale_channels, hidden_channels, 1)
        self.context_projection = nn.Conv2d(detail_channels, hidden_channels, 1)
        self.frequency_filter = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=(9, 1),
            padding=(4, 0),
            groups=hidden_channels,
        )
        self.output = nn.Conv2d(hidden_channels, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        decoded: torch.Tensor,
        noisy_magnitude: torch.Tensor,
        magnitude_power: float,
    ) -> torch.Tensor:
        compressed = noisy_magnitude.float().clamp_min(1e-7).pow(magnitude_power)
        frame_level = compressed.mean(dim=-2, keepdim=True).clamp_min(1e-5)
        relative = compressed / frame_level
        features = [
            torch.log1p(compressed),
            torch.log1p(frame_level).expand_as(compressed),
            torch.log1p(relative),
        ]
        if self.use_frequency_coordinate:
            coordinate = torch.linspace(
                -1.0,
                1.0,
                compressed.shape[-2],
                device=compressed.device,
                dtype=compressed.dtype,
            )
            features.append(coordinate[None, :, None].expand_as(compressed))
        scale_features = torch.stack(features, dim=1).to(decoded.dtype)
        current = self.scale_projection(scale_features)
        current = torch.nn.functional.silu(current + self.context_projection(decoded))
        current = torch.nn.functional.silu(self.frequency_filter(current))
        return self.residual_scale * self.output(current).float()


class ScaleAwareDirectDecoder(MixtureConsistentComplexDecoder):
    def __init__(
        self,
        channels: int,
        *,
        magnitude_power: float,
        adapter_hidden_channels: int,
        use_frequency_coordinate: bool,
        adapter_residual_scale: float,
        mask_bound: float = 2.0,
    ) -> None:
        super().__init__(
            channels,
            mask_bound=mask_bound,
            reconstruction_mode="direct_scalar_mask",
            magnitude_power=magnitude_power,
            scale_preserving_detail=False,
        )
        self.scale_adapter = ContextualScaleAdapter(
            channels // 2,
            hidden_channels=adapter_hidden_channels,
            use_frequency_coordinate=use_frequency_coordinate,
            residual_scale=adapter_residual_scale,
        )

    def _raw_controls(
        self, decoded: torch.Tensor, noisy_magnitude: torch.Tensor | None
    ) -> torch.Tensor:
        raw = super()._raw_controls(decoded, noisy_magnitude)
        if noisy_magnitude is None:
            raise ValueError("Scale-aware direct decoding requires noisy magnitude")
        return raw + self.scale_adapter(decoded, noisy_magnitude, self.magnitude_power)


class CausalScaleAwareMambaV11(PredictiveNoiseVQMambaV8):
    """V8 with a selective explicit-scale path at the full-resolution decoder."""

    def __init__(
        self,
        *args,
        scale_adapter_hidden_channels: int = 16,
        use_frequency_coordinate: bool = True,
        scale_adapter_residual_scale: float = 0.1,
        **kwargs,
    ) -> None:
        reconstruction_mode = str(kwargs.get("reconstruction_mode", "direct_scalar_mask"))
        if reconstruction_mode != "direct_scalar_mask":
            raise ValueError("V11 scale-aware decoding requires direct_scalar_mask")
        if bool(kwargs.get("scale_preserving_detail", False)):
            raise ValueError("V11 replaces the legacy scale_preserving_detail path")
        enforce_parameter_cap = bool(kwargs.get("enforce_parameter_cap", True))
        kwargs["enforce_parameter_cap"] = False
        super().__init__(*args, **kwargs)
        self.decoder = ScaleAwareDirectDecoder(
            self.channels,
            magnitude_power=self.magnitude_power,
            adapter_hidden_channels=scale_adapter_hidden_channels,
            use_frequency_coordinate=use_frequency_coordinate,
            adapter_residual_scale=scale_adapter_residual_scale,
            mask_bound=float(kwargs.get("mask_bound", 2.0)),
        )
        count = self.parameter_count()
        if enforce_parameter_cap and count > self.parameter_cap:
            raise ValueError(
                f"V11 {self.variant} has {count:,} parameters, exceeding {self.parameter_cap:,}"
            )


__all__ = [
    "CausalScaleAwareMambaV11",
    "ContextualScaleAdapter",
    "ScaleAwareDirectDecoder",
]
