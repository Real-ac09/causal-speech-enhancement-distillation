from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .causal_prototype_dual_axis_mamba_v9 import FrameChannelRMSNorm, FrequencyResidualUnitV9
from .causal_single_scale_mamba_v92 import CausalSingleScaleMambaV92
from .streaming_hybrid_v2 import CausalConv2d


class ScalePreservingCausalTFResidualUnit(nn.Module):
    """Local time-frequency detail unit with left-only temporal context."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.depthwise = CausalConv2d(
            channels, channels, kernel_size=(3, 3), groups=channels
        )
        self.expand = nn.Conv2d(channels, hidden * 2, 1)
        self.project = nn.Conv2d(hidden, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(x)
        value, gate = self.expand(y).chunk(2, dim=1)
        y = self.project(F.silu(value) * torch.sigmoid(gate))
        return x + torch.tanh(self.scale) * y


class CausalTemporalDetailEncoderV94(nn.Module):
    def __init__(self, detail_channels: int, core_channels: int) -> None:
        super().__init__()
        self.stem_conv = CausalConv2d(3, detail_channels, kernel_size=(5, 3))
        self.stem_unit = ScalePreservingCausalTFResidualUnit(detail_channels)
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


class CausalTemporalDetailDecoderV94(nn.Module):
    def __init__(self, detail_channels: int, core_channels: int) -> None:
        super().__init__()
        self.core_to_detail = nn.Conv2d(core_channels, detail_channels, 1)
        self.detail_fuse = nn.Sequential(
            CausalConv2d(detail_channels * 2 + 1, detail_channels, kernel_size=(5, 3)),
            nn.SiLU(),
            ScalePreservingCausalTFResidualUnit(detail_channels),
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


class CausalTemporalDetailMambaV94(CausalSingleScaleMambaV92):
    """V9.4: V9.2 with causal temporal context on the detail path."""

    # The temporal detail decoder adds more weights than V9.2.  The teacher's
    # core is narrowed one increment so both published variants remain inside
    # the same deployment envelopes.
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
            "core_channels": 232,
            "blocks": 3,
            "noise_dim": 96,
            "cap": 2_700_000,
        },
    }

    def __init__(self, *args, **kwargs) -> None:
        enforce_parameter_cap = bool(kwargs.get("enforce_parameter_cap", True))
        kwargs["enforce_parameter_cap"] = False
        super().__init__(*args, **kwargs)
        self.encoder = CausalTemporalDetailEncoderV94(
            self.detail_channels, self.core_channels
        )
        self.decoder = CausalTemporalDetailDecoderV94(
            self.detail_channels, self.core_channels
        )
        if enforce_parameter_cap and self.parameter_count() > self.parameter_cap:
            raise ValueError(
                f"V9.4 {self.variant} has {self.parameter_count():,} parameters, "
                f"exceeding cap {self.parameter_cap:,}"
            )


__all__ = [
    "CausalTemporalDetailMambaV94",
    "CausalTemporalDetailEncoderV94",
    "CausalTemporalDetailDecoderV94",
    "ScalePreservingCausalTFResidualUnit",
]
