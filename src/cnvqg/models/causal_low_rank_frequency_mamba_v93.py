from __future__ import annotations

import torch
from torch import nn

from .causal_single_scale_mamba_v92 import CausalSingleScaleMambaV92
from .complex_cnvqg_model import TemporalStack


class LowRankBidirectionalFrequencyMamba(nn.Module):
    """Within-frame full-band adapter; bidirectional frequency is time-causal."""

    def __init__(
        self,
        channels: int,
        rank: int,
        use_mamba: bool,
        mamba_d_state: int,
        mamba_d_conv: int,
        mamba_expand: int,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("frequency rank must be positive")
        self.rank = int(rank)
        self.input_projection = nn.Conv2d(channels, rank, 1, bias=False)
        self.norm = nn.LayerNorm(rank)
        self.frequency_mamba = TemporalStack(
            dim=rank,
            layers=1,
            use_mamba=use_mamba,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
        )
        self.merge = nn.Linear(rank * 2, rank)
        self.output_projection = nn.Conv2d(rank, channels, 1, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        # Preserve the trained/control topology at initialisation. The output
        # projection learns first, then opens gradients into the adapter.
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, bins, frames = x.shape
        low_rank = self.input_projection(x)
        sequence = low_rank.permute(0, 3, 2, 1).reshape(batch * frames, bins, self.rank)
        sequence = self.norm(sequence)
        forward = self.frequency_mamba(sequence) - sequence
        reversed_sequence = sequence.flip(1)
        backward = (
            self.frequency_mamba(reversed_sequence) - reversed_sequence
        ).flip(1)
        frequency = self.merge(torch.cat((forward, backward), dim=-1))
        frequency = frequency.view(batch, frames, bins, self.rank).permute(0, 3, 2, 1)
        update = self.output_projection(frequency)
        return x + torch.tanh(self.residual_scale) * update


class CausalLowRankFrequencyMambaV93(CausalSingleScaleMambaV92):
    """V9.3: V9.2 plus low-rank full-band context in each current frame."""

    PRESETS = {
        "student": {
            "detail_channels": 40,
            "half_channels": 80,
            "core_channels": 176,
            "blocks": 2,
            "noise_dim": 64,
            "frequency_rank": 32,
            "cap": 1_100_000,
        },
        "teacher": {
            "detail_channels": 64,
            "half_channels": 128,
            "core_channels": 232,
            "blocks": 3,
            "noise_dim": 96,
            "frequency_rank": 48,
            "cap": 2_700_000,
        },
    }

    def __init__(
        self,
        *args,
        frequency_rank: int | None = None,
        use_mamba: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        **kwargs,
    ) -> None:
        variant = kwargs.get("variant", args[0] if args else "student")
        preset = self.PRESETS[variant]
        enforce_parameter_cap = bool(kwargs.get("enforce_parameter_cap", True))
        kwargs["enforce_parameter_cap"] = False
        super().__init__(
            *args,
            use_mamba=use_mamba,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            **kwargs,
        )
        self.frequency_rank = int(frequency_rank or preset["frequency_rank"])
        self.frequency_context = LowRankBidirectionalFrequencyMamba(
            channels=self.core_channels,
            rank=self.frequency_rank,
            use_mamba=use_mamba,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
        )
        if enforce_parameter_cap and self.parameter_count() > self.parameter_cap:
            raise ValueError(
                f"V9.3 {self.variant} has {self.parameter_count():,} parameters, "
                f"exceeding cap {self.parameter_cap:,}"
            )

    def _process_core(self, core: torch.Tensor) -> torch.Tensor:
        return self.frequency_context(super()._process_core(core))


__all__ = ["CausalLowRankFrequencyMambaV93", "LowRankBidirectionalFrequencyMamba"]
