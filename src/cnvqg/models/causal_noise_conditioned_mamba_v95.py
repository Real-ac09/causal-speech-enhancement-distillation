from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .causal_single_scale_mamba_v92 import CausalSingleScaleMambaV92


class CausalNoiseConditionedMambaV95(CausalSingleScaleMambaV92):
    """V9.2 plus bounded conditioning from a continuous causal noise state."""

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

    def __init__(
        self,
        *args,
        noise_condition_limit: float = 0.05,
        use_noise_conditioning: bool = True,
        **kwargs,
    ) -> None:
        if not 0.0 <= noise_condition_limit <= 0.1:
            raise ValueError("noise_condition_limit must be in [0, 0.1]")
        enforce_parameter_cap = bool(kwargs.get("enforce_parameter_cap", True))
        kwargs["enforce_parameter_cap"] = False
        # The control class deliberately rejects this flag. V9.5 implements it
        # locally and keeps the unconditioned path available for exact ablation.
        kwargs["use_noise_conditioning"] = False
        super().__init__(*args, **kwargs)
        self.use_noise_conditioning = bool(use_noise_conditioning)
        self.noise_condition_limit = float(noise_condition_limit)
        self.noise_encoder = nn.Sequential(
            nn.Linear(self.core_channels, self.noise_dim),
            nn.SiLU(),
            nn.LayerNorm(self.noise_dim),
        )
        self.noise_film = nn.Linear(self.noise_dim, self.core_channels * 2)
        # Exact V9.2 behaviour at initialization; conditioning must earn its use.
        nn.init.zeros_(self.noise_film.weight)
        nn.init.zeros_(self.noise_film.bias)
        if enforce_parameter_cap and self.parameter_count() > self.parameter_cap:
            raise ValueError(
                f"V9.5 {self.variant} has {self.parameter_count():,} parameters, "
                f"exceeding cap {self.parameter_cap:,}"
            )

    def _rolling_noise(self, core: torch.Tensor) -> torch.Tensor:
        sequence = core.mean(dim=-2).transpose(1, 2)
        # Four causal 10 ms frames, including the current frame.
        sequence = F.pad(sequence.transpose(1, 2), (3, 0), mode="replicate")
        sequence = F.avg_pool1d(sequence, kernel_size=4, stride=1).transpose(1, 2)
        return self.noise_encoder(sequence)

    def _prepare_core(self, core: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        noise = self._rolling_noise(core)
        current = core
        if self.use_noise_conditioning:
            scale, shift = self.noise_film(noise).transpose(1, 2).chunk(2, dim=1)
            limit = self.noise_condition_limit
            current = current * (1.0 + limit * torch.tanh(scale[:, :, None]))
            current = current + limit * torch.tanh(shift[:, :, None])
        return self._process_core(current), noise


__all__ = ["CausalNoiseConditionedMambaV95"]
