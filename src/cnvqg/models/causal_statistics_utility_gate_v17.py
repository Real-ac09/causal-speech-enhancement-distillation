from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .causal_utility_safety_gate_v17 import (
    CausalUtilitySafetyGateV17,
    CausalUtilitySafetySelector,
)


class CausalStatisticsUtilitySafetySelector(CausalUtilitySafetySelector):
    """Recipe-5 selector augmented with constant-memory prefix statistics."""

    STATISTIC_COUNT = 5

    def __init__(
        self,
        noise_dim: int,
        hidden_dim: int = 16,
        strength_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        metric_count: int = 4,
        violation_penalty: float = 2.0,
        feasibility_log_weight: float = 2.0,
        selection_temperature: float = 0.5,
    ) -> None:
        super().__init__(
            noise_dim=noise_dim,
            hidden_dim=hidden_dim,
            strength_grid=strength_grid,
            metric_count=metric_count,
            violation_penalty=violation_penalty,
            feasibility_log_weight=feasibility_log_weight,
            selection_temperature=selection_temperature,
        )
        statistic_input_dim = (
            self.noise_dim + len(self.FEATURE_NAMES)
        ) * self.STATISTIC_COUNT
        self.statistics_projection = nn.Linear(
            statistic_input_dim,
            self.hidden_dim,
        )
        # A Recipe-5b checkpoint is an exact functional initialisation: the
        # statistics branch starts as a zero residual and is learned safely.
        nn.init.zeros_(self.statistics_projection.weight)
        nn.init.zeros_(self.statistics_projection.bias)

    @staticmethod
    def causal_statistics(features: torch.Tensor) -> torch.Tensor:
        """Return mean, std, minimum, maximum and change-rate per prefix."""

        if features.ndim != 3:
            raise ValueError("features must be [batch, frames, channels]")
        values = features.float()
        frames = values.shape[1]
        count = torch.arange(
            1,
            frames + 1,
            device=values.device,
            dtype=values.dtype,
        ).view(1, frames, 1)
        mean = values.cumsum(dim=1) / count
        second_moment = values.square().cumsum(dim=1) / count
        standard_deviation = (
            (second_moment - mean.square()).clamp_min(0.0) + 1e-6
        ).sqrt()
        minimum = values.cummin(dim=1).values
        maximum = values.cummax(dim=1).values
        change = torch.cat(
            (
                torch.zeros_like(values[:, :1]),
                (values[:, 1:] - values[:, :-1]).abs(),
            ),
            dim=1,
        )
        change_count = (count - 1.0).clamp_min(1.0)
        change_rate = change.cumsum(dim=1) / change_count
        return torch.cat(
            (
                mean,
                standard_deviation,
                minimum,
                maximum,
                change_rate,
            ),
            dim=-1,
        ).to(features.dtype)

    def forward_with_details(
        self,
        continuous_noise_state: torch.Tensor,
        mixture_spectrum: torch.Tensor,
        base_spectrum: torch.Tensor,
    ):
        summaries = self.rich_summaries(
            mixture_spectrum,
            base_spectrum,
        )
        frames = min(continuous_noise_state.shape[1], summaries.shape[1])
        noise = continuous_noise_state[:, :frames]
        summaries = summaries[:, :frames]
        normalised_noise = self.noise_norm(noise)
        statistics = self.causal_statistics(
            torch.cat((normalised_noise, summaries), dim=-1)
        )
        current = self.noise_projection(normalised_noise)
        current = F.silu(
            current
            + self.summary_projection(summaries)
            + self.statistics_projection(statistics)
        )
        current, _ = self.temporal(current)
        return self.details_from_hidden(current)


class CausalStatisticsUtilityGateV17(CausalUtilitySafetyGateV17):
    """Recipe 6: Recipe 5b plus deployable causal summary statistics."""

    def __init__(
        self,
        backbone_checkpoint: str,
        gate_hidden_dim: int = 16,
        strength_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        metric_count: int = 4,
        violation_penalty: float = 2.0,
        feasibility_log_weight: float = 2.0,
        selection_temperature: float = 0.5,
        gate_parameter_cap: int = 10_000,
    ) -> None:
        super().__init__(
            backbone_checkpoint=backbone_checkpoint,
            gate_hidden_dim=gate_hidden_dim,
            strength_grid=strength_grid,
            metric_count=metric_count,
            violation_penalty=violation_penalty,
            feasibility_log_weight=feasibility_log_weight,
            selection_temperature=selection_temperature,
            gate_parameter_cap=gate_parameter_cap,
        )
        self.confidence_gate = CausalStatisticsUtilitySafetySelector(
            noise_dim=self.backbone.noise_dim,
            hidden_dim=gate_hidden_dim,
            strength_grid=strength_grid,
            metric_count=metric_count,
            violation_penalty=violation_penalty,
            feasibility_log_weight=feasibility_log_weight,
            selection_temperature=selection_temperature,
        )
        if self.gate_parameter_count() > int(gate_parameter_cap):
            raise ValueError("Recipe-6 gate exceeds its parameter cap")


__all__ = [
    "CausalStatisticsUtilityGateV17",
    "CausalStatisticsUtilitySafetySelector",
]
