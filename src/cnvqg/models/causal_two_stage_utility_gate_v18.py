from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .causal_statistics_utility_gate_v17 import (
    CausalStatisticsUtilityGateV17,
    CausalStatisticsUtilitySafetySelector,
)


class CausalTwoStageUtilitySafetySelector(
    CausalStatisticsUtilitySafetySelector
):
    """Factor full-enhancement routing from reduced-strength selection."""

    def __init__(
        self,
        noise_dim: int,
        hidden_dim: int = 16,
        strength_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        metric_count: int = 4,
        violation_penalty: float = 2.0,
        feasibility_log_weight: float = 2.0,
        selection_temperature: float = 0.5,
        full_route_temperature: float = 1.0,
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
        if full_route_temperature <= 0.0:
            raise ValueError("full_route_temperature must be positive")
        self.full_route_temperature = float(full_route_temperature)
        self.full_route_head = nn.Linear(self.hidden_dim, 1)
        self.reduced_policy_head = nn.Linear(
            self.hidden_dim,
            self.strength_grid.numel() - 1,
        )
        nn.init.zeros_(self.full_route_head.weight)
        nn.init.zeros_(self.full_route_head.bias)
        nn.init.zeros_(self.reduced_policy_head.weight)
        nn.init.zeros_(self.reduced_policy_head.bias)

    def details_from_hidden(self, hidden: torch.Tensor):
        utility = self.utility_head(hidden)
        log_violation = F.softplus(self.violation_head(hidden))
        feasibility_logits = self.feasibility_head(hidden)
        metric_deltas = self.metric_delta_head(hidden).unflatten(
            -1,
            (self.strength_grid.numel(), self.metric_count),
        )

        base_scores = (
            utility
            - self.violation_penalty * log_violation
            + self.feasibility_log_weight
            * F.logsigmoid(feasibility_logits)
        ) / self.selection_temperature
        reduced_scores = (
            base_scores[..., :-1]
            + self.reduced_policy_head(hidden)
        )
        full_route_logit = (
            base_scores[..., -1:]
            - torch.logsumexp(base_scores[..., :-1], dim=-1, keepdim=True)
            + self.full_route_head(hidden)
        ) / self.full_route_temperature
        full_probability = torch.sigmoid(full_route_logit.float())
        reduced_probability = torch.softmax(
            reduced_scores.float(),
            dim=-1,
        )
        probabilities = torch.cat(
            (
                (1.0 - full_probability) * reduced_probability,
                full_probability,
            ),
            dim=-1,
        ).to(hidden.dtype)
        selection_logits = probabilities.float().clamp_min(1e-7).log().to(
            hidden.dtype
        )
        strength = torch.sum(
            probabilities
            * self.strength_grid.to(
                device=probabilities.device,
                dtype=probabilities.dtype,
            ),
            dim=-1,
            keepdim=True,
        )
        return (
            strength,
            selection_logits,
            probabilities,
            utility,
            log_violation,
            feasibility_logits,
            metric_deltas,
        )


class CausalTwoStageUtilityGateV18(CausalStatisticsUtilityGateV17):
    """Recipe 8 two-stage controller around frozen Candidate A."""

    def __init__(
        self,
        backbone_checkpoint: str,
        gate_hidden_dim: int = 16,
        strength_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        metric_count: int = 4,
        violation_penalty: float = 2.0,
        feasibility_log_weight: float = 2.0,
        selection_temperature: float = 0.5,
        full_route_temperature: float = 1.0,
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
        self.confidence_gate = CausalTwoStageUtilitySafetySelector(
            noise_dim=self.backbone.noise_dim,
            hidden_dim=gate_hidden_dim,
            strength_grid=strength_grid,
            metric_count=metric_count,
            violation_penalty=violation_penalty,
            feasibility_log_weight=feasibility_log_weight,
            selection_temperature=selection_temperature,
            full_route_temperature=full_route_temperature,
        )
        if self.gate_parameter_count() > int(gate_parameter_cap):
            raise ValueError("Recipe-8 gate exceeds its parameter cap")


__all__ = [
    "CausalTwoStageUtilityGateV18",
    "CausalTwoStageUtilitySafetySelector",
]
