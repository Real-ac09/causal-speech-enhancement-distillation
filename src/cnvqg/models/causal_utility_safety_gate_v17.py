from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .causal_feature_cumulative_gate_v17 import (
    CausalFeatureCumulativeStrengthGate,
)
from .causal_ordinal_residual_gate_v17 import (
    CausalOrdinalResidualGateV17,
    CausalOrdinalResidualGateV17Output,
)


@dataclass
class CausalUtilitySafetyGateV17Output(CausalOrdinalResidualGateV17Output):
    """Waveform output plus Recipe-5 utility and safety predictions."""

    gate_utility: torch.Tensor
    gate_log_violation: torch.Tensor
    gate_feasibility_logits: torch.Tensor
    gate_metric_deltas: torch.Tensor


class CausalUtilitySafetySelector(nn.Module):
    """Causal multi-task selector over a fixed enhancement-strength grid."""

    FEATURE_NAMES = CausalFeatureCumulativeStrengthGate.FEATURE_NAMES
    mixture_summaries = staticmethod(
        CausalFeatureCumulativeStrengthGate.mixture_summaries
    )
    rich_summaries = classmethod(
        CausalFeatureCumulativeStrengthGate.rich_summaries.__func__
    )

    def __init__(
        self,
        noise_dim: int,
        hidden_dim: int = 24,
        strength_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        metric_count: int = 4,
        violation_penalty: float = 2.0,
        feasibility_log_weight: float = 2.0,
        selection_temperature: float = 0.5,
    ) -> None:
        super().__init__()
        grid = torch.as_tensor(tuple(strength_grid), dtype=torch.float32)
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if grid.shape != (5,) or not bool(torch.all(grid[1:] > grid[:-1])):
            raise ValueError("Recipe 5 requires five ordered strengths")
        if metric_count != 4:
            raise ValueError("Recipe 5 requires four quality metrics")
        if violation_penalty < 0.0 or feasibility_log_weight < 0.0:
            raise ValueError("Selection safety weights must be non-negative")
        if selection_temperature <= 0.0:
            raise ValueError("selection_temperature must be positive")

        self.noise_dim = int(noise_dim)
        self.hidden_dim = int(hidden_dim)
        self.metric_count = int(metric_count)
        self.violation_penalty = float(violation_penalty)
        self.feasibility_log_weight = float(feasibility_log_weight)
        self.selection_temperature = float(selection_temperature)
        self.register_buffer("strength_grid", grid)

        self.noise_norm = nn.LayerNorm(self.noise_dim)
        self.noise_projection = nn.Linear(self.noise_dim, self.hidden_dim)
        self.summary_projection = nn.Linear(
            len(self.FEATURE_NAMES),
            self.hidden_dim,
        )
        self.temporal = nn.GRU(
            self.hidden_dim,
            self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        levels = grid.numel()
        self.utility_head = nn.Linear(self.hidden_dim, levels)
        self.violation_head = nn.Linear(self.hidden_dim, levels)
        self.feasibility_head = nn.Linear(self.hidden_dim, levels)
        self.metric_delta_head = nn.Linear(
            self.hidden_dim,
            levels * self.metric_count,
        )
        for head in (
            self.utility_head,
            self.violation_head,
            self.feasibility_head,
            self.metric_delta_head,
        ):
            nn.init.normal_(head.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(head.bias)

    def details_from_hidden(
        self,
        hidden: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        utility = self.utility_head(hidden)
        log_violation = F.softplus(self.violation_head(hidden))
        feasibility_logits = self.feasibility_head(hidden)
        metric_deltas = self.metric_delta_head(hidden).unflatten(
            -1,
            (self.strength_grid.numel(), self.metric_count),
        )
        selection_logits = (
            utility
            - self.violation_penalty * log_violation
            + self.feasibility_log_weight
            * F.logsigmoid(feasibility_logits)
        ) / self.selection_temperature
        probabilities = torch.softmax(
            selection_logits.float(),
            dim=-1,
        ).to(selection_logits.dtype)
        strength = torch.sum(
            probabilities * self.strength_grid.to(probabilities.dtype),
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

    def forward_with_details(
        self,
        continuous_noise_state: torch.Tensor,
        mixture_spectrum: torch.Tensor,
        base_spectrum: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        summaries = self.rich_summaries(
            mixture_spectrum,
            base_spectrum,
        )
        frames = min(continuous_noise_state.shape[1], summaries.shape[1])
        noise = continuous_noise_state[:, :frames]
        summaries = summaries[:, :frames]
        current = self.noise_projection(self.noise_norm(noise))
        current = F.silu(current + self.summary_projection(summaries))
        current, _ = self.temporal(current)
        return self.details_from_hidden(current)

    def forward(
        self,
        continuous_noise_state: torch.Tensor,
        mixture_spectrum: torch.Tensor,
        base_spectrum: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_with_details(
            continuous_noise_state,
            mixture_spectrum,
            base_spectrum,
        )[0]


class CausalUtilitySafetyGateV17(CausalOrdinalResidualGateV17):
    """Frozen Candidate A with Recipe-5 utility/safety strength control."""

    def __init__(
        self,
        backbone_checkpoint: str,
        gate_hidden_dim: int = 24,
        strength_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        metric_count: int = 4,
        violation_penalty: float = 2.0,
        feasibility_log_weight: float = 2.0,
        selection_temperature: float = 0.5,
        gate_parameter_cap: int = 15_000,
    ) -> None:
        super().__init__(
            backbone_checkpoint=backbone_checkpoint,
            gate_hidden_dim=gate_hidden_dim,
            strength_grid=strength_grid,
            gate_parameter_cap=gate_parameter_cap,
        )
        self.confidence_gate = CausalUtilitySafetySelector(
            noise_dim=self.backbone.noise_dim,
            hidden_dim=gate_hidden_dim,
            strength_grid=strength_grid,
            metric_count=metric_count,
            violation_penalty=violation_penalty,
            feasibility_log_weight=feasibility_log_weight,
            selection_temperature=selection_temperature,
        )
        if self.gate_parameter_count() > int(gate_parameter_cap):
            raise ValueError("Recipe-5 gate exceeds its parameter cap")

    def train(self, mode: bool = True) -> CausalUtilitySafetyGateV17:
        super().train(mode)
        return self

    def _forward_waveform(
        self,
        noisy: torch.Tensor,
        pad_end: bool,
    ) -> CausalUtilitySafetyGateV17Output:
        if noisy.ndim != 3 or noisy.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, T], got {tuple(noisy.shape)}")
        with torch.no_grad():
            base = self.backbone._forward_waveform(noisy, pad_end=pad_end)
        mixture_spectrum = (
            base.speech_spectrum + base.noise_spectrum
        ).detach()
        (
            frame_strength,
            selection_logits,
            probabilities,
            utility,
            log_violation,
            feasibility_logits,
            metric_deltas,
        ) = self.confidence_gate.forward_with_details(
            base.continuous_noise_state.detach(),
            mixture_spectrum,
            base.speech_spectrum.detach(),
        )
        base_enhanced = base.enhanced[..., : noisy.shape[-1]].detach()
        sample_strength = self.frame_strength_to_samples(
            frame_strength,
            hop_length=self.hop_length,
            output_length=noisy.shape[-1],
        )
        enhanced = self.blend_waveforms(
            noisy,
            base_enhanced,
            sample_strength,
        )
        values = dict(base.__dict__)
        values.update(
            enhanced=enhanced,
            gate_strength=frame_strength,
            sample_strength=sample_strength,
            base_enhanced=base_enhanced,
            gate_logits=selection_logits,
            gate_probabilities=probabilities,
            gate_utility=utility,
            gate_log_violation=log_violation,
            gate_feasibility_logits=feasibility_logits,
            gate_metric_deltas=metric_deltas,
        )
        return CausalUtilitySafetyGateV17Output(**values)


__all__ = [
    "CausalUtilitySafetyGateV17",
    "CausalUtilitySafetyGateV17Output",
    "CausalUtilitySafetySelector",
]
