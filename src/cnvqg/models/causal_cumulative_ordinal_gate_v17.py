from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .causal_confidence_gate_v14 import CausalResidualConfidenceGate
from .causal_ordinal_residual_gate_v17 import (
    CausalOrdinalResidualGateV17,
    CausalOrdinalResidualGateV17Output,
)


@dataclass
class CausalCumulativeOrdinalGateV17Output(
    CausalOrdinalResidualGateV17Output
):
    """V17 output augmented with four ordered cumulative logits."""

    gate_ordinal_logits: torch.Tensor


class CausalCumulativeOrdinalStrengthGate(nn.Module):
    """Rank-consistent cumulative ordinal controller.

    A shared scalar score is compared with four strictly ordered cutpoints.
    Consequently P(class > k) cannot cross between thresholds, and the five
    reconstructed class probabilities are always non-negative and sum to one.
    """

    def __init__(
        self,
        noise_dim: int,
        hidden_dim: int = 16,
        strength_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        initial_cutpoints: Sequence[float] = (-1.5, -0.5, 0.5, 1.5),
    ) -> None:
        super().__init__()
        grid = torch.as_tensor(tuple(strength_grid), dtype=torch.float32)
        cutpoints = torch.as_tensor(
            tuple(initial_cutpoints),
            dtype=torch.float32,
        )
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if grid.ndim != 1 or grid.numel() != 5:
            raise ValueError("Recipe 3 requires five strength levels")
        if not bool(torch.all(grid[1:] > grid[:-1])):
            raise ValueError("strength_grid must be strictly increasing")
        if cutpoints.shape != (grid.numel() - 1,):
            raise ValueError("One cutpoint is required between each class")
        if not bool(torch.all(cutpoints[1:] > cutpoints[:-1])):
            raise ValueError("initial_cutpoints must be strictly increasing")

        self.noise_dim = int(noise_dim)
        self.hidden_dim = int(hidden_dim)
        self.register_buffer("strength_grid", grid)
        self.noise_norm = nn.LayerNorm(self.noise_dim)
        self.noise_projection = nn.Linear(self.noise_dim, self.hidden_dim)
        self.summary_projection = nn.Linear(4, self.hidden_dim)
        self.temporal = nn.GRU(
            self.hidden_dim,
            self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output = nn.Linear(self.hidden_dim, 1)
        self.first_cutpoint = nn.Parameter(cutpoints[:1].clone())
        gaps = cutpoints[1:] - cutpoints[:-1]
        self.raw_cutpoint_gaps = nn.Parameter(
            torch.log(torch.expm1(gaps))
        )

        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output.bias)

    mixture_summaries = staticmethod(
        CausalResidualConfidenceGate.mixture_summaries
    )

    def ordered_cutpoints(self) -> torch.Tensor:
        gaps = F.softplus(self.raw_cutpoint_gaps).clamp_min(1e-4)
        return torch.cat(
            (
                self.first_cutpoint,
                self.first_cutpoint + torch.cumsum(gaps, dim=0),
            )
        )

    @staticmethod
    def class_probabilities(
        ordinal_logits: torch.Tensor,
    ) -> torch.Tensor:
        cumulative = torch.sigmoid(ordinal_logits.float())
        probabilities = torch.cat(
            (
                1.0 - cumulative[..., :1],
                cumulative[..., :-1] - cumulative[..., 1:],
                cumulative[..., -1:],
            ),
            dim=-1,
        )
        return probabilities.clamp_min(0.0).to(ordinal_logits.dtype)

    def details_from_hidden(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        score = self.output(hidden)
        ordinal_logits = score - self.ordered_cutpoints().view(1, 1, -1)
        probabilities = self.class_probabilities(ordinal_logits)
        strength = torch.sum(
            probabilities * self.strength_grid.to(probabilities.dtype),
            dim=-1,
            keepdim=True,
        )
        return strength, ordinal_logits, probabilities

    def forward_with_details(
        self,
        continuous_noise_state: torch.Tensor,
        mixture_spectrum: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        summaries = self.mixture_summaries(mixture_spectrum)
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
    ) -> torch.Tensor:
        strength, _, _ = self.forward_with_details(
            continuous_noise_state,
            mixture_spectrum,
        )
        return strength


class CausalCumulativeOrdinalGateV17(CausalOrdinalResidualGateV17):
    """Recipe-3 cumulative ordinal controller over the V17 waveform mixer."""

    def __init__(
        self,
        backbone_checkpoint: str,
        gate_hidden_dim: int = 16,
        strength_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        initial_cutpoints: Sequence[float] = (-1.5, -0.5, 0.5, 1.5),
        gate_parameter_cap: int = 10_000,
    ) -> None:
        super().__init__(
            backbone_checkpoint=backbone_checkpoint,
            gate_hidden_dim=gate_hidden_dim,
            strength_grid=strength_grid,
            gate_parameter_cap=gate_parameter_cap,
        )
        self.confidence_gate = CausalCumulativeOrdinalStrengthGate(
            noise_dim=self.backbone.noise_dim,
            hidden_dim=gate_hidden_dim,
            strength_grid=strength_grid,
            initial_cutpoints=initial_cutpoints,
        )
        if self.gate_parameter_count() > int(gate_parameter_cap):
            raise ValueError("Recipe-3 gate exceeds its parameter cap")

    def train(self, mode: bool = True) -> CausalCumulativeOrdinalGateV17:
        super().train(mode)
        return self

    def _forward_waveform(
        self,
        noisy: torch.Tensor,
        pad_end: bool,
    ) -> CausalCumulativeOrdinalGateV17Output:
        if noisy.ndim != 3 or noisy.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, T], got {tuple(noisy.shape)}")
        with torch.no_grad():
            base = self.backbone._forward_waveform(noisy, pad_end=pad_end)

        mixture_spectrum = (
            base.speech_spectrum + base.noise_spectrum
        ).detach()
        frame_strength, ordinal_logits, probabilities = (
            self.confidence_gate.forward_with_details(
                base.continuous_noise_state.detach(),
                mixture_spectrum,
            )
        )
        class_logits = torch.log(probabilities.float().clamp_min(1e-7)).to(
            ordinal_logits.dtype
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
            gate_logits=class_logits,
            gate_probabilities=probabilities,
            gate_ordinal_logits=ordinal_logits,
        )
        return CausalCumulativeOrdinalGateV17Output(**values)


__all__ = [
    "CausalCumulativeOrdinalGateV17",
    "CausalCumulativeOrdinalGateV17Output",
    "CausalCumulativeOrdinalStrengthGate",
]
