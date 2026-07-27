from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .causal_aux_vq_mamba_v5 import CausalAuxVQMambaV5, V5StreamState
from .causal_confidence_gate_v14 import CausalResidualConfidenceGate
from .causal_oracle_residual_gate_v16 import CausalOracleResidualGateV16
from .predictive_noise_vq_mamba_v8 import (
    PredictiveNoiseVQMambaV8,
    PredictiveNoiseVQMambaV8Output,
)


@dataclass
class CausalOrdinalResidualGateV17Output(PredictiveNoiseVQMambaV8Output):
    """Frozen-backbone output with an ordinal waveform residual controller."""

    gate_strength: torch.Tensor
    sample_strength: torch.Tensor
    base_enhanced: torch.Tensor
    gate_logits: torch.Tensor
    gate_probabilities: torch.Tensor


class CausalOrdinalStrengthGate(nn.Module):
    """Causal categorical controller whose expectation is the blend strength."""

    def __init__(
        self,
        noise_dim: int,
        hidden_dim: int = 16,
        strength_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    ) -> None:
        super().__init__()
        grid = torch.as_tensor(tuple(strength_grid), dtype=torch.float32)
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if grid.ndim != 1 or grid.numel() < 2:
            raise ValueError("strength_grid must contain at least two values")
        if not bool(torch.all(grid[1:] > grid[:-1])):
            raise ValueError("strength_grid must be strictly increasing")
        if float(grid[0]) < 0.0 or float(grid[-1]) > 1.0:
            raise ValueError("strength_grid values must lie in [0, 1]")

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
        self.output = nn.Linear(self.hidden_dim, grid.numel())

        # Uniform initial probabilities give an unbiased strength of 0.5 and
        # avoid the saturated scalar sigmoid that trapped V16 near strength 1.
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output.bias)

    mixture_summaries = staticmethod(
        CausalResidualConfidenceGate.mixture_summaries
    )

    def forward_with_logits(
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
        logits = self.output(current)
        probabilities = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
        strength = torch.sum(
            probabilities * self.strength_grid.to(probabilities.dtype),
            dim=-1,
            keepdim=True,
        )
        return strength, logits, probabilities

    def forward(
        self,
        continuous_noise_state: torch.Tensor,
        mixture_spectrum: torch.Tensor,
    ) -> torch.Tensor:
        strength, _, _ = self.forward_with_logits(
            continuous_noise_state,
            mixture_spectrum,
        )
        return strength


class CausalOrdinalResidualGateV17(nn.Module):
    """Frozen Candidate-A enhancer with a five-level causal residual gate."""

    SUPPORTED_BACKBONE_ARCHITECTURES = {
        "causal_temporal_core_v12",
        "cnvqg_v12",
        "predictive_noise_vq_mamba_v8",
    }

    def __init__(
        self,
        backbone_checkpoint: str,
        gate_hidden_dim: int = 16,
        strength_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        gate_parameter_cap: int = 10_000,
    ) -> None:
        super().__init__()
        checkpoint_path = Path(backbone_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"V17 backbone checkpoint not found: {checkpoint_path}"
            )
        checkpoint: dict[str, Any] = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        backbone_config = dict(checkpoint["config"]["model"])
        architecture = backbone_config.pop("architecture", "")
        if architecture not in self.SUPPORTED_BACKBONE_ARCHITECTURES:
            raise ValueError(
                f"Unsupported V17 backbone architecture: {architecture!r}"
            )
        self.backbone_checkpoint = str(backbone_checkpoint)
        self.backbone = PredictiveNoiseVQMambaV8(**backbone_config)
        self.backbone.load_state_dict(checkpoint["model_state_dict"])
        if self.backbone.reconstruction_mode != "direct_scalar_mask":
            raise ValueError(
                "V17 requires the candidate-A direct scalar-mask backbone"
            )
        if self.backbone.phase_residual_scale != 0.0:
            raise ValueError(
                "V17 requires the candidate-A zero-phase-residual backbone"
            )
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        self.confidence_gate = CausalOrdinalStrengthGate(
            noise_dim=self.backbone.noise_dim,
            hidden_dim=gate_hidden_dim,
            strength_grid=strength_grid,
        )
        gate_parameters = self.gate_parameter_count()
        if gate_parameters > int(gate_parameter_cap):
            raise ValueError(
                f"V17 gate has {gate_parameters:,} parameters, exceeding "
                f"the {int(gate_parameter_cap):,} cap"
            )

        self.sample_rate = self.backbone.sample_rate
        self.n_fft = self.backbone.n_fft
        self.hop_length = self.backbone.hop_length
        self.win_length = self.backbone.win_length
        self.algorithmic_latency_samples = (
            self.backbone.algorithmic_latency_samples
        )

    def train(self, mode: bool = True) -> CausalOrdinalResidualGateV17:
        super().train(mode)
        self.backbone.eval()
        return self

    def gate_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.confidence_gate.parameters()
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def temporal(self) -> nn.Module:
        return self.backbone.temporal

    frame_strength_to_samples = staticmethod(
        CausalOracleResidualGateV16.frame_strength_to_samples
    )
    blend_waveforms = staticmethod(
        CausalOracleResidualGateV16.blend_waveforms
    )

    def _forward_waveform(
        self,
        noisy: torch.Tensor,
        pad_end: bool,
    ) -> CausalOrdinalResidualGateV17Output:
        if noisy.ndim != 3 or noisy.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, T], got {tuple(noisy.shape)}")
        with torch.no_grad():
            base = self.backbone._forward_waveform(noisy, pad_end=pad_end)

        mixture_spectrum = (
            base.speech_spectrum + base.noise_spectrum
        ).detach()
        frame_strength, gate_logits, gate_probabilities = (
            self.confidence_gate.forward_with_logits(
                base.continuous_noise_state.detach(),
                mixture_spectrum,
            )
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
            gate_logits=gate_logits,
            gate_probabilities=gate_probabilities,
        )
        return CausalOrdinalResidualGateV17Output(**values)

    def forward(
        self,
        noisy: torch.Tensor,
    ) -> CausalOrdinalResidualGateV17Output:
        return self._forward_waveform(noisy, pad_end=True)

    init_stream_state = CausalAuxVQMambaV5.init_stream_state
    forward_chunk = CausalAuxVQMambaV5.forward_chunk
    flush = CausalAuxVQMambaV5.flush


__all__ = [
    "CausalOrdinalResidualGateV17",
    "CausalOrdinalResidualGateV17Output",
    "CausalOrdinalStrengthGate",
    "V5StreamState",
]
