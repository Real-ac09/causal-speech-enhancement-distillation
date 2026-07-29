from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .causal_aux_vq_mamba_v5 import CausalAuxVQMambaV5, V5StreamState
from .predictive_noise_vq_mamba_v8 import (
    PredictiveNoiseVQMambaV8,
    PredictiveNoiseVQMambaV8Output,
)


@dataclass
class CausalConfidenceGateV14Output(PredictiveNoiseVQMambaV8Output):
    """V13 output augmented with the causal residual-strength decision."""

    gate_strength: torch.Tensor


class CausalResidualConfidenceGate(nn.Module):
    """Small causal frame gate driven only by mixture-derived features."""

    def __init__(
        self,
        noise_dim: int,
        hidden_dim: int = 16,
        minimum_strength: float = 0.0,
        initial_strength: float = 0.995,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= minimum_strength < 1.0:
            raise ValueError("minimum_strength must be in [0, 1)")
        if not minimum_strength < initial_strength < 1.0:
            raise ValueError(
                "initial_strength must be between minimum_strength and one"
            )
        self.noise_dim = int(noise_dim)
        self.hidden_dim = int(hidden_dim)
        self.minimum_strength = float(minimum_strength)

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

        # Start extremely close to the frozen V13 output while retaining a
        # useful derivative for the first optimisation steps.
        fraction = (
            (float(initial_strength) - self.minimum_strength)
            / (1.0 - self.minimum_strength)
        )
        initial_logit = math.log(fraction / (1.0 - fraction))
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.output.bias, initial_logit)

    @staticmethod
    def mixture_summaries(mixture_spectrum: torch.Tensor) -> torch.Tensor:
        """Return four frame-local/causal descriptors in stable ranges."""

        magnitude = mixture_spectrum.abs().float().clamp_min(1e-7)
        rms = magnitude.square().mean(dim=-2).sqrt()
        log_rms = torch.log10(rms).clamp(-5.0, 1.0) / 3.0
        flatness = torch.exp(torch.log(magnitude).mean(dim=-2)) / magnitude.mean(
            dim=-2
        ).clamp_min(1e-7)
        split = max(1, (3 * magnitude.shape[-2]) // 4)
        high_energy = magnitude[:, split:].square().sum(dim=-2)
        total_energy = magnitude.square().sum(dim=-2).clamp_min(1e-7)
        high_ratio = high_energy / total_energy
        compressed = torch.log1p(magnitude)
        previous = F.pad(compressed[..., :-1], (1, 0), mode="replicate")
        flux = (compressed - previous).abs().mean(dim=-2).clamp_max(1.0)
        return torch.stack((log_rms, flatness, high_ratio, flux), dim=-1)

    def forward(
        self,
        continuous_noise_state: torch.Tensor,
        mixture_spectrum: torch.Tensor,
    ) -> torch.Tensor:
        summaries = self.mixture_summaries(mixture_spectrum)
        frames = min(continuous_noise_state.shape[1], summaries.shape[1])
        noise = continuous_noise_state[:, :frames]
        summaries = summaries[:, :frames]
        current = self.noise_projection(self.noise_norm(noise))
        current = F.silu(current + self.summary_projection(summaries))
        current, _ = self.temporal(current)
        fraction = torch.sigmoid(self.output(current))
        return self.minimum_strength + (1.0 - self.minimum_strength) * fraction


class CausalConfidenceGateV14(nn.Module):
    """Frozen V13 enhancer with a trainable causal residual confidence gate."""

    SUPPORTED_BACKBONE_ARCHITECTURES = {
        "causal_temporal_core_v12",
        "cnvqg_v12",
        "predictive_noise_vq_mamba_v8",
    }

    def __init__(
        self,
        backbone_checkpoint: str,
        gate_hidden_dim: int = 16,
        gate_minimum_strength: float = 0.0,
        gate_initial_strength: float = 0.995,
        gate_parameter_cap: int = 10_000,
    ) -> None:
        super().__init__()
        checkpoint_path = Path(backbone_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"V14 backbone checkpoint not found: {checkpoint_path}"
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
                f"Unsupported V14 backbone architecture: {architecture!r}"
            )
        self.backbone_checkpoint = str(backbone_checkpoint)
        self.backbone = PredictiveNoiseVQMambaV8(**backbone_config)
        self.backbone.load_state_dict(checkpoint["model_state_dict"])
        if self.backbone.reconstruction_mode != "direct_scalar_mask":
            raise ValueError("V14.1 requires the V13 direct scalar-mask backbone")
        if self.backbone.phase_residual_scale != 0.0:
            raise ValueError("V14.1 requires the frozen zero-phase-residual backbone")
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        self.confidence_gate = CausalResidualConfidenceGate(
            noise_dim=self.backbone.noise_dim,
            hidden_dim=gate_hidden_dim,
            minimum_strength=gate_minimum_strength,
            initial_strength=gate_initial_strength,
        )
        gate_parameters = self.gate_parameter_count()
        if gate_parameters > int(gate_parameter_cap):
            raise ValueError(
                f"V14 confidence gate has {gate_parameters:,} parameters, "
                f"exceeding the {int(gate_parameter_cap):,} cap"
            )

        self.sample_rate = self.backbone.sample_rate
        self.n_fft = self.backbone.n_fft
        self.hop_length = self.backbone.hop_length
        self.win_length = self.backbone.win_length
        self.algorithmic_latency_samples = self.backbone.algorithmic_latency_samples

    def train(self, mode: bool = True) -> CausalConfidenceGateV14:
        super().train(mode)
        # ``nn.Module.train`` recurses into children, so restore the frozen
        # backbone's deterministic inference behaviour after every call.
        self.backbone.eval()
        return self

    def gate_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.confidence_gate.parameters())

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def temporal(self) -> nn.Module:
        """Expose the frozen temporal stack for trainer/runtime diagnostics."""

        return self.backbone.temporal

    def _forward_waveform(
        self,
        noisy: torch.Tensor,
        pad_end: bool,
    ) -> CausalConfidenceGateV14Output:
        if noisy.ndim != 3 or noisy.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, T], got {tuple(noisy.shape)}")
        with torch.no_grad():
            base = self.backbone._forward_waveform(noisy, pad_end=pad_end)

        mixture_spectrum = base.speech_spectrum + base.noise_spectrum
        strength = self.confidence_gate(
            base.continuous_noise_state.detach(),
            mixture_spectrum.detach(),
        )
        frames = min(mixture_spectrum.shape[-1], strength.shape[1])
        mixture_spectrum = mixture_spectrum[..., :frames]
        frame_strength = strength[:, :frames].transpose(1, 2)
        # The frozen V13 backbone is a non-negative direct scalar mask with
        # noisy phase. Blend its mask algebraically to avoid complex division
        # and angle gradients at spectral nulls.
        base_mask = base.magnitude_mask[..., :frames].detach()
        magnitude_mask = 1.0 + frame_strength * (
            base_mask - 1.0
        )
        speech_spectrum = magnitude_mask * mixture_spectrum
        noise_spectrum = mixture_spectrum - speech_spectrum
        enhanced = self.backbone._synthesis(
            speech_spectrum,
            noisy.shape[-1],
        ).unsqueeze(1)
        estimated_magnitude = speech_spectrum.abs().clamp_min(1e-7)
        speech_mask = magnitude_mask.to(mixture_spectrum.dtype)
        noise_mask = 1.0 - speech_mask
        predicted_phase = base.predicted_phase[..., :frames].detach()
        mixture_residual = mixture_spectrum - speech_spectrum - noise_spectrum

        values = dict(base.__dict__)
        values.update(
            enhanced=enhanced,
            estimated_magnitude=estimated_magnitude,
            predicted_phase=predicted_phase,
            phase_candidate=base.phase_candidate[..., :frames].detach(),
            magnitude_mask=magnitude_mask,
            magnitude_residual=torch.zeros_like(magnitude_mask),
            phase_confidence=torch.ones_like(estimated_magnitude),
            speech_spectrum=speech_spectrum,
            noise_spectrum=noise_spectrum,
            speech_mask=speech_mask,
            noise_mask=noise_mask,
            mixture_residual=mixture_residual,
            gate_strength=strength[:, :frames],
        )
        return CausalConfidenceGateV14Output(**values)

    def forward(self, noisy: torch.Tensor) -> CausalConfidenceGateV14Output:
        return self._forward_waveform(noisy, pad_end=True)

    # The correctness-first streaming contract buffers causal input and calls
    # this wrapper's `_forward_waveform`, so it also exercises the gate.
    init_stream_state = CausalAuxVQMambaV5.init_stream_state
    forward_chunk = CausalAuxVQMambaV5.forward_chunk
    flush = CausalAuxVQMambaV5.flush


__all__ = [
    "CausalConfidenceGateV14",
    "CausalConfidenceGateV14Output",
    "CausalResidualConfidenceGate",
    "V5StreamState",
]
