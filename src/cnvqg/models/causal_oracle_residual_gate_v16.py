from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .causal_aux_vq_mamba_v5 import CausalAuxVQMambaV5, V5StreamState
from .causal_confidence_gate_v14 import CausalResidualConfidenceGate
from .predictive_noise_vq_mamba_v8 import (
    PredictiveNoiseVQMambaV8,
    PredictiveNoiseVQMambaV8Output,
)


@dataclass
class CausalOracleResidualGateV16Output(PredictiveNoiseVQMambaV8Output):
    """Frozen-backbone output with a true waveform residual controller."""

    gate_strength: torch.Tensor
    sample_strength: torch.Tensor
    base_enhanced: torch.Tensor


class CausalOracleResidualGateV16(nn.Module):
    """Frozen enhancer with a supervised causal waveform-residual gate.

    Unlike the V15 spectral-mask blend, strength zero is an exact waveform
    identity path and strength one is the frozen backbone output. The gate
    remains causal and constant-work; only its small recurrent controller is
    trainable.
    """

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
        gate_initial_strength: float = 0.75,
        gate_parameter_cap: int = 10_000,
    ) -> None:
        super().__init__()
        checkpoint_path = Path(backbone_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"V16 backbone checkpoint not found: {checkpoint_path}"
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
                f"Unsupported V16 backbone architecture: {architecture!r}"
            )
        self.backbone_checkpoint = str(backbone_checkpoint)
        self.backbone = PredictiveNoiseVQMambaV8(**backbone_config)
        self.backbone.load_state_dict(checkpoint["model_state_dict"])
        if self.backbone.reconstruction_mode != "direct_scalar_mask":
            raise ValueError(
                "V16 requires the candidate-A direct scalar-mask backbone"
            )
        if self.backbone.phase_residual_scale != 0.0:
            raise ValueError(
                "V16 requires the candidate-A zero-phase-residual backbone"
            )
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
                f"V16 gate has {gate_parameters:,} parameters, exceeding "
                f"the {int(gate_parameter_cap):,} cap"
            )

        self.sample_rate = self.backbone.sample_rate
        self.n_fft = self.backbone.n_fft
        self.hop_length = self.backbone.hop_length
        self.win_length = self.backbone.win_length
        self.algorithmic_latency_samples = (
            self.backbone.algorithmic_latency_samples
        )

    def train(self, mode: bool = True) -> CausalOracleResidualGateV16:
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

    @staticmethod
    def frame_strength_to_samples(
        frame_strength: torch.Tensor,
        *,
        hop_length: int,
        output_length: int,
    ) -> torch.Tensor:
        if frame_strength.ndim != 3 or frame_strength.shape[-1] != 1:
            raise ValueError(
                "Expected frame strength shaped [batch, frames, 1], got "
                f"{tuple(frame_strength.shape)}"
            )
        if hop_length < 1 or output_length < 0:
            raise ValueError("hop_length must be positive and length non-negative")
        samples = frame_strength.transpose(1, 2).repeat_interleave(
            hop_length,
            dim=-1,
        )
        if samples.shape[-1] < output_length:
            samples = F.pad(
                samples,
                (0, output_length - samples.shape[-1]),
                mode="replicate",
            )
        return samples[..., :output_length]

    @staticmethod
    def blend_waveforms(
        noisy: torch.Tensor,
        base_enhanced: torch.Tensor,
        sample_strength: torch.Tensor,
    ) -> torch.Tensor:
        if noisy.shape != base_enhanced.shape:
            raise ValueError("Noisy and enhanced waveforms must have equal shape")
        if sample_strength.shape != noisy.shape:
            raise ValueError("Sample strength must match the waveform shape")
        return noisy + sample_strength * (base_enhanced - noisy)

    def _forward_waveform(
        self,
        noisy: torch.Tensor,
        pad_end: bool,
    ) -> CausalOracleResidualGateV16Output:
        if noisy.ndim != 3 or noisy.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, T], got {tuple(noisy.shape)}")
        with torch.no_grad():
            base = self.backbone._forward_waveform(
                noisy,
                pad_end=pad_end,
            )

        mixture_spectrum = (
            base.speech_spectrum + base.noise_spectrum
        ).detach()
        frame_strength = self.confidence_gate(
            base.continuous_noise_state.detach(),
            mixture_spectrum,
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
        )
        return CausalOracleResidualGateV16Output(**values)

    def forward(
        self,
        noisy: torch.Tensor,
    ) -> CausalOracleResidualGateV16Output:
        return self._forward_waveform(noisy, pad_end=True)

    init_stream_state = CausalAuxVQMambaV5.init_stream_state
    forward_chunk = CausalAuxVQMambaV5.forward_chunk
    flush = CausalAuxVQMambaV5.flush


__all__ = [
    "CausalOracleResidualGateV16",
    "CausalOracleResidualGateV16Output",
    "V5StreamState",
]
