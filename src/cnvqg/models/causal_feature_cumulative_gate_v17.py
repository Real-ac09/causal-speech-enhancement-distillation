from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .causal_cumulative_ordinal_gate_v17 import (
    CausalCumulativeOrdinalGateV17,
    CausalCumulativeOrdinalGateV17Output,
    CausalCumulativeOrdinalStrengthGate,
)


class CausalFeatureCumulativeStrengthGate(
    CausalCumulativeOrdinalStrengthGate
):
    """Recipe-3 ordinal gate with richer causal mixture/backbone features."""

    FEATURE_NAMES = (
        "log_rms",
        "spectral_flatness",
        "high_band_ratio",
        "spectral_flux",
        "low_band_ratio",
        "mid_band_ratio",
        "spectral_centroid",
        "base_mask_mean",
        "base_mask_std",
        "heavy_attenuation_fraction",
        "estimated_snr",
        "estimated_noise_fraction",
    )

    def __init__(
        self,
        noise_dim: int,
        hidden_dim: int = 16,
        strength_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        initial_cutpoints: Sequence[float] = (-1.5, -0.5, 0.5, 1.5),
    ) -> None:
        super().__init__(
            noise_dim=noise_dim,
            hidden_dim=hidden_dim,
            strength_grid=strength_grid,
            initial_cutpoints=initial_cutpoints,
        )
        self.summary_projection = nn.Linear(
            len(self.FEATURE_NAMES),
            self.hidden_dim,
        )

    @classmethod
    def rich_summaries(
        cls,
        mixture_spectrum: torch.Tensor,
        base_spectrum: torch.Tensor,
    ) -> torch.Tensor:
        if mixture_spectrum.ndim != 3 or base_spectrum.ndim != 3:
            raise ValueError("Spectra must be [batch, bins, frames]")
        bins = min(mixture_spectrum.shape[-2], base_spectrum.shape[-2])
        frames = min(mixture_spectrum.shape[-1], base_spectrum.shape[-1])
        mixture = mixture_spectrum[:, :bins, :frames]
        base = base_spectrum[:, :bins, :frames]
        original = cls.mixture_summaries(mixture)

        magnitude = mixture.abs().float().clamp_min(1e-7)
        power = magnitude.square()
        total_power = power.sum(dim=-2).clamp_min(1e-7)
        normalised_frequency = torch.linspace(
            0.0,
            1.0,
            bins,
            device=magnitude.device,
            dtype=magnitude.dtype,
        )
        low = normalised_frequency <= 0.125
        mid = (
            (normalised_frequency > 0.125)
            & (normalised_frequency <= 0.5)
        )
        low_ratio = power[:, low].sum(dim=-2) / total_power
        mid_ratio = power[:, mid].sum(dim=-2) / total_power
        centroid = (
            magnitude * normalised_frequency[None, :, None]
        ).sum(dim=-2) / magnitude.sum(dim=-2).clamp_min(1e-7)

        base_magnitude = base.abs().float()
        base_mask = (base_magnitude / magnitude).clamp(0.0, 2.0)
        mask_mean = base_mask.mean(dim=-2) / 2.0
        mask_std = base_mask.std(dim=-2, unbiased=False).clamp_max(1.0)
        heavy_attenuation = (base_mask < 0.5).float().mean(dim=-2)

        noise_magnitude = (mixture - base).abs().float()
        speech_energy = base_magnitude.square().sum(dim=-2).clamp_min(1e-7)
        noise_energy = noise_magnitude.square().sum(dim=-2).clamp_min(1e-7)
        estimated_snr = (
            10.0 * torch.log10(speech_energy / noise_energy)
        ).clamp(-30.0, 30.0) / 30.0
        noise_fraction = noise_energy / (
            speech_energy + noise_energy
        ).clamp_min(1e-7)

        additional = torch.stack(
            (
                low_ratio,
                mid_ratio,
                centroid,
                mask_mean,
                mask_std,
                heavy_attenuation,
                estimated_snr,
                noise_fraction,
            ),
            dim=-1,
        )
        return torch.cat((original, additional), dim=-1)

    def forward_with_details(
        self,
        continuous_noise_state: torch.Tensor,
        mixture_spectrum: torch.Tensor,
        base_spectrum: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        strength, _, _ = self.forward_with_details(
            continuous_noise_state,
            mixture_spectrum,
            base_spectrum,
        )
        return strength


class CausalFeatureCumulativeGateV17(CausalCumulativeOrdinalGateV17):
    """Recipe-4 controller with explicit causal acoustic descriptors."""

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
            initial_cutpoints=initial_cutpoints,
            gate_parameter_cap=gate_parameter_cap,
        )
        self.confidence_gate = CausalFeatureCumulativeStrengthGate(
            noise_dim=self.backbone.noise_dim,
            hidden_dim=gate_hidden_dim,
            strength_grid=strength_grid,
            initial_cutpoints=initial_cutpoints,
        )
        if self.gate_parameter_count() > int(gate_parameter_cap):
            raise ValueError("Recipe-4 gate exceeds its parameter cap")

    def train(self, mode: bool = True) -> CausalFeatureCumulativeGateV17:
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
        base_spectrum = base.speech_spectrum.detach()
        frame_strength, ordinal_logits, probabilities = (
            self.confidence_gate.forward_with_details(
                base.continuous_noise_state.detach(),
                mixture_spectrum,
                base_spectrum,
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
    "CausalFeatureCumulativeGateV17",
    "CausalFeatureCumulativeStrengthGate",
]
