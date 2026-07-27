from __future__ import annotations

import torch
import torch.nn.functional as F

from .causal_feature_cumulative_gate_v17 import (
    CausalFeatureCumulativeGateV17,
)
from .v16_native_streaming import V16NativeStreamer
from .v17_cumulative_native_streaming import V17CumulativeNativeStreamer
from .v8_native_streaming import V8NativeStreamer, V8NativeStreamState


class V17FeatureNativeStreamer(V17CumulativeNativeStreamer):
    """Constant-work native path for recipe 4's rich causal features."""

    LAST_STRENGTH_KEY = "v17_feature_last_strength"
    BASE_SPECTRUM_KEY = "v17_feature_base_spectrum"

    def __init__(self, model: CausalFeatureCumulativeGateV17) -> None:
        if type(model) is not CausalFeatureCumulativeGateV17:
            raise TypeError("V17FeatureNativeStreamer requires recipe 4")
        if model.training:
            raise ValueError("Native streaming requires model.eval()")
        self.gated_model = model
        V8NativeStreamer.__init__(self, model.backbone)

    def _rich_gate_strength(
        self,
        continuous_noise: torch.Tensor,
        mixture_spectrum: torch.Tensor,
        base_spectrum: torch.Tensor,
        state: V8NativeStreamState,
    ) -> torch.Tensor:
        gate = self.gated_model.confidence_gate
        previous_mixture = state.gru.get(self.GATE_SPECTRUM_KEY)
        previous_base = state.gru.get(self.BASE_SPECTRUM_KEY)
        if previous_mixture is None:
            summary_mixture = mixture_spectrum.expand(-1, -1, 2)
            summary_base = base_spectrum.expand(-1, -1, 2)
        else:
            summary_mixture = torch.cat(
                (previous_mixture, mixture_spectrum),
                dim=-1,
            )
            summary_base = torch.cat(
                (previous_base, base_spectrum),
                dim=-1,
            )
        state.gru[self.GATE_SPECTRUM_KEY] = mixture_spectrum
        state.gru[self.BASE_SPECTRUM_KEY] = base_spectrum
        summaries = gate.rich_summaries(
            summary_mixture,
            summary_base,
        )[:, -1:]
        current = gate.noise_projection(gate.noise_norm(continuous_noise))
        current = F.silu(current + gate.summary_projection(summaries))
        hidden = state.gru.get(self.GATE_STATE_KEY)
        current, hidden = gate.temporal(current, hidden)
        state.gru[self.GATE_STATE_KEY] = hidden
        strength, _, _ = gate.details_from_hidden(current)
        return strength.squeeze(-1)

    def _base_spectrum_and_strength(
        self,
        audio_frame: torch.Tensor,
        state: V8NativeStreamState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model = self.model
        window = model.analysis_window.to(
            device=audio_frame.device,
            dtype=audio_frame.dtype,
        )
        mixture_spectrum = torch.fft.rfft(
            audio_frame.squeeze(1) * window,
            n=model.n_fft,
            dim=-1,
        )
        magnitude = mixture_spectrum.abs().clamp_min(1e-7).unsqueeze(-1)
        noisy_phase = torch.angle(mixture_spectrum).unsqueeze(-1)
        inputs = torch.stack(
            (
                magnitude.squeeze(1).pow(model.magnitude_power),
                torch.cos(noisy_phase.squeeze(1)),
                torch.sin(noisy_phase.squeeze(1)),
            ),
            dim=1,
        )
        detail = self._sequence(
            model.encoder.detail,
            inputs,
            state,
            "encoder.detail",
        )
        encoded = self._sequence(
            model.encoder.down,
            detail,
            state,
            "encoder.down",
        )
        continuous_noise = self._rolling_noise(encoded, state)
        local = self._sequence(
            model.bottleneck.local,
            encoded,
            state,
            "bottleneck.local",
        )
        current = encoded + local
        condition = model.bottleneck.noise_condition(
            continuous_noise
        ).transpose(1, 2)
        scale, shift = condition.chunk(2, dim=1)
        current = current * (
            1.0 + 0.05 * torch.tanh(scale[:, :, None])
        )
        current = current + 0.05 * shift[:, :, None]
        temporal = self._temporal(current, state)
        latent = current + torch.tanh(
            model.bottleneck.temporal_scale
        ) * temporal
        base_spectrum = self._decode(
            latent,
            detail,
            magnitude,
            noisy_phase,
            state,
        )
        strength = self._rich_gate_strength(
            continuous_noise,
            mixture_spectrum.unsqueeze(-1),
            base_spectrum.unsqueeze(-1),
            state,
        )
        return base_spectrum, mixture_spectrum, strength


__all__ = ["V17FeatureNativeStreamer"]
