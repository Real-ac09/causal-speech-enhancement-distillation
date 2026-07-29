from __future__ import annotations

import torch
import torch.nn.functional as F

from .causal_confidence_gate_v14 import CausalConfidenceGateV14
from .v8_native_streaming import V8NativeStreamer, V8NativeStreamState


class V14NativeStreamer(V8NativeStreamer):
    """Constant-work streaming inference for the causal confidence gate."""

    GATE_STATE_KEY = "confidence_gate"
    GATE_SPECTRUM_KEY = "confidence_gate_spectrum"

    def __init__(self, model: CausalConfidenceGateV14) -> None:
        if type(model) is not CausalConfidenceGateV14:
            raise TypeError(
                "V14NativeStreamer supports CausalConfidenceGateV14 exactly"
            )
        if model.training:
            raise ValueError("V14NativeStreamer requires model.eval()")
        self.gated_model = model
        super().__init__(model.backbone)

    def _gate_strength(
        self,
        continuous_noise: torch.Tensor,
        mixture_spectrum: torch.Tensor,
        state: V8NativeStreamState,
    ) -> torch.Tensor:
        gate = self.gated_model.confidence_gate
        previous = state.gru.get(self.GATE_SPECTRUM_KEY)
        if previous is None:
            summary_spectrum = mixture_spectrum.expand(-1, -1, 2)
        else:
            summary_spectrum = torch.cat(
                (previous, mixture_spectrum),
                dim=-1,
            )
        state.gru[self.GATE_SPECTRUM_KEY] = mixture_spectrum
        summaries = gate.mixture_summaries(summary_spectrum)[:, -1:]
        current = gate.noise_projection(gate.noise_norm(continuous_noise))
        current = F.silu(
            current + gate.summary_projection(summaries)
        )
        hidden = state.gru.get(self.GATE_STATE_KEY)
        current, hidden = gate.temporal(current, hidden)
        state.gru[self.GATE_STATE_KEY] = hidden
        fraction = torch.sigmoid(gate.output(current))
        return (
            gate.minimum_strength
            + (1.0 - gate.minimum_strength) * fraction
        )

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
        latent = (
            current
            + torch.tanh(model.bottleneck.temporal_scale) * temporal
        )
        base_spectrum = self._decode(
            latent,
            detail,
            magnitude,
            noisy_phase,
            state,
        )
        base_mask = (
            base_spectrum.abs() / magnitude.squeeze(-1)
        ).clamp_min(0.0)
        strength = self._gate_strength(
            continuous_noise,
            mixture_spectrum.unsqueeze(-1),
            state,
        ).squeeze(-1)
        return base_spectrum, mixture_spectrum, strength

    def _enhance_frame(
        self,
        audio_frame: torch.Tensor,
        state: V8NativeStreamState,
    ) -> torch.Tensor:
        base_spectrum, mixture_spectrum, strength = (
            self._base_spectrum_and_strength(audio_frame, state)
        )
        magnitude = mixture_spectrum.abs().clamp_min(1e-7)
        base_mask = (base_spectrum.abs() / magnitude).clamp_min(0.0)
        gated_mask = 1.0 + strength * (base_mask - 1.0)
        return gated_mask * mixture_spectrum


__all__ = ["V14NativeStreamer"]
