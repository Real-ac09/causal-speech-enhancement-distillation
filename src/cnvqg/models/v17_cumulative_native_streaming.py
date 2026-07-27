from __future__ import annotations

import torch
import torch.nn.functional as F

from .causal_cumulative_ordinal_gate_v17 import (
    CausalCumulativeOrdinalGateV17,
)
from .v17_native_streaming import V17NativeStreamer
from .v8_native_streaming import V8NativeStreamer


class V17CumulativeNativeStreamer(V17NativeStreamer):
    """Constant-work streamer for the recipe-3 cumulative controller."""

    LAST_STRENGTH_KEY = "v17_cumulative_last_strength"

    def __init__(self, model: CausalCumulativeOrdinalGateV17) -> None:
        if type(model) is not CausalCumulativeOrdinalGateV17:
            raise TypeError(
                "V17CumulativeNativeStreamer requires the cumulative model"
            )
        if model.training:
            raise ValueError("Native streaming requires model.eval()")
        self.gated_model = model
        V8NativeStreamer.__init__(self, model.backbone)

    def _gate_strength(
        self,
        continuous_noise: torch.Tensor,
        mixture_spectrum: torch.Tensor,
        state,
    ) -> torch.Tensor:
        gate = self.gated_model.confidence_gate
        previous = state.gru.get(self.GATE_SPECTRUM_KEY)
        summary_spectrum = (
            mixture_spectrum.expand(-1, -1, 2)
            if previous is None
            else torch.cat((previous, mixture_spectrum), dim=-1)
        )
        state.gru[self.GATE_SPECTRUM_KEY] = mixture_spectrum
        summaries = gate.mixture_summaries(summary_spectrum)[:, -1:]
        current = gate.noise_projection(gate.noise_norm(continuous_noise))
        current = F.silu(current + gate.summary_projection(summaries))
        hidden = state.gru.get(self.GATE_STATE_KEY)
        current, hidden = gate.temporal(current, hidden)
        state.gru[self.GATE_STATE_KEY] = hidden
        strength, _, _ = gate.details_from_hidden(current)
        return strength.squeeze(-1)


__all__ = ["V17CumulativeNativeStreamer"]
