from __future__ import annotations

import torch
import torch.nn.functional as F

from .causal_ordinal_residual_gate_v17 import CausalOrdinalResidualGateV17
from .v16_native_streaming import V16NativeStreamer
from .v8_native_streaming import V8NativeStreamer


class V17NativeStreamer(V16NativeStreamer):
    """Constant-work streamer for the V17 ordinal waveform residual path."""

    LAST_STRENGTH_KEY = "v17_last_strength"

    def __init__(self, model: CausalOrdinalResidualGateV17) -> None:
        if type(model) is not CausalOrdinalResidualGateV17:
            raise TypeError(
                "V17NativeStreamer supports "
                "CausalOrdinalResidualGateV17 exactly"
            )
        if model.training:
            raise ValueError("V17NativeStreamer requires model.eval()")
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
        probabilities = torch.softmax(gate.output(current).float(), dim=-1)
        return torch.sum(
            probabilities * gate.strength_grid.float(),
            dim=-1,
        )


__all__ = ["V17NativeStreamer"]
