from __future__ import annotations

import torch
import torch.nn.functional as F

from .causal_utility_safety_gate_v17 import CausalUtilitySafetyGateV17
from .v17_feature_native_streaming import V17FeatureNativeStreamer
from .v8_native_streaming import V8NativeStreamer, V8NativeStreamState


class V17UtilityNativeStreamer(V17FeatureNativeStreamer):
    """Constant-work native path for Recipe 5's utility/safety selector."""

    def __init__(self, model: CausalUtilitySafetyGateV17) -> None:
        if type(model) is not CausalUtilitySafetyGateV17:
            raise TypeError("V17UtilityNativeStreamer requires Recipe 5")
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
        strength = gate.details_from_hidden(current)[0]
        return strength.squeeze(-1)


__all__ = ["V17UtilityNativeStreamer"]
