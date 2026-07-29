from __future__ import annotations

import torch
import torch.nn.functional as F

from .causal_statistics_utility_gate_v17 import (
    CausalStatisticsUtilityGateV17,
)
from .v17_feature_native_streaming import V17FeatureNativeStreamer
from .v8_native_streaming import V8NativeStreamer, V8NativeStreamState


class V17StatisticsNativeStreamer(V17FeatureNativeStreamer):
    """Constant-memory native streamer for Recipe 6."""

    STAT_COUNT_KEY = "v17_statistics_count"
    STAT_SUM_KEY = "v17_statistics_sum"
    STAT_SQUARE_SUM_KEY = "v17_statistics_square_sum"
    STAT_MIN_KEY = "v17_statistics_min"
    STAT_MAX_KEY = "v17_statistics_max"
    STAT_PREVIOUS_KEY = "v17_statistics_previous"
    STAT_CHANGE_SUM_KEY = "v17_statistics_change_sum"

    def __init__(self, model: CausalStatisticsUtilityGateV17) -> None:
        if type(model) is not CausalStatisticsUtilityGateV17:
            raise TypeError("V17StatisticsNativeStreamer requires Recipe 6")
        if model.training:
            raise ValueError("Native streaming requires model.eval()")
        self.gated_model = model
        V8NativeStreamer.__init__(self, model.backbone)

    def _update_statistics(
        self,
        features: torch.Tensor,
        state: V8NativeStreamState,
    ) -> torch.Tensor:
        values = features.float()
        previous = state.gru.get(self.STAT_PREVIOUS_KEY)
        count = state.gru.get(self.STAT_COUNT_KEY)
        if count is None:
            count = values.new_zeros(values.shape[0], 1, 1)
            running_sum = torch.zeros_like(values)
            square_sum = torch.zeros_like(values)
            minimum = values
            maximum = values
            change_sum = torch.zeros_like(values)
            change = torch.zeros_like(values)
        else:
            running_sum = state.gru[self.STAT_SUM_KEY]
            square_sum = state.gru[self.STAT_SQUARE_SUM_KEY]
            minimum = torch.minimum(
                state.gru[self.STAT_MIN_KEY],
                values,
            )
            maximum = torch.maximum(
                state.gru[self.STAT_MAX_KEY],
                values,
            )
            change_sum = state.gru[self.STAT_CHANGE_SUM_KEY]
            change = (values - previous).abs()
        count = count + 1.0
        running_sum = running_sum + values
        square_sum = square_sum + values.square()
        change_sum = change_sum + change
        mean = running_sum / count
        standard_deviation = (
            (square_sum / count - mean.square()).clamp_min(0.0) + 1e-6
        ).sqrt()
        change_rate = change_sum / (count - 1.0).clamp_min(1.0)
        state.gru[self.STAT_COUNT_KEY] = count
        state.gru[self.STAT_SUM_KEY] = running_sum
        state.gru[self.STAT_SQUARE_SUM_KEY] = square_sum
        state.gru[self.STAT_MIN_KEY] = minimum
        state.gru[self.STAT_MAX_KEY] = maximum
        state.gru[self.STAT_PREVIOUS_KEY] = values
        state.gru[self.STAT_CHANGE_SUM_KEY] = change_sum
        return torch.cat(
            (mean, standard_deviation, minimum, maximum, change_rate),
            dim=-1,
        ).to(features.dtype)

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
        normalised_noise = gate.noise_norm(continuous_noise)
        statistics = self._update_statistics(
            torch.cat((normalised_noise, summaries), dim=-1),
            state,
        )
        current = gate.noise_projection(normalised_noise)
        current = F.silu(
            current
            + gate.summary_projection(summaries)
            + gate.statistics_projection(statistics)
        )
        hidden = state.gru.get(self.GATE_STATE_KEY)
        current, hidden = gate.temporal(current, hidden)
        state.gru[self.GATE_STATE_KEY] = hidden
        strength = gate.details_from_hidden(current)[0]
        return strength.squeeze(-1)


__all__ = ["V17StatisticsNativeStreamer"]
