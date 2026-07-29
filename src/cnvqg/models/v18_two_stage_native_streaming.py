from __future__ import annotations

from .causal_two_stage_utility_gate_v18 import (
    CausalTwoStageUtilityGateV18,
)
from .v17_statistics_native_streaming import V17StatisticsNativeStreamer
from .v8_native_streaming import V8NativeStreamer


class V18TwoStageNativeStreamer(V17StatisticsNativeStreamer):
    """Constant-memory native streamer for Recipe 8."""

    def __init__(self, model: CausalTwoStageUtilityGateV18) -> None:
        if type(model) is not CausalTwoStageUtilityGateV18:
            raise TypeError("V18TwoStageNativeStreamer requires Recipe 8")
        if model.training:
            raise ValueError("Native streaming requires model.eval()")
        self.gated_model = model
        V8NativeStreamer.__init__(self, model.backbone)


__all__ = ["V18TwoStageNativeStreamer"]
