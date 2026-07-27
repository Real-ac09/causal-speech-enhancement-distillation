from __future__ import annotations

import torch
from torch import nn

from .causal_aux_vq_mamba_v51 import FrameGroupNorm
from .causal_prototype_dual_axis_mamba_v9 import FrameChannelRMSNorm
from .causal_single_scale_mamba_v92 import CausalSingleScaleMambaV92


def _replace_frame_norms(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, FrameChannelRMSNorm):
            channels = int(child.weight.numel())
            replacement = FrameGroupNorm(channels)
            with torch.no_grad():
                replacement.norm.weight.copy_(child.weight)
                replacement.norm.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            _replace_frame_norms(child)


class CausalRepresentationNormMambaV96(CausalSingleScaleMambaV92):
    """V9.2 switches for isolated V8 representation and normalization tests."""

    def __init__(
        self,
        *args,
        v8_input_features: bool = False,
        frame_group_norm: bool = False,
        **kwargs,
    ) -> None:
        self.v8_input_features = bool(v8_input_features)
        self.frame_group_norm = bool(frame_group_norm)
        super().__init__(*args, **kwargs)
        if self.frame_group_norm:
            _replace_frame_norms(self)

    def _input_features(
        self,
        magnitude: torch.Tensor,
        compressed: torch.Tensor,
        unit: torch.Tensor,
    ) -> torch.Tensor:
        if self.v8_input_features:
            return torch.stack((compressed, unit.real, unit.imag), dim=1)
        return super()._input_features(magnitude, compressed, unit)


__all__ = ["CausalRepresentationNormMambaV96"]
