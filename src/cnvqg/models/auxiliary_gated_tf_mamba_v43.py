from __future__ import annotations

from .continuous_adaptive_tf_mamba_v42 import (
    ContinuousAdaptiveTFMambaV42,
    ContinuousAdaptiveTFV42Output,
)


class AuxiliaryGatedTFMambaV43(ContinuousAdaptiveTFMambaV42):
    """V4.3: continuous conditioning with an optional learned VQ residual."""

    def __init__(
        self,
        *args,
        use_noise_codebook: bool = True,
        vq_gate_learnable: bool = True,
        vq_gate_initial: float = 0.01,
        vq_update_codebook: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            use_noise_codebook=use_noise_codebook,
            vq_gate_learnable=vq_gate_learnable,
            vq_gate_initial=vq_gate_initial,
            vq_update_codebook=vq_update_codebook,
            **kwargs,
        )


AuxiliaryGatedTFV43Output = ContinuousAdaptiveTFV42Output

__all__ = ["AuxiliaryGatedTFMambaV43", "AuxiliaryGatedTFV43Output"]
