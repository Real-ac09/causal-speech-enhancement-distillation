from __future__ import annotations

from typing import Literal

from .causal_aux_vq_mamba_v51 import CausalAuxVQMambaV51


class CausalV43RecoveryV45(CausalAuxVQMambaV51):
    """Minimal causal V4.3 recovery baseline.

    V4.5 deliberately reuses the validated V4.4 causal frontend, frame-local
    normalisation, full-resolution skip, and dual reconstruction heads. Its
    defaults remove the two pieces that did not earn their deployment cost:
    the second tied refinement pass and all VQ computation. Continuous causal
    noise conditioning remains on the enhancement path.
    """

    def __init__(
        self,
        *args,
        refinement_passes: int = 1,
        magnitude_mode: Literal[
            "bounded_mask", "log_ratio", "compressed_residual"
        ] = "bounded_mask",
        vq_mode: Literal["disabled", "train_only", "bounded_adapter"] = "disabled",
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            refinement_passes=refinement_passes,
            magnitude_mode=magnitude_mode,
            vq_mode=vq_mode,
            **kwargs,
        )


__all__ = ["CausalV43RecoveryV45"]
