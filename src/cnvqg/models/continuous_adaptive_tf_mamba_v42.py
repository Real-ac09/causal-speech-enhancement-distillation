from __future__ import annotations

from .noise_adaptive_tf_mamba_v41 import (
    NoiseAdaptiveTFMambaV41,
    NoiseAdaptiveTFV41Output,
)


class ContinuousAdaptiveTFMambaV42(NoiseAdaptiveTFMambaV41):
    """V4.2: continuous segment conditioning with a compact adaptive depth."""

    def __init__(
        self,
        *args,
        max_iterations: int = 3,
        use_noise_codebook: bool = False,
        condition_dynamics: bool = True,
        adaptive_iterations: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            max_iterations=max_iterations,
            use_noise_codebook=use_noise_codebook,
            condition_dynamics=condition_dynamics,
            adaptive_iterations=adaptive_iterations,
            **kwargs,
        )


ContinuousAdaptiveTFV42Output = NoiseAdaptiveTFV41Output

__all__ = ["ContinuousAdaptiveTFMambaV42", "ContinuousAdaptiveTFV42Output"]
