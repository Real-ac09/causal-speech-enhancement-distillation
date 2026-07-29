from __future__ import annotations

import torch
from torch import nn

from .predictive_noise_vq_mamba_v8 import PredictiveNoiseVQMambaV8


class LowRankGlobalFrequencyAttention(nn.Module):
    """Full-band attention within each frame through a channel bottleneck.

    Frequency attention is bidirectional, but time frames are folded into the
    batch dimension and never share an attention sequence. It therefore adds
    global spectral context without adding future-time context.

    The separation of temporal recurrence and frequency attention is motivated
    by FastEnhancer (Ahn et al., ICASSP 2026). This compact residual block and
    its placement in the local V8 enhancement path are repository-specific.
    """

    def __init__(
        self,
        channels: int,
        attention_dim: int = 40,
        heads: int = 4,
        expansion: int = 2,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if attention_dim <= 0 or attention_dim % heads:
            raise ValueError("attention_dim must be positive and divisible by heads")
        if expansion < 1:
            raise ValueError("expansion must be positive")
        if not 0.0 < residual_scale <= 1.0:
            raise ValueError("residual_scale must be in (0, 1]")

        hidden = attention_dim * expansion
        self.norm = nn.LayerNorm(channels)
        self.input_projection = nn.Linear(channels, attention_dim)
        self.attention_norm = nn.LayerNorm(attention_dim)
        self.attention = nn.MultiheadAttention(
            attention_dim, heads, dropout=0.0, batch_first=True
        )
        self.feed_forward_norm = nn.LayerNorm(attention_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(attention_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, attention_dim),
        )
        self.output_projection = nn.Linear(attention_dim, channels)
        self.residual_scale = float(residual_scale)

        # A small non-zero branch permits gradients throughout the block from
        # the first update while remaining close to the V8 control at start-up.
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, bins, frames = features.shape
        tokens = features.permute(0, 3, 2, 1).reshape(batch * frames, bins, channels)
        current = self.input_projection(self.norm(tokens))
        attention_input = self.attention_norm(current)
        attended, _ = self.attention(
            attention_input, attention_input, attention_input, need_weights=False
        )
        current = current + attended
        current = current + self.feed_forward(self.feed_forward_norm(current))
        correction = self.output_projection(current)
        correction = correction.view(batch, frames, bins, channels).permute(0, 3, 2, 1)
        return features + self.residual_scale * correction


class FrequencyMixedV8Bottleneck(nn.Module):
    """Wrap V8's temporal bottleneck with an optional within-frame mixer."""

    def __init__(
        self,
        temporal: nn.Module,
        channels: int,
        *,
        enabled: bool,
        attention_dim: int,
        heads: int,
        expansion: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.temporal_bottleneck = temporal
        self.frequency_mixer = (
            LowRankGlobalFrequencyAttention(
                channels,
                attention_dim=attention_dim,
                heads=heads,
                expansion=expansion,
                residual_scale=residual_scale,
            )
            if enabled
            else nn.Identity()
        )

    def forward(self, features: torch.Tensor, noise_state: torch.Tensor) -> torch.Tensor:
        return self.frequency_mixer(self.temporal_bottleneck(features, noise_state))


class CausalGlobalFrequencyMambaV10(PredictiveNoiseVQMambaV8):
    """The strongest V8 magnitude path plus efficient global frequency context."""

    def __init__(
        self,
        *args,
        use_global_frequency_attention: bool = True,
        frequency_attention_dim: int = 40,
        frequency_attention_heads: int = 4,
        frequency_attention_expansion: int = 2,
        frequency_residual_scale: float = 0.1,
        **kwargs,
    ) -> None:
        if bool(kwargs.get("auxiliary_vq", False)):
            raise ValueError("V10 isolates enhancement and requires auxiliary_vq=false")
        enforce_parameter_cap = bool(kwargs.get("enforce_parameter_cap", True))
        kwargs["auxiliary_vq"] = False
        kwargs["enforce_parameter_cap"] = False
        super().__init__(*args, **kwargs)

        # These heads are unreachable when auxiliary VQ is disabled. Removing
        # their parameters preserves V8 enhancement exactly and releases 27k
        # parameters for the frequency path.
        self.noise_predictor = nn.Identity()
        self.prototype_predictor = nn.Identity()
        original_bottleneck = self.bottleneck
        self.bottleneck = FrequencyMixedV8Bottleneck(
            original_bottleneck,
            self.channels,
            enabled=bool(use_global_frequency_attention),
            attention_dim=frequency_attention_dim,
            heads=frequency_attention_heads,
            expansion=frequency_attention_expansion,
            residual_scale=frequency_residual_scale,
        )
        self.temporal = original_bottleneck.temporal
        self.use_global_frequency_attention = bool(use_global_frequency_attention)

        count = self.parameter_count()
        if enforce_parameter_cap and count > self.parameter_cap:
            raise ValueError(
                f"V10 {self.variant} has {count:,} parameters, exceeding {self.parameter_cap:,}"
            )


__all__ = ["CausalGlobalFrequencyMambaV10", "LowRankGlobalFrequencyAttention"]
