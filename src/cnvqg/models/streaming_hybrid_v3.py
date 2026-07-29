from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .streaming_hybrid_v2 import (
    CausalConv2d,
    StreamingHybridCNVQGV2,
    StreamingHybridV2Output,
)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class StreamingHybridCNVQGV3(StreamingHybridCNVQGV2):
    """Efficient bounded-lookahead waveform/TF enhancer.

    V2 copied the complete VQ vector to every time-frequency cell.  V3 first
    projects that vector to a small conditioning space.  This preserves the
    noise identity signal while greatly reducing TF activation memory and the
    cost of the first 2-D convolution.
    """

    PRESETS: Dict[str, Dict[str, int]] = {
        "teacher": {
            "encoder_channels": 160,
            "latent_dim": 320,
            "speech_dim": 320,
            "noise_dim": 128,
            "waveform_temporal_layers": 8,
            "codebook_size": 128,
            "tf_hidden_dim": 40,
            "tf_temporal_layers": 3,
            "frequency_subbands": 12,
        },
        "student": {
            "encoder_channels": 96,
            "latent_dim": 192,
            "speech_dim": 192,
            "noise_dim": 80,
            "waveform_temporal_layers": 6,
            "codebook_size": 96,
            "tf_hidden_dim": 32,
            "tf_temporal_layers": 2,
            "frequency_subbands": 8,
        },
        "tiny": {
            "encoder_channels": 64,
            "latent_dim": 128,
            "speech_dim": 128,
            "noise_dim": 64,
            "waveform_temporal_layers": 4,
            "codebook_size": 64,
            "tf_hidden_dim": 24,
            "tf_temporal_layers": 2,
            "frequency_subbands": 8,
        },
    }

    def __init__(
        self,
        variant: str = "student",
        tf_condition_dim: int = 12,
        encoder_type: str = "lookahead",
        decoder_type: str = "lookahead_transpose",
        **kwargs,
    ) -> None:
        super().__init__(
            variant=variant,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
            **kwargs,
        )
        if tf_condition_dim < 1:
            raise ValueError("tf_condition_dim must be positive")
        self.tf_condition_dim = int(tf_condition_dim)
        self.noise_tf_projection = nn.Sequential(
            nn.Conv1d(self.noise_dim, self.tf_condition_dim, 1),
            nn.SiLU(),
        )
        self.tf_input = nn.Sequential(
            CausalConv2d(5 + self.tf_condition_dim, self.tf_hidden_dim, (3, 3)),
            nn.GroupNorm(_group_count(self.tf_hidden_dim), self.tf_hidden_dim),
            nn.SiLU(),
        )
        if not self.enable_tf_refiner:
            self.noise_tf_projection.requires_grad_(False)
            self.tf_input.requires_grad_(False)

    def _align_noise_to_tf(
        self,
        quantized_noise: torch.Tensor,
        frames: int,
        bins: int,
    ) -> torch.Tensor:
        noise = self.noise_tf_projection(quantized_noise.transpose(1, 2))
        noise = torch.nn.functional.interpolate(noise, size=frames, mode="nearest")
        return noise.unsqueeze(2).expand(-1, -1, bins, -1)

    @property
    def tf_condition_compression(self) -> float:
        return self.noise_dim / self.tf_condition_dim


class StreamingHybridV3Teacher(StreamingHybridCNVQGV3):
    def __init__(self, **kwargs) -> None:
        super().__init__(variant="teacher", **kwargs)


class StreamingHybridV3Student(StreamingHybridCNVQGV3):
    def __init__(self, **kwargs) -> None:
        super().__init__(variant="student", **kwargs)


class StreamingHybridV3Tiny(StreamingHybridCNVQGV3):
    def __init__(self, **kwargs) -> None:
        super().__init__(variant="tiny", **kwargs)


__all__ = [
    "StreamingHybridCNVQGV3",
    "StreamingHybridV2Output",
    "StreamingHybridV3Teacher",
    "StreamingHybridV3Student",
    "StreamingHybridV3Tiny",
]
