from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cnvqg.models.decoder import NoiseConditionedDecoder
from cnvqg.models.encoder import ConvEncoder
from cnvqg.models.mamba_blocks import TemporalBlock
from cnvqg.models.noise_vq import VQOutput, VectorQuantizer


@dataclass
class CNVQGOutput:
    enhanced: torch.Tensor
    speech_latent: torch.Tensor
    noise_latent: torch.Tensor
    quantized_noise: torch.Tensor
    vq: VQOutput


class CNVQGModel(nn.Module):
    """
    CN-VQG: Causal Noise-State Vector-Quantised Gating Network.

    First working version:
        waveform
        -> shared encoder
        -> speech branch
        -> noise branch
        -> VQ noise state
        -> temporal speech block
        -> noise-conditioned decoder
        -> enhanced waveform
    """

    def __init__(
        self,
        encoder_channels: int = 64,
        latent_dim: int = 128,
        speech_dim: int = 128,
        noise_dim: int = 64,
        codebook_size: int = 256,
        temporal_hidden_dim: int = 128,
        temporal_layers: int = 4,
        use_mamba: bool = True,
    ) -> None:
        super().__init__()

        self.encoder = ConvEncoder(
            in_channels=1,
            base_channels=encoder_channels,
            latent_dim=latent_dim,
        )

        self.speech_projection = nn.Conv1d(latent_dim, speech_dim, kernel_size=1)
        self.noise_projection = nn.Conv1d(latent_dim, noise_dim, kernel_size=1)

        self.noise_vq = VectorQuantizer(
            num_codes=codebook_size,
            code_dim=noise_dim,
            commitment_weight=0.25,
        )

        self.temporal = TemporalBlock(
            channels=speech_dim,
            hidden_channels=temporal_hidden_dim,
            num_layers=temporal_layers,
            use_mamba=use_mamba,
        )

        self.decoder = NoiseConditionedDecoder(
            speech_dim=speech_dim,
            noise_dim=noise_dim,
            base_channels=encoder_channels,
            out_channels=1,
        )

    def forward(self, noisy: torch.Tensor) -> CNVQGOutput:
        if noisy.ndim != 3:
            raise ValueError(f"Expected noisy shape [B, 1, T], got {noisy.shape}")

        original_length = noisy.shape[-1]

        z = self.encoder(noisy)

        speech_latent = self.speech_projection(z)
        noise_latent = self.noise_projection(z)

        vq = self.noise_vq(noise_latent)

        speech_latent = self.temporal(speech_latent)

        enhanced = self.decoder(
            speech_latent=speech_latent,
            noise_latent=vq.quantized,
        )

        enhanced = enhanced[..., :original_length]

        return CNVQGOutput(
            enhanced=enhanced,
            speech_latent=speech_latent,
            noise_latent=noise_latent,
            quantized_noise=vq.quantized,
            vq=vq,
        )
