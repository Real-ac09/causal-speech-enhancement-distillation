from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from cnvqg.models.decoder import NoiseConditionedDecoder
from cnvqg.models.encoder import ConvEncoder
from cnvqg.models.mamba_blocks import TemporalBlock
from cnvqg.models.noise_vq import VQOutput, VectorQuantizer


@dataclass
class CNVQGOutput:
    enhanced: torch.Tensor
    residual: torch.Tensor
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

        # Predict a small correction to the noisy waveform instead of fully
        # reconstructing speech from scratch. This makes the initial model much
        # safer because enhanced speech starts close to the noisy input.
        self.residual_scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, noisy: torch.Tensor) -> CNVQGOutput:
        if noisy.ndim != 3:
            raise ValueError(f"Expected noisy shape [B, 1, T], got {noisy.shape}")

        original_length = noisy.shape[-1]

        # The encoder downsamples by 16 overall. If the waveform length is not
        # divisible by 16, the decoder output becomes slightly shorter.
        # Pad before encoding, then crop the enhanced output back to the
        # original waveform length. This is important for full-utterance
        # evaluation where clips are not always exactly 4 seconds.
        downsample_factor = 16
        remainder = original_length % downsample_factor

        if remainder != 0:
            pad_amount = downsample_factor - remainder
            noisy_padded = F.pad(noisy, (0, pad_amount))
        else:
            noisy_padded = noisy

        z = self.encoder(noisy_padded)

        speech_latent = self.speech_projection(z)
        noise_latent = self.noise_projection(z)

        vq = self.noise_vq(noise_latent)

        speech_latent = self.temporal(speech_latent)

        residual = self.decoder(
            speech_latent=speech_latent,
            noise_latent=vq.quantized,
        )

        residual = residual[..., :original_length]

        enhanced = noisy_padded[..., :residual.shape[-1]] + self.residual_scale * residual
        enhanced = enhanced[..., :original_length]
        residual = residual[..., :original_length]

        return CNVQGOutput(
            enhanced=enhanced,
            residual=residual,
            speech_latent=speech_latent,
            noise_latent=noise_latent,
            quantized_noise=vq.quantized,
            vq=vq,
        )
