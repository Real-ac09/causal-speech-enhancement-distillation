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


class IdentityTemporal(nn.Module):
    """
    Identity temporal block used for no-temporal ablations.
    """

    uses_mamba = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class CNVQGModel(nn.Module):
    """
    CN-VQG speech enhancement model.

    Main path:
        noisy waveform
        -> shared causal encoder
        -> speech/noise latent split
        -> optional VQ on noise latent
        -> optional temporal modelling on speech latent
        -> optional noise-conditioned decoder
        -> residual or direct waveform output

    Important ablation switches:
        use_residual:
            If true, enhanced = noisy + residual_scale * decoder_output.
            If false, enhanced = decoder_output.

        use_vq:
            If true, quantise the noise latent through a VQ codebook.
            If false, use the continuous noise latent directly.

        use_noise_conditioning:
            If true, pass noise latent to the decoder.
            If false, pass zeros to remove noise-specific conditioning.

        use_temporal:
            If true, use the temporal block.
            If false, skip temporal modelling.

        use_mamba:
            If true, TemporalBlock will try to use Mamba.
            If Mamba is unavailable, TemporalBlock falls back to causal Conv.
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
        use_mamba: bool = False,
        use_residual: bool = True,
        use_vq: bool = True,
        use_noise_conditioning: bool = True,
        use_temporal: bool = True,
        residual_scale_init: float = 0.05,
        learn_residual_scale: bool = True,
    ) -> None:
        super().__init__()

        self.encoder_channels = encoder_channels
        self.latent_dim = latent_dim
        self.speech_dim = speech_dim
        self.noise_dim = noise_dim
        self.codebook_size = codebook_size

        self.use_residual = use_residual
        self.use_vq = use_vq
        self.use_noise_conditioning = use_noise_conditioning
        self.use_temporal = use_temporal

        self.encoder = ConvEncoder(
            in_channels=1,
            base_channels=encoder_channels,
            latent_dim=latent_dim,
        )

        self.speech_projection = nn.Conv1d(
            in_channels=latent_dim,
            out_channels=speech_dim,
            kernel_size=1,
        )

        self.noise_projection = nn.Conv1d(
            in_channels=latent_dim,
            out_channels=noise_dim,
            kernel_size=1,
        )

        self.noise_vq = VectorQuantizer(
            num_codes=codebook_size,
            code_dim=noise_dim,
        )

        if use_temporal:
            self.temporal = TemporalBlock(
                channels=speech_dim,
                hidden_dim=temporal_hidden_dim,
                num_layers=temporal_layers,
                use_mamba=use_mamba,
            )
        else:
            self.temporal = IdentityTemporal()

        self.decoder = NoiseConditionedDecoder(
            speech_dim=speech_dim,
            noise_dim=noise_dim,
            base_channels=encoder_channels,
            out_channels=1,
        )

        # Used only when use_residual=True.
        # For standard residual training this can be learnable.
        # For Mamba ablations, fixing it prevents collapse to exact identity.
        if learn_residual_scale:
            self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        else:
            self.register_buffer("residual_scale", torch.tensor(float(residual_scale_init)))

    def _make_no_vq_output(self, noise_latent: torch.Tensor) -> VQOutput:
        batch_size, _, latent_frames = noise_latent.shape

        zero = noise_latent.new_tensor(0.0)

        indices = torch.full(
            size=(batch_size, latent_frames),
            fill_value=-1,
            dtype=torch.long,
            device=noise_latent.device,
        )

        return VQOutput(
            quantized=noise_latent,
            indices=indices,
            loss=zero,
            commitment_loss=zero,
            codebook_loss=zero,
            perplexity=zero,
        )

    def forward(self, noisy: torch.Tensor) -> CNVQGOutput:
        if noisy.ndim != 3:
            raise ValueError(f"Expected noisy shape [B, 1, T], got {noisy.shape}")

        original_length = noisy.shape[-1]

        # The encoder downsamples by 16 overall. If the waveform length is not
        # divisible by 16, pad before encoding and crop back after decoding.
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

        if self.use_vq:
            vq = self.noise_vq(noise_latent)
            quantized_noise = vq.quantized
        else:
            vq = self._make_no_vq_output(noise_latent)
            quantized_noise = noise_latent

        if self.use_temporal:
            speech_latent = self.temporal(speech_latent)

        if self.use_noise_conditioning:
            decoder_noise = quantized_noise
        else:
            decoder_noise = torch.zeros_like(quantized_noise)

        residual = self.decoder(
            speech_latent=speech_latent,
            noise_latent=decoder_noise,
        )

        residual = residual[..., : noisy_padded.shape[-1]]

        if self.use_residual:
            enhanced = noisy_padded[..., : residual.shape[-1]] + self.residual_scale * residual
        else:
            enhanced = residual

        enhanced = enhanced[..., :original_length]
        residual = residual[..., :original_length]

        return CNVQGOutput(
            enhanced=enhanced,
            residual=residual,
            speech_latent=speech_latent,
            noise_latent=noise_latent,
            quantized_noise=quantized_noise,
            vq=vq,
        )
