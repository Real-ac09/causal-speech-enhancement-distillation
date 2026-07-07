from __future__ import annotations

import torch
from torch import nn


class NoiseConditionedDecoder(nn.Module):
    """
    Decoder conditioned by quantised noise latents using FiLM-style modulation.

    Inputs:
        speech_latent: [B, speech_dim, T]
        noise_latent:  [B, noise_dim, T]

    Output:
        enhanced waveform: [B, 1, T * 16]
    """

    def __init__(
        self,
        speech_dim: int = 128,
        noise_dim: int = 64,
        base_channels: int = 64,
        out_channels: int = 1,
    ) -> None:
        super().__init__()

        self.noise_to_film = nn.Conv1d(noise_dim, speech_dim * 2, kernel_size=1)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(speech_dim, base_channels * 2, kernel_size=8, stride=2, padding=3),
            nn.PReLU(),

            nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=8, stride=2, padding=3),
            nn.PReLU(),

            nn.ConvTranspose1d(base_channels, base_channels, kernel_size=8, stride=2, padding=3),
            nn.PReLU(),

            nn.ConvTranspose1d(base_channels, out_channels, kernel_size=8, stride=2, padding=3),
            nn.Tanh(),
        )

    def forward(
        self,
        speech_latent: torch.Tensor,
        noise_latent: torch.Tensor,
    ) -> torch.Tensor:
        gamma_beta = self.noise_to_film(noise_latent)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=1)

        conditioned = speech_latent * (1.0 + torch.tanh(gamma)) + beta

        return self.decoder(conditioned)
