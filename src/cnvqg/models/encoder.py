from __future__ import annotations

import torch
from torch import nn


class ConvEncoder(nn.Module):
    """
    Causal-ish 1D convolutional encoder for waveform speech enhancement.

    Input:
        x: [B, 1, T]

    Output:
        z: [B, latent_dim, T / 16]
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=8, stride=2, padding=3),
            nn.PReLU(),

            nn.Conv1d(base_channels, base_channels, kernel_size=8, stride=2, padding=3),
            nn.PReLU(),

            nn.Conv1d(base_channels, base_channels * 2, kernel_size=8, stride=2, padding=3),
            nn.PReLU(),

            nn.Conv1d(base_channels * 2, latent_dim, kernel_size=8, stride=2, padding=3),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
