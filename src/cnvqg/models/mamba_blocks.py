from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CausalConvBlock(nn.Module):
    """
    Lightweight causal temporal block.

    Input/output:
        [B, C, T] -> [B, C, T]
    """

    def __init__(
        self,
        channels: int,
        hidden_dim: int | None = None,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = channels

        self.kernel_size = kernel_size

        self.in_projection = nn.Conv1d(
            in_channels=channels,
            out_channels=hidden_dim,
            kernel_size=1,
        )

        self.depthwise = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            groups=hidden_dim,
        )

        self.out_projection = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=channels,
            kernel_size=1,
        )

        self.activation = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.in_projection(x)
        x = self.activation(x)

        # Left padding only = causal convolution.
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.depthwise(x)
        x = self.activation(x)

        x = self.out_projection(x)

        return x + residual


class TemporalBlock(nn.Module):
    """
    Temporal modelling block.

    If use_mamba=True, this tries to use mamba_ssm.Mamba.
    If Mamba is unavailable, it safely falls back to causal Conv1D blocks.
    """

    def __init__(
        self,
        channels: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        kernel_size: int = 5,
        use_mamba: bool = False,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.requested_mamba = use_mamba
        self.uses_mamba = False

        layers: list[nn.Module] = []

        if use_mamba:
            try:
                from mamba_ssm import Mamba

                layers = [
                    Mamba(
                        d_model=channels,
                        d_state=16,
                        d_conv=4,
                        expand=2,
                    )
                    for _ in range(num_layers)
                ]

                self.uses_mamba = True

            except Exception:
                layers = [
                    CausalConvBlock(
                        channels=channels,
                        hidden_dim=hidden_dim,
                        kernel_size=kernel_size,
                    )
                    for _ in range(num_layers)
                ]

                self.uses_mamba = False
        else:
            layers = [
                CausalConvBlock(
                    channels=channels,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                )
                for _ in range(num_layers)
            ]

            self.uses_mamba = False

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B, C, T], got {x.shape}")

        if self.uses_mamba:
            # Mamba expects [B, T, C].
            x = x.transpose(1, 2)

            for layer in self.layers:
                x = x + layer(x)

            return x.transpose(1, 2)

        for layer in self.layers:
            x = layer(x)

        return x
