from __future__ import annotations

import torch
from torch import nn


class CausalConvBlock(nn.Module):
    """
    Lightweight fallback temporal block.

    This is not a true Mamba block. It is a stable causal Conv/TCN-style fallback
    so the model can run even when mamba-ssm is unavailable.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        self.left_padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            ),
            nn.PReLU(),
            nn.Conv1d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pad = nn.functional.pad(x, (self.left_padding, 0))
        return x + self.net(x_pad)


class TemporalBlock(nn.Module):
    """
    Temporal sequence block.

    Input:
        x: [B, C, T]

    If use_mamba=True and mamba_ssm is installed, it uses Mamba.
    Otherwise, it uses causal Conv blocks.
    """

    def __init__(
        self,
        channels: int = 128,
        hidden_channels: int = 128,
        num_layers: int = 4,
        use_mamba: bool = True,
    ) -> None:
        super().__init__()

        self.uses_mamba = False

        if use_mamba:
            try:
                from mamba_ssm import Mamba

                self.layers = nn.ModuleList(
                    [
                        Mamba(
                            d_model=channels,
                            d_state=16,
                            d_conv=4,
                            expand=2,
                        )
                        for _ in range(num_layers)
                    ]
                )
                self.uses_mamba = True
            except Exception:
                self.layers = self._make_fallback_layers(
                    channels=channels,
                    hidden_channels=hidden_channels,
                    num_layers=num_layers,
                )
        else:
            self.layers = self._make_fallback_layers(
                channels=channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
            )

    @staticmethod
    def _make_fallback_layers(
        channels: int,
        hidden_channels: int,
        num_layers: int,
    ) -> nn.ModuleList:
        return nn.ModuleList(
            [
                CausalConvBlock(
                    channels=channels,
                    hidden_channels=hidden_channels,
                    kernel_size=5,
                    dilation=2 ** (layer_idx % 4),
                )
                for layer_idx in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.uses_mamba:
            # Mamba expects [B, T, C]
            y = x.permute(0, 2, 1).contiguous()

            for layer in self.layers:
                y = y + layer(y)

            return y.permute(0, 2, 1).contiguous()

        for layer in self.layers:
            x = layer(x)

        return x
