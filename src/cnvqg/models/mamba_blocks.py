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

        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.depthwise(x)
        x = self.activation(x)

        x = self.out_projection(x)

        return x + residual


class MambaResidualBlock(nn.Module):
    """
    Stabilised Mamba residual block.

    Uses:
    - LayerNorm before Mamba
    - small learnable layer scale
    - residual connection

    Input/output:
        [B, C, T] -> [B, C, T]
    """

    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        layer_scale_init: float = 0.1,
    ) -> None:
        super().__init__()

        from mamba_ssm import Mamba

        self.norm = nn.LayerNorm(channels)

        self.mamba = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.layer_scale = nn.Parameter(torch.full((channels,), layer_scale_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, T] -> [B, T, C]
        x_t = x.transpose(1, 2)

        y = self.norm(x_t)
        y = self.mamba(y)
        y = y * self.layer_scale

        x_t = x_t + y

        return x_t.transpose(1, 2)


class TemporalBlock(nn.Module):
    """
    Temporal modelling block.

    If use_mamba=True, this tries to use stabilised Mamba residual blocks.
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

        if use_mamba:
            try:
                self.layers = nn.ModuleList(
                    [
                        MambaResidualBlock(
                            channels=channels,
                            d_state=16,
                            d_conv=4,
                            expand=2,
                            layer_scale_init=0.1,
                        )
                        for _ in range(num_layers)
                    ]
                )

                self.uses_mamba = True

            except Exception as exc:
                raise RuntimeError(
                    "use_mamba=True was requested, but Mamba could not be initialised. "
                    "This would silently change the experiment into a causal-conv model. "
                    "Check that mamba-ssm is installed and that LD_LIBRARY_PATH includes "
                    "$CONDA_PREFIX/lib and $CONDA_PREFIX/lib64. "
                    f"Original error: {exc}"
                ) from exc
        else:
            self.layers = nn.ModuleList(
                [
                    CausalConvBlock(
                        channels=channels,
                        hidden_dim=hidden_dim,
                        kernel_size=kernel_size,
                    )
                    for _ in range(num_layers)
                ]
            )

            self.uses_mamba = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B, C, T], got {x.shape}")

        for layer in self.layers:
            x = layer(x)

        return x
