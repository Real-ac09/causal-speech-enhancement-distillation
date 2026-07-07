from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class VQOutput:
    quantized: torch.Tensor
    indices: torch.Tensor
    loss: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor
    perplexity: torch.Tensor


class VectorQuantizer(nn.Module):
    """
    Basic vector quantisation module for noise-state latents.

    Input:
        z: [B, C, T]

    Output:
        quantized: [B, C, T]
    """

    def __init__(
        self,
        num_codes: int = 256,
        code_dim: int = 64,
        commitment_weight: float = 0.25,
    ) -> None:
        super().__init__()

        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_weight = commitment_weight

        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z: torch.Tensor) -> VQOutput:
        if z.ndim != 3:
            raise ValueError(f"Expected z shape [B, C, T], got {z.shape}")

        b, c, t = z.shape

        if c != self.code_dim:
            raise ValueError(f"Expected code_dim={self.code_dim}, got C={c}")

        # [B, C, T] -> [B*T, C]
        z_flat = z.permute(0, 2, 1).contiguous().view(-1, c)

        # Squared L2 distance to codebook.
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=1)
        )

        indices = torch.argmin(distances, dim=1)
        quantized_flat = self.codebook(indices)

        # [B*T, C] -> [B, C, T]
        quantized = quantized_flat.view(b, t, c).permute(0, 2, 1).contiguous()

        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(quantized.detach(), z)
        loss = codebook_loss + self.commitment_weight * commitment_loss

        # Straight-through estimator.
        quantized_st = z + (quantized - z).detach()

        one_hot = F.one_hot(indices, self.num_codes).float()
        avg_probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return VQOutput(
            quantized=quantized_st,
            indices=indices.view(b, t),
            loss=loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            perplexity=perplexity,
        )
