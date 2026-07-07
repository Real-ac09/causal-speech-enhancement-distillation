from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn


def si_sdr_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Negative SI-SDR loss.

    Inputs:
        estimate: [B, 1, T]
        target:   [B, 1, T]
    """

    estimate = estimate.squeeze(1)
    target = target.squeeze(1)

    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    projection = torch.sum(estimate * target, dim=-1, keepdim=True) * target / target_energy
    noise = estimate - projection

    ratio = torch.sum(projection ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps)
    si_sdr = 10.0 * torch.log10(ratio + eps)

    return -si_sdr.mean()


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss.

    Combines:
    - spectral convergence loss
    - log magnitude loss
    """

    def __init__(
        self,
        fft_sizes: Sequence[int] = (512, 1024, 2048),
        hop_sizes: Sequence[int] = (128, 256, 512),
        win_lengths: Sequence[int] = (512, 1024, 2048),
        eps: float = 1e-7,
    ) -> None:
        super().__init__()

        if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths)):
            raise ValueError("fft_sizes, hop_sizes and win_lengths must have the same length.")

        self.fft_sizes = tuple(fft_sizes)
        self.hop_sizes = tuple(hop_sizes)
        self.win_lengths = tuple(win_lengths)
        self.eps = eps

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        estimate = estimate.squeeze(1)
        target = target.squeeze(1)

        total_loss = estimate.new_tensor(0.0)

        for n_fft, hop_length, win_length in zip(
            self.fft_sizes,
            self.hop_sizes,
            self.win_lengths,
        ):
            window = torch.hann_window(win_length, device=estimate.device)

            estimate_stft = torch.stft(
                estimate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                return_complex=True,
            )

            target_stft = torch.stft(
                target,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                return_complex=True,
            )

            estimate_mag = estimate_stft.abs().clamp_min(self.eps)
            target_mag = target_stft.abs().clamp_min(self.eps)

            # Flatten frequency/time dimensions per batch item.
            # STFT magnitudes are [B, F, Frames], while torch.linalg.norm with
            # ord="fro" only supports 1D/2D unless dimensions are specified.
            diff = (target_mag - estimate_mag).reshape(target_mag.shape[0], -1)
            target_flat = target_mag.reshape(target_mag.shape[0], -1)

            spectral_convergence = (
                torch.linalg.vector_norm(diff, ord=2, dim=1)
                / (torch.linalg.vector_norm(target_flat, ord=2, dim=1) + self.eps)
            ).mean()

            log_mag = F.l1_loss(torch.log(estimate_mag), torch.log(target_mag))

            total_loss = total_loss + spectral_convergence + log_mag

        return total_loss / len(self.fft_sizes)


@dataclass
class LossOutput:
    total: torch.Tensor
    waveform_l1: torch.Tensor
    si_sdr: torch.Tensor
    stft: torch.Tensor
    vq: torch.Tensor

    def as_dict(self) -> Dict[str, float]:
        return {
            "loss_total": float(self.total.detach().cpu()),
            "loss_waveform_l1": float(self.waveform_l1.detach().cpu()),
            "loss_si_sdr": float(self.si_sdr.detach().cpu()),
            "loss_stft": float(self.stft.detach().cpu()),
            "loss_vq": float(self.vq.detach().cpu()),
        }


class EnhancementLoss(nn.Module):
    def __init__(
        self,
        waveform_l1_weight: float = 1.0,
        si_sdr_weight: float = 0.5,
        stft_weight: float = 1.0,
        vq_weight: float = 0.25,
    ) -> None:
        super().__init__()

        self.waveform_l1_weight = waveform_l1_weight
        self.si_sdr_weight = si_sdr_weight
        self.stft_weight = stft_weight
        self.vq_weight = vq_weight

        self.stft_loss = MultiResolutionSTFTLoss()

    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        vq_loss: torch.Tensor,
    ) -> LossOutput:
        min_len = min(enhanced.shape[-1], clean.shape[-1])
        enhanced = enhanced[..., :min_len]
        clean = clean[..., :min_len]

        waveform_l1 = F.l1_loss(enhanced, clean)
        si_sdr = si_sdr_loss(enhanced, clean)
        stft = self.stft_loss(enhanced, clean)
        vq = vq_loss

        total = (
            self.waveform_l1_weight * waveform_l1
            + self.si_sdr_weight * si_sdr
            + self.stft_weight * stft
            + self.vq_weight * vq
        )

        return LossOutput(
            total=total,
            waveform_l1=waveform_l1,
            si_sdr=si_sdr,
            stft=stft,
            vq=vq,
        )
