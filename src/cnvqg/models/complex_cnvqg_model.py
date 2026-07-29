from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class VQOutput:
    loss: torch.Tensor
    perplexity: torch.Tensor


@dataclass
class ComplexCNVQGOutput:
    enhanced: torch.Tensor
    vq: VQOutput


class SimpleVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int = 128,
        commitment_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.codebook_size = int(codebook_size)
        self.commitment_weight = float(commitment_weight)

        self.codebook = nn.Embedding(self.codebook_size, self.dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, VQOutput]:
        # z: [B, T, D]
        b, t, d = z.shape
        flat = z.reshape(-1, d)

        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=1).unsqueeze(0)
        )

        indices = torch.argmin(distances, dim=1)
        quantized = self.codebook(indices).view(b, t, d)

        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(quantized.detach(), z)
        loss = codebook_loss + self.commitment_weight * commitment_loss

        quantized = z + (quantized - z).detach()

        one_hot = F.one_hot(indices, num_classes=self.codebook_size).float()
        avg_probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-8)))

        return quantized, VQOutput(loss=loss, perplexity=perplexity)


class MambaTemporalBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        layer_scale_init: float = 0.1,
    ) -> None:
        super().__init__()

        try:
            from mamba_ssm import Mamba
        except Exception as exc:
            raise RuntimeError(
                "use_mamba=True was requested for ComplexCNVQGModel, but mamba_ssm could not be imported."
            ) from exc

        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.layer_scale = nn.Parameter(torch.ones(dim) * layer_scale_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        normalized = self.norm(x)
        if normalized.device.type == "cpu":
            transformed = self._reference_mamba(normalized)
        else:
            transformed = self.mamba(normalized)
        return x + self.layer_scale * transformed

    def _reference_mamba(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch Mamba-1 scan for CPU validation and export fallback."""
        mamba = self.mamba
        xz = F.linear(hidden_states, mamba.in_proj.weight)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = F.conv1d(
            F.pad(x, (mamba.d_conv - 1, 0)),
            mamba.conv1d.weight,
            mamba.conv1d.bias,
            groups=x.shape[1],
        ).transpose(1, 2)
        x = F.silu(x)
        projected = F.linear(x, mamba.x_proj.weight)
        dt_rank = mamba.dt_rank
        dt, b_term, c_term = torch.split(
            projected, (dt_rank, mamba.d_state, mamba.d_state), dim=-1
        )
        dt = F.softplus(F.linear(dt, mamba.dt_proj.weight, mamba.dt_proj.bias))
        a_term = -torch.exp(mamba.A_log.float()).to(x.dtype)
        state = x.new_zeros(x.shape[0], x.shape[2], mamba.d_state)
        outputs = []
        for frame in range(x.shape[1]):
            delta = dt[:, frame]
            transition = torch.exp(delta.unsqueeze(-1) * a_term.unsqueeze(0))
            state = transition * state + (
                delta.unsqueeze(-1)
                * b_term[:, frame].unsqueeze(1)
                * x[:, frame].unsqueeze(-1)
            )
            value = (state * c_term[:, frame].unsqueeze(1)).sum(-1)
            value = value + mamba.D.to(x.dtype) * x[:, frame]
            outputs.append(value * F.silu(z[:, frame]))
        return F.linear(torch.stack(outputs, dim=1), mamba.out_proj.weight)


class ConvTemporalBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5, dilation: int = 1) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.pad = (self.kernel_size - 1) * self.dilation

        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, dilation=dilation),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        y = self.norm(x).transpose(1, 2)
        y = F.pad(y, (self.pad, 0))
        y = self.net(y)
        y = y.transpose(1, 2)
        return x + y


class GRUTemporalBlock(nn.Module):
    """Frame-causal GRU residual block with an optional narrow state."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        layer_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim or dim)
        self.norm = nn.LayerNorm(self.dim)
        self.input_projection: nn.Module
        if self.hidden_dim == self.dim:
            self.input_projection = nn.Identity()
        else:
            self.input_projection = nn.Linear(self.dim, self.hidden_dim)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.output_projection = nn.Linear(self.hidden_dim, self.dim)
        self.layer_scale = nn.Parameter(torch.ones(self.dim) * layer_scale_init)

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        normalized = self.input_projection(self.norm(x))
        transformed, next_hidden = self.gru(normalized, hidden)
        output = x + self.layer_scale * self.output_projection(transformed)
        if hidden is None:
            return output
        return output, next_hidden


class TemporalStack(nn.Module):
    def __init__(
        self,
        dim: int,
        layers: int,
        use_mamba: bool,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        core: str | None = None,
        gru_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        selected_core = core or ("mamba" if use_mamba else "causal_conv")
        if selected_core not in {"mamba", "gru", "causal_conv"}:
            raise ValueError(
                f"Unknown temporal core {selected_core!r}; "
                "expected 'mamba', 'gru', or 'causal_conv'"
            )
        self.core = selected_core
        self.uses_mamba = selected_core == "mamba"

        blocks = []
        for idx in range(int(layers)):
            if selected_core == "mamba":
                blocks.append(
                    MambaTemporalBlock(
                        dim=dim,
                        d_state=mamba_d_state,
                        d_conv=mamba_d_conv,
                        expand=mamba_expand,
                    )
                )
            elif selected_core == "gru":
                blocks.append(GRUTemporalBlock(dim=dim, hidden_dim=gru_hidden_dim))
            else:
                blocks.append(
                    ConvTemporalBlock(
                        dim=dim,
                        kernel_size=5,
                        dilation=2 ** (idx % 4),
                    )
                )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ComplexCNVQGModel(nn.Module):
    """
    Complex STFT-domain CN-VQG prototype.

    Input:
      noisy waveform [B, 1, T]

    Output:
      enhanced waveform [B, 1, T]

    The decoder predicts a residual complex mask:
      enhanced_stft = noisy_stft * ((1 + a*tanh(r)) + j*a*tanh(i))

    The final decoder layer is zero-initialised, so the model starts close to
    identity/noisy reconstruction rather than destroying the signal.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        hidden_dim: int = 128,
        noise_dim: int = 64,
        codebook_size: int = 128,
        temporal_layers: int = 4,
        use_mamba: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mask_scale: float = 0.5,
        center: bool = True,
        commitment_weight: float = 0.25,
    ) -> None:
        super().__init__()

        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.hidden_dim = int(hidden_dim)
        self.noise_dim = int(noise_dim)
        self.mask_scale = float(mask_scale)
        self.center = bool(center)

        self.n_bins = self.n_fft // 2 + 1
        in_channels = self.n_bins * 3

        self.input_projection = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

        self.temporal = TemporalStack(
            dim=hidden_dim,
            layers=temporal_layers,
            use_mamba=use_mamba,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
        )

        self.noise_projection = nn.Linear(hidden_dim, noise_dim)
        self.vq = SimpleVectorQuantizer(
            dim=noise_dim,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
        )

        self.noise_to_hidden = nn.Conv1d(noise_dim, hidden_dim, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, self.n_bins * 2, kernel_size=1),
        )

        final = self.decoder[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(self.win_length, device=waveform.device)

        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            return_complex=True,
        )

    def _istft(self, stft: torch.Tensor, length: int) -> torch.Tensor:
        window = torch.hann_window(self.win_length, device=stft.device)

        return torch.istft(
            stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            length=length,
        )

    def forward(self, noisy: torch.Tensor) -> ComplexCNVQGOutput:
        length = noisy.shape[-1]
        noisy_wave = noisy.squeeze(1)

        noisy_stft = self._stft(noisy_wave)

        mag = noisy_stft.abs().clamp_min(1e-7)
        log_mag = torch.log1p(mag)
        cos_phase = noisy_stft.real / mag
        sin_phase = noisy_stft.imag / mag

        features = torch.cat([log_mag, cos_phase, sin_phase], dim=1)
        h = self.input_projection(features)

        h_time = h.transpose(1, 2)
        h_time = self.temporal(h_time)

        noise_latent = self.noise_projection(h_time)
        noise_quantized, vq_output = self.vq(noise_latent)

        h = h_time.transpose(1, 2)
        noise_h = self.noise_to_hidden(noise_quantized.transpose(1, 2))

        gate = self.gate(torch.cat([h, noise_h], dim=1))
        conditioned = h + gate * noise_h

        mask_raw = self.decoder(conditioned)
        b, _, frames = mask_raw.shape
        mask_raw = mask_raw.view(b, 2, self.n_bins, frames)

        mask_real = 1.0 + self.mask_scale * torch.tanh(mask_raw[:, 0])
        mask_imag = self.mask_scale * torch.tanh(mask_raw[:, 1])
        complex_mask = torch.complex(mask_real, mask_imag)

        enhanced_stft = noisy_stft * complex_mask
        enhanced = self._istft(enhanced_stft, length=length).unsqueeze(1)

        return ComplexCNVQGOutput(enhanced=enhanced, vq=vq_output)
