from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from .cnvqg_model import CNVQGModel
from .complex_cnvqg_model import TemporalStack


@dataclass
class HybridTFRefinerOutput:
    enhanced: torch.Tensor
    vq: Any


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: tuple[int, int] = (3, 5),
        dilation: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()

        padding = (
            ((kernel_size[0] - 1) * dilation[0]) // 2,
            ((kernel_size[1] - 1) * dilation[1]) // 2,
        )

        self.net = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=channels,
            ),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class HybridTFRefinerCNVQGModel(nn.Module):
    """
    Hybrid waveform + STFT refinement model.

    Stage 1:
      A pretrained waveform CN-VQG model produces a strong enhanced waveform.

    Stage 2:
      A lightweight 2D time-frequency refiner predicts bounded corrections:
        gain = exp(gain_scale * tanh(gain_logits))
        phase_shift = phase_scale * tanh(phase_logits)

    The final refiner layer is zero-initialised, so the model starts as:
      final_output ≈ waveform_branch_output

    This avoids the Phase 4A/4B problem where a pure complex mask could damage
    intelligibility.
    """

    def __init__(
        self,
        waveform_checkpoint: str,
        waveform_model_config: Optional[Dict[str, Any]] = None,
        freeze_waveform: bool = True,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        hidden_dim: int = 32,
        temporal_layers: int = 2,
        use_mamba: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        predict_phase: bool = False,
        gain_scale: float = 0.20,
        phase_scale: float = 0.10,
        center: bool = True,
    ) -> None:
        super().__init__()

        self.waveform_checkpoint = str(waveform_checkpoint)
        self.freeze_waveform = bool(freeze_waveform)

        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.hidden_dim = int(hidden_dim)
        self.predict_phase = bool(predict_phase)
        self.gain_scale = float(gain_scale)
        self.phase_scale = float(phase_scale)
        self.center = bool(center)

        checkpoint = torch.load(self.waveform_checkpoint, map_location="cpu")

        if waveform_model_config is None:
            waveform_model_config = dict(checkpoint["config"]["model"])

        waveform_model_config = dict(waveform_model_config)
        waveform_model_config.pop("architecture", None)

        self.waveform_branch = CNVQGModel(**waveform_model_config)
        self.waveform_branch.load_state_dict(checkpoint["model_state_dict"])

        if self.freeze_waveform:
            self.waveform_branch.eval()
            for parameter in self.waveform_branch.parameters():
                parameter.requires_grad_(False)

        # Features:
        # 1. noisy log-mag
        # 2. base enhanced log-mag
        # 3. base-minus-noisy log-mag
        # 4. base cos phase
        # 5. base sin phase
        in_channels = 5

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            DepthwiseSeparableConv2d(hidden_dim, kernel_size=(3, 5), dilation=(1, 1)),
            DepthwiseSeparableConv2d(hidden_dim, kernel_size=(3, 5), dilation=(1, 2)),
            DepthwiseSeparableConv2d(hidden_dim, kernel_size=(3, 5), dilation=(1, 4)),
        )

        self.temporal = TemporalStack(
            dim=hidden_dim,
            layers=temporal_layers,
            use_mamba=use_mamba,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
        )

        self.temporal_projection = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
        )

        out_channels = 2 if self.predict_phase else 1

        self.decoder = nn.Sequential(
            DepthwiseSeparableConv2d(hidden_dim, kernel_size=(3, 5), dilation=(1, 1)),
            DepthwiseSeparableConv2d(hidden_dim, kernel_size=(3, 5), dilation=(1, 2)),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
        )

        final = self.decoder[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze_waveform:
            self.waveform_branch.eval()

        return self

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

    def _waveform_forward(self, noisy: torch.Tensor):
        if self.freeze_waveform:
            with torch.no_grad():
                return self.waveform_branch(noisy)

        return self.waveform_branch(noisy)

    def forward(self, noisy: torch.Tensor) -> HybridTFRefinerOutput:
        length = noisy.shape[-1]

        base_output = self._waveform_forward(noisy)
        base_wave = base_output.enhanced.squeeze(1)
        noisy_wave = noisy.squeeze(1)

        noisy_stft = self._stft(noisy_wave)
        base_stft = self._stft(base_wave)

        noisy_mag = noisy_stft.abs().clamp_min(1e-7)
        base_mag = base_stft.abs().clamp_min(1e-7)

        noisy_log_mag = torch.log1p(noisy_mag)
        base_log_mag = torch.log1p(base_mag)
        log_mag_diff = base_log_mag - noisy_log_mag

        base_cos = base_stft.real / base_mag
        base_sin = base_stft.imag / base_mag

        features = torch.stack(
            [
                noisy_log_mag,
                base_log_mag,
                log_mag_diff,
                base_cos,
                base_sin,
            ],
            dim=1,
        )

        h = self.encoder(features)

        # Mamba over time using frequency-pooled TF features.
        # h: [B, C, F, T]
        temporal_features = h.mean(dim=2).transpose(1, 2)  # [B, T, C]
        temporal_features = self.temporal(temporal_features)
        temporal_features = temporal_features.transpose(1, 2).unsqueeze(2)

        h = h + self.temporal_projection(temporal_features)

        correction = self.decoder(h)

        gain_logits = correction[:, 0]
        gain = torch.exp(self.gain_scale * torch.tanh(gain_logits))

        base_phase = torch.angle(base_stft)

        if self.predict_phase:
            phase_logits = correction[:, 1]
            phase_shift = self.phase_scale * torch.tanh(phase_logits)
        else:
            phase_shift = torch.zeros_like(base_phase)

        refined_mag = base_mag * gain
        refined_phase = base_phase + phase_shift

        refined_stft = torch.polar(refined_mag, refined_phase)
        refined_wave = self._istft(refined_stft, length=length).unsqueeze(1)

        return HybridTFRefinerOutput(
            enhanced=refined_wave,
            vq=base_output.vq,
        )
