from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .causal_aux_vq_mamba_v5 import CausalAuxVQMambaV5Output
from .causal_aux_vq_mamba_v51 import FrameGroupNorm, _match_frequency
from .causal_complex_mamba_v6 import CausalComplexMambaV6, CausalResidualUnitV6
from .complex_cnvqg_model import TemporalStack
from .streaming_hybrid_v2 import CausalConv2d, EMANoiseVectorQuantizer


def _norm_act(channels: int) -> nn.Sequential:
    return nn.Sequential(FrameGroupNorm(channels), nn.SiLU())


class MultiScaleCausalEncoderV7(nn.Module):
    """Two frequency reductions with skips at every reconstruction scale."""

    def __init__(self, full: int, middle: int, bottleneck: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            CausalConv2d(3, full, (5, 3)),
            _norm_act(full),
            CausalResidualUnitV6(full),
        )
        self.down_one = nn.Sequential(
            nn.Conv2d(full, middle, (4, 1), stride=(2, 1), padding=(1, 0)),
            _norm_act(middle),
            CausalResidualUnitV6(middle),
        )
        self.down_two = nn.Sequential(
            nn.Conv2d(middle, bottleneck, (4, 1), stride=(2, 1), padding=(1, 0)),
            _norm_act(bottleneck),
            CausalResidualUnitV6(bottleneck),
        )

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full = self.stem(features)
        middle = self.down_one(full)
        return self.down_two(middle), middle, full


class FullBandFrequencyBlockV7(nn.Module):
    """Bidirectional frequency context within each independently causal frame."""

    def __init__(self, channels: int, frequency_dim: int) -> None:
        super().__init__()
        if frequency_dim % 2:
            raise ValueError("V7 frequency_dim must be even")
        self.frequency_in = nn.Linear(channels, frequency_dim)
        self.frequency_gru = nn.GRU(
            frequency_dim,
            frequency_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.frequency_out = nn.Linear(frequency_dim, channels)
        self.norm = nn.LayerNorm(frequency_dim)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, bins, frames = features.shape
        sequence = features.permute(0, 3, 2, 1).reshape(batch * frames, bins, channels)
        projected = self.frequency_in(sequence)
        # Deep-copied structural probes can invalidate cuDNN's packed RNN
        # storage. Re-packing is a cheap no-op once the weights are contiguous.
        self.frequency_gru.flatten_parameters()
        frequency, _ = self.frequency_gru(self.norm(projected))
        frequency = self.frequency_out(frequency)
        frequency = frequency.view(batch, frames, bins, channels).permute(0, 3, 2, 1)
        return features + torch.tanh(self.scale) * frequency


class SubBandTemporalMambaV7(nn.Module):
    """One causal temporal model shared independently across frequency bands."""

    def __init__(
        self,
        channels: int,
        temporal_dim: int,
        use_mamba: bool,
        d_state: int,
        d_conv: int,
        expand: int,
    ) -> None:
        super().__init__()
        self.temporal_in = nn.Linear(channels, temporal_dim)
        self.temporal = TemporalStack(
            dim=temporal_dim,
            layers=1,
            use_mamba=use_mamba,
            mamba_d_state=d_state,
            mamba_d_conv=d_conv,
            mamba_expand=expand,
        )
        self.temporal_out = nn.Linear(temporal_dim, channels)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, bins, frames = features.shape
        sequence = features.permute(0, 2, 3, 1).reshape(batch * bins, frames, channels)
        temporal = self.temporal_out(self.temporal(self.temporal_in(sequence)))
        temporal = temporal.view(batch, bins, frames, channels).permute(0, 3, 1, 2)
        return features + torch.tanh(self.scale) * temporal


class SelectedRecurrentCoreV7(nn.Module):
    """Convolutional core with one full-band block and one temporal Mamba."""

    def __init__(
        self,
        channels: int,
        temporal_dim: int,
        frequency_dim: int,
        use_full_band: bool,
        use_mamba: bool,
        d_state: int,
        d_conv: int,
        expand: int,
    ) -> None:
        super().__init__()
        self.pre = CausalResidualUnitV6(channels)
        self.full_band = (
            FullBandFrequencyBlockV7(channels, frequency_dim)
            if use_full_band
            else nn.Identity()
        )
        self.sub_band = SubBandTemporalMambaV7(
            channels, temporal_dim, use_mamba, d_state, d_conv, expand
        )
        self.post = CausalResidualUnitV6(channels, dilation=2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        current = self.pre(features)
        current = self.full_band(current)
        current = self.sub_band(current)
        return self.post(current)


class HighResolutionComplexDecoderV7(nn.Module):
    """Multi-scale reconstruction with an optional bounded phase detail head."""

    def __init__(
        self,
        full: int,
        middle: int,
        bottleneck: int,
        mask_representation: Literal["complex_ratio", "polar_residual"],
        complex_residual_limit: float,
        log_magnitude_limit: float,
        adaptive_phase: bool,
        phase_limit: float,
        phase_gate_max: float,
        phase_gate_init: float,
    ) -> None:
        super().__init__()
        if mask_representation not in {"complex_ratio", "polar_residual"}:
            raise ValueError(
                f"Unknown V7 mask representation: {mask_representation}"
            )
        if not 0.0 <= phase_gate_init < phase_gate_max <= 1.0:
            raise ValueError("V7 phase gate requires 0 <= init < max <= 1")
        self.mask_representation = mask_representation
        self.complex_residual_limit = float(complex_residual_limit)
        self.log_magnitude_limit = float(log_magnitude_limit)
        self.adaptive_phase = bool(adaptive_phase)
        self.phase_limit = float(phase_limit)
        self.phase_gate_max = float(phase_gate_max)
        self.up_two = nn.Sequential(
            nn.ConvTranspose2d(
                bottleneck, middle, (4, 1), stride=(2, 1), padding=(1, 0)
            ),
            _norm_act(middle),
        )
        self.fuse_middle = nn.Sequential(
            nn.Conv2d(middle * 2, middle, 1),
            _norm_act(middle),
            CausalResidualUnitV6(middle),
        )
        self.up_one = nn.Sequential(
            nn.ConvTranspose2d(middle, full, (4, 1), stride=(2, 1), padding=(1, 0)),
            _norm_act(full),
        )
        self.fuse_full = nn.Sequential(
            nn.Conv2d(full * 2, full, 1),
            _norm_act(full),
            CausalResidualUnitV6(full),
        )
        self.detail_path = nn.Sequential(
            CausalResidualUnitV6(full),
            CausalConv2d(full, full, (5, 1), groups=full),
            nn.Conv2d(full, full, 1),
            _norm_act(full),
        )
        self.detail_scale = nn.Parameter(torch.tensor(0.1))
        self.mask_head = nn.Conv2d(full, 2, 1)
        self.phase_head = nn.Conv2d(full, 1, 1)
        self.phase_gate_head = nn.Conv2d(full, 1, 1)
        nn.init.zeros_(self.mask_head.weight)
        nn.init.zeros_(self.mask_head.bias)
        nn.init.zeros_(self.phase_head.weight)
        nn.init.zeros_(self.phase_head.bias)
        nn.init.zeros_(self.phase_gate_head.weight)
        ratio = phase_gate_init / phase_gate_max
        nn.init.constant_(self.phase_gate_head.bias, math.log(ratio / (1.0 - ratio)))

    def forward(
        self,
        latent: torch.Tensor,
        middle_skip: torch.Tensor,
        full_skip: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        middle = _match_frequency(self.up_two(latent), middle_skip)
        middle = self.fuse_middle(torch.cat((middle, middle_skip), dim=1))
        full = _match_frequency(self.up_one(middle), full_skip)
        full = self.fuse_full(torch.cat((full, full_skip), dim=1))
        full = full + torch.tanh(self.detail_scale) * self.detail_path(full_skip)

        controls = self.mask_head(full).float()
        if self.mask_representation == "complex_ratio":
            residual = self.complex_residual_limit * torch.tanh(controls)
            complex_mask = torch.complex(1.0 + residual[:, 0], residual[:, 1])
        else:
            # Match V6's successful polar-residual reconstruction exactly:
            # one bounded log-magnitude control and one bounded phase control.
            magnitude = torch.exp(
                self.log_magnitude_limit * torch.tanh(controls[:, 0])
            )
            phase = self.phase_limit * torch.tanh(controls[:, 1])
            complex_mask = torch.polar(magnitude, phase)
        if self.adaptive_phase:
            phase_delta = self.phase_limit * torch.tanh(self.phase_head(full).squeeze(1).float())
            phase_gate = self.phase_gate_max * torch.sigmoid(
                self.phase_gate_head(full).squeeze(1).float()
            )
        else:
            phase_delta = controls[:, 0] * 0.0
            phase_gate = controls[:, 0] * 0.0
        return complex_mask, phase_delta, phase_gate, full


class CausalMultiScaleMambaV7(CausalComplexMambaV6):
    """Modular causal full-band/sub-band enhancement architecture."""

    PRESETS = {
        "student": {
            "full": 80,
            "middle": 128,
            "bottleneck": 208,
            "temporal": 128,
            "frequency": 96,
            "noise": 64,
            "cap": 1_100_000,
        },
        "teacher": {
            "full": 120,
            "middle": 192,
            "bottleneck": 320,
            "temporal": 208,
            "frequency": 144,
            "noise": 96,
            "cap": 2_700_000,
        },
    }

    def __init__(
        self,
        variant: Literal["student", "teacher"] = "teacher",
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 320,
        magnitude_power: float = 0.3,
        full_channels: int | None = None,
        middle_channels: int | None = None,
        bottleneck_channels: int | None = None,
        temporal_dim: int | None = None,
        frequency_dim: int | None = None,
        noise_dim: int | None = None,
        use_mamba: bool = True,
        use_full_band: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mask_representation: Literal[
            "complex_ratio", "polar_residual"
        ] = "complex_ratio",
        complex_mask_residual_limit: float = 1.0,
        log_magnitude_limit: float = math.log(4.0),
        adaptive_phase: bool = True,
        phase_limit: float = math.pi / 2.0,
        phase_gate_max: float = 0.85,
        phase_gate_init: float = 0.75,
        auxiliary_vq: bool = False,
        codebook_size: int = 32,
        vq_commitment_weight: float = 0.02,
        vq_usage_weight: float = 0.005,
        enforce_parameter_cap: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        if variant not in self.PRESETS:
            raise ValueError(f"Unknown V7 variant: {variant}")
        if n_fft < win_length or hop_length > win_length:
            raise ValueError("V7 requires n_fft >= win_length >= hop_length")
        preset = self.PRESETS[variant]
        self.variant = variant
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.magnitude_power = float(magnitude_power)
        self.full_channels = int(full_channels or preset["full"])
        self.middle_channels = int(middle_channels or preset["middle"])
        self.channels = int(bottleneck_channels or preset["bottleneck"])
        self.temporal_dim = int(temporal_dim or preset["temporal"])
        self.frequency_dim = int(frequency_dim or preset["frequency"])
        self.noise_dim = int(noise_dim or preset["noise"])
        self.auxiliary_vq = bool(auxiliary_vq)
        self.vq_usage_weight = float(vq_usage_weight)
        self.parameter_cap = int(preset["cap"])
        self.algorithmic_latency_samples = self.win_length
        self.phase_residual_scale = 1.0
        self.magnitude_residual_scale = 1.0
        self.register_buffer(
            "analysis_window",
            torch.hann_window(self.win_length, periodic=True),
            persistent=False,
        )

        self.encoder = MultiScaleCausalEncoderV7(
            self.full_channels, self.middle_channels, self.channels
        )
        self.core = SelectedRecurrentCoreV7(
            self.channels,
            self.temporal_dim,
            self.frequency_dim,
            use_full_band,
            use_mamba,
            int(mamba_d_state),
            int(mamba_d_conv),
            int(mamba_expand),
        )
        self.temporal = self.core.sub_band.temporal
        self.decoder = HighResolutionComplexDecoderV7(
            self.full_channels,
            self.middle_channels,
            self.channels,
            mask_representation,
            complex_mask_residual_limit,
            log_magnitude_limit,
            adaptive_phase,
            phase_limit,
            phase_gate_max,
            phase_gate_init,
        )
        if self.auxiliary_vq:
            self.noise_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(self.channels, self.noise_dim),
                nn.SiLU(),
                nn.LayerNorm(self.noise_dim),
            )
            self.noise_vq: EMANoiseVectorQuantizer | None = EMANoiseVectorQuantizer(
                codebook_size=codebook_size,
                code_dim=self.noise_dim,
                update_interval=1,
                commitment_weight=vq_commitment_weight,
            )
        else:
            self.noise_encoder = None
            self.noise_vq = None

        count = self.parameter_count()
        if enforce_parameter_cap and count > self.parameter_cap:
            raise ValueError(
                f"V7 {variant} has {count:,} parameters, exceeding cap "
                f"{self.parameter_cap:,}."
            )

    def _forward_waveform(
        self, noisy: torch.Tensor, pad_end: bool
    ) -> CausalAuxVQMambaV5Output:
        if noisy.ndim != 3 or noisy.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, T], got {tuple(noisy.shape)}")
        spectrum, original_length = self._analysis(noisy.squeeze(1), pad_end=pad_end)
        if spectrum.shape[-1] == 0:
            raise ValueError("At least one complete analysis window is required")
        magnitude = spectrum.abs().clamp_min(1e-7)
        unit = spectrum / magnitude
        inputs = torch.stack(
            (magnitude.pow(self.magnitude_power), unit.real, unit.imag), dim=1
        )
        latent, middle_skip, full_skip = self.encoder(inputs)
        current = self.core(latent)
        complex_mask, phase_delta, phase_gate, decoded = self.decoder(
            current, middle_skip, full_skip
        )
        raw_magnitude_mask = complex_mask.abs().clamp_min(1e-7)
        magnitude_mask = torch.exp(
            float(self.magnitude_residual_scale) * raw_magnitude_mask.log()
        )
        phase_residual = torch.angle(complex_mask) + phase_gate * phase_delta
        phase_residual = float(self.phase_residual_scale) * phase_residual
        estimated_magnitude = magnitude * magnitude_mask
        predicted_phase = torch.angle(spectrum) + phase_residual
        enhanced = self._synthesis(
            torch.polar(estimated_magnitude, predicted_phase), original_length
        ).unsqueeze(1)

        continuous_noise = self._noise_state(current)
        vq, code_indices, code_posterior = self._auxiliary_outputs(
            continuous_noise, magnitude.shape[-2]
        )
        zero = enhanced.new_tensor(0.0)
        noise_prediction = enhanced.new_zeros(
            enhanced.shape[0], magnitude.shape[-2], continuous_noise.shape[1]
        )
        return CausalAuxVQMambaV5Output(
            enhanced=enhanced,
            estimated_magnitude=estimated_magnitude,
            predicted_phase=predicted_phase,
            phase_candidate=predicted_phase,
            magnitude_mask=magnitude_mask,
            continuous_noise_state=continuous_noise,
            code_indices=code_indices,
            code_posterior=code_posterior,
            code_perplexity=vq.perplexity,
            vq_adapter_strength=zero,
            noise_prediction=noise_prediction,
            encoder_features=latent,
            mamba_features=current,
            vq=vq,
            phase_confidence=phase_gate,
        )


__all__ = [
    "CausalMultiScaleMambaV7",
    "FullBandFrequencyBlockV7",
    "HighResolutionComplexDecoderV7",
    "MultiScaleCausalEncoderV7",
    "SelectedRecurrentCoreV7",
    "SubBandTemporalMambaV7",
]
