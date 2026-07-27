from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .causal_aux_vq_mamba_v5 import AuxiliaryVQOutput, V5StreamState
from .complex_cnvqg_model import TemporalStack


class FrameChannelRMSNorm(nn.Module):
    """Normalise channels independently at every time-frequency location."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().square().mean(dim=1, keepdim=True) + self.eps)
        y = x * scale.to(x.dtype)
        return y * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class FrequencyResidualUnitV9(nn.Module):
    """Local spectral processing with no temporal receptive field."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.norm = FrameChannelRMSNorm(channels)
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=(3, 1), padding=(1, 0), groups=channels
        )
        self.expand = nn.Conv2d(channels, hidden * 2, 1)
        self.project = nn.Conv2d(hidden, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(self.norm(x))
        value, gate = self.expand(y).chunk(2, dim=1)
        y = self.project(F.silu(value) * torch.sigmoid(gate))
        return x + torch.tanh(self.scale) * y


class TwoScaleFrequencyEncoderV9(nn.Module):
    def __init__(self, detail_channels: int, half_channels: int, core_channels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, detail_channels, kernel_size=(5, 1), padding=(2, 0)),
            FrameChannelRMSNorm(detail_channels),
            nn.SiLU(),
            FrequencyResidualUnitV9(detail_channels),
        )
        # Kernel three produces 257 -> 129 -> 65 bins exactly.
        self.down_one = nn.Sequential(
            nn.Conv2d(
                detail_channels,
                half_channels,
                kernel_size=(3, 1),
                stride=(2, 1),
                padding=(1, 0),
            ),
            FrameChannelRMSNorm(half_channels),
            nn.SiLU(),
            FrequencyResidualUnitV9(half_channels),
        )
        self.down_two = nn.Sequential(
            nn.Conv2d(
                half_channels,
                core_channels,
                kernel_size=(3, 1),
                stride=(2, 1),
                padding=(1, 0),
            ),
            FrameChannelRMSNorm(core_channels),
            nn.SiLU(),
            FrequencyResidualUnitV9(core_channels),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        detail = self.stem(x)
        half = self.down_one(detail)
        core = self.down_two(half)
        return core, half, detail


class DualAxisBlockV9(nn.Module):
    """Untied temporal Mamba plus an optional within-frame frequency Mamba."""

    def __init__(
        self,
        channels: int,
        use_mamba: bool,
        use_frequency_mamba: bool,
        mamba_d_state: int,
        mamba_d_conv: int,
        mamba_expand: int,
    ) -> None:
        super().__init__()
        kwargs = dict(
            dim=channels,
            layers=1,
            use_mamba=use_mamba,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
        )
        self.local = FrequencyResidualUnitV9(channels)
        self.temporal_norm = FrameChannelRMSNorm(channels)
        self.temporal = TemporalStack(**kwargs)
        self.temporal_scale = nn.Parameter(torch.tensor(0.1))
        self.use_frequency_mamba = bool(use_frequency_mamba)
        if self.use_frequency_mamba:
            self.frequency_norm = FrameChannelRMSNorm(channels)
            self.frequency = TemporalStack(**kwargs)
            self.frequency_merge = nn.Linear(channels * 2, channels)
            self.frequency_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local(x)
        batch, channels, bins, frames = x.shape
        source = self.temporal_norm(x)
        sequence = source.permute(0, 2, 3, 1).reshape(batch * bins, frames, channels)
        temporal = self.temporal(sequence) - sequence
        temporal = temporal.view(batch, bins, frames, channels).permute(0, 3, 1, 2)
        x = x + torch.tanh(self.temporal_scale) * temporal

        if self.use_frequency_mamba:
            source = self.frequency_norm(x)
            sequence = source.permute(0, 3, 2, 1).reshape(batch * frames, bins, channels)
            forward = self.frequency(sequence) - sequence
            reversed_sequence = sequence.flip(1)
            backward = (self.frequency(reversed_sequence) - reversed_sequence).flip(1)
            frequency = self.frequency_merge(torch.cat((forward, backward), dim=-1))
            frequency = frequency.view(batch, frames, bins, channels).permute(0, 3, 2, 1)
            x = x + torch.tanh(self.frequency_scale) * frequency
        return x


class TwoScaleComplexMaskDecoderV9(nn.Module):
    def __init__(self, detail_channels: int, half_channels: int, core_channels: int) -> None:
        super().__init__()
        self.core_to_half = nn.Conv2d(core_channels, half_channels, 1)
        self.half_fuse = nn.Sequential(
            nn.Conv2d(half_channels * 2, half_channels, 1),
            FrameChannelRMSNorm(half_channels),
            nn.SiLU(),
            FrequencyResidualUnitV9(half_channels),
        )
        self.half_to_detail = nn.Conv2d(half_channels, detail_channels, 1)
        self.detail_fuse = nn.Sequential(
            nn.Conv2d(detail_channels * 2, detail_channels, 1),
            FrameChannelRMSNorm(detail_channels),
            nn.SiLU(),
            FrequencyResidualUnitV9(detail_channels),
        )
        self.mask_head = nn.Conv2d(detail_channels, 2, 1)
        nn.init.zeros_(self.mask_head.weight)
        nn.init.zeros_(self.mask_head.bias)

    def forward(
        self, core: torch.Tensor, half_skip: torch.Tensor, detail_skip: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        half = F.interpolate(
            self.core_to_half(core), size=half_skip.shape[-2:], mode="nearest"
        )
        half = self.half_fuse(torch.cat((half, half_skip), dim=1))
        detail = F.interpolate(
            self.half_to_detail(half), size=detail_skip.shape[-2:], mode="nearest"
        )
        detail = self.detail_fuse(torch.cat((detail, detail_skip), dim=1))
        return self.mask_head(detail), detail


@dataclass
class CausalPrototypeDualAxisMambaV9Output:
    enhanced: torch.Tensor
    estimated_magnitude: torch.Tensor
    predicted_phase: torch.Tensor
    phase_candidate: torch.Tensor
    phase_confidence: torch.Tensor
    magnitude_mask: torch.Tensor
    complex_mask: torch.Tensor
    continuous_noise_state: torch.Tensor
    code_indices: torch.Tensor
    code_posterior: torch.Tensor
    code_perplexity: torch.Tensor
    vq_adapter_strength: torch.Tensor
    noise_prediction: None
    encoder_features: torch.Tensor
    mamba_features: torch.Tensor
    vq: AuxiliaryVQOutput

    @property
    def estimated_phase(self) -> torch.Tensor:
        return self.predicted_phase

    @property
    def noise_state(self) -> torch.Tensor:
        return self.continuous_noise_state


class CausalPrototypeDualAxisMambaV9(nn.Module):
    """Causal-native V9 backbone.

    Control A deliberately enables only the two-scale complex U-Net and
    temporal Mamba. Noise conditioning, VQ, and phase-detail modules are added
    only after this backbone passes its locked validation gate.
    """

    PRESETS = {
        "student": {
            "detail_channels": 40,
            "half_channels": 80,
            "core_channels": 168,
            "blocks": 2,
            "noise_dim": 64,
            "cap": 1_100_000,
        },
        "teacher": {
            "detail_channels": 64,
            "half_channels": 128,
            "core_channels": 224,
            "blocks": 3,
            "noise_dim": 96,
            "cap": 2_700_000,
        },
    }

    def __init__(
        self,
        variant: Literal["student", "teacher"] = "student",
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 320,
        magnitude_power: float = 0.3,
        detail_channels: int | None = None,
        half_channels: int | None = None,
        core_channels: int | None = None,
        blocks: int | None = None,
        noise_dim: int | None = None,
        mask_residual_limit: float = 1.0,
        use_mamba: bool = True,
        use_frequency_mamba: bool = False,
        use_noise_conditioning: bool = False,
        use_auxiliary_vq: bool = False,
        use_phase_detail: bool = False,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        enforce_parameter_cap: bool = True,
    ) -> None:
        super().__init__()
        if variant not in self.PRESETS:
            raise ValueError(f"Unknown V9 variant: {variant}")
        if n_fft < win_length or hop_length > win_length:
            raise ValueError("V9 requires n_fft >= win_length >= hop_length")
        unsupported = {
            "use_noise_conditioning": use_noise_conditioning,
            "use_auxiliary_vq": use_auxiliary_vq,
            "use_phase_detail": use_phase_detail,
        }
        enabled = [name for name, value in unsupported.items() if value]
        if enabled:
            raise ValueError(
                "V9 control A does not yet enable " + ", ".join(enabled) +
                "; these branches require separate validation gates"
            )
        preset = self.PRESETS[variant]
        self.variant = variant
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.magnitude_power = float(magnitude_power)
        self.detail_channels = int(detail_channels or preset["detail_channels"])
        self.half_channels = int(half_channels or preset["half_channels"])
        self.core_channels = int(core_channels or preset["core_channels"])
        self.blocks = int(blocks or preset["blocks"])
        self.noise_dim = int(noise_dim or preset["noise_dim"])
        self.mask_residual_limit = float(mask_residual_limit)
        self.parameter_cap = int(preset["cap"])
        self.algorithmic_latency_samples = self.win_length
        self.use_frequency_mamba = bool(use_frequency_mamba)

        self.register_buffer(
            "analysis_window", torch.hann_window(self.win_length, periodic=True), persistent=False
        )
        self.encoder = TwoScaleFrequencyEncoderV9(
            self.detail_channels, self.half_channels, self.core_channels
        )
        self.core = nn.ModuleList(
            DualAxisBlockV9(
                self.core_channels,
                use_mamba=use_mamba,
                use_frequency_mamba=use_frequency_mamba,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
            )
            for _ in range(self.blocks)
        )
        self.decoder = TwoScaleComplexMaskDecoderV9(
            self.detail_channels, self.half_channels, self.core_channels
        )
        # Compatibility alias used by optimiser grouping and distillation tools.
        self.temporal = self.core[0].temporal

        count = self.parameter_count()
        if enforce_parameter_cap and count > self.parameter_cap:
            raise ValueError(
                f"V9 {variant} has {count:,} parameters, exceeding cap "
                f"{self.parameter_cap:,}. Reduce widths in increments of eight."
            )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _analysis(self, waveform: torch.Tensor, pad_end: bool) -> tuple[torch.Tensor, int]:
        original_length = waveform.shape[-1]
        if original_length < self.win_length:
            if not pad_end:
                return waveform.new_zeros(
                    waveform.shape[0], self.n_fft // 2 + 1, 0, dtype=torch.complex64
                ), original_length
            pad = self.win_length - original_length
        elif pad_end:
            pad = (
                self.hop_length - (original_length - self.win_length) % self.hop_length
            ) % self.hop_length
        else:
            pad = 0
        if pad:
            waveform = F.pad(waveform, (0, pad))
        frames = waveform.unfold(-1, self.win_length, self.hop_length)
        windowed = frames * self.analysis_window.to(frames)
        spectrum = torch.fft.rfft(windowed.float(), n=self.n_fft, dim=-1)
        return spectrum.transpose(-1, -2), original_length

    def _synthesis(self, spectrum: torch.Tensor, length: int) -> torch.Tensor:
        frames = torch.fft.irfft(spectrum.transpose(-1, -2), n=self.n_fft, dim=-1)[
            ..., : self.win_length
        ]
        window = self.analysis_window.to(frames)
        frames = frames * window
        frame_count = frames.shape[-2]
        output_length = (frame_count - 1) * self.hop_length + self.win_length
        output = F.fold(
            frames.transpose(1, 2),
            output_size=(1, output_length),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze(1).squeeze(1)
        weight = F.fold(
            window.square()[None, :, None].expand(1, self.win_length, frame_count),
            output_size=(1, output_length),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()
        output = torch.where(
            weight[None] >= 5e-3,
            output / weight.clamp_min(5e-3),
            torch.zeros_like(output),
        )
        return F.pad(output, (0, max(0, length - output.shape[-1])))[:, :length]

    def _empty_vq(
        self, reference: torch.Tensor, batch: int, frames: int
    ) -> tuple[AuxiliaryVQOutput, torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = reference.new_tensor(0.0)
        vq = AuxiliaryVQOutput(
            loss=zero,
            commitment_loss=zero,
            usage_kl=zero,
            reconstruction_loss=zero,
            perplexity=zero,
            active_fraction=zero,
            dead_fraction=zero,
        )
        indices = torch.zeros(batch, frames, device=reference.device, dtype=torch.long)
        posterior = reference.new_ones(batch, frames, 1)
        noise = reference.new_zeros(batch, frames, self.noise_dim)
        return vq, indices, posterior, noise

    def _forward_waveform(
        self, noisy: torch.Tensor, pad_end: bool
    ) -> CausalPrototypeDualAxisMambaV9Output:
        if noisy.ndim != 3 or noisy.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, T], got {tuple(noisy.shape)}")
        spectrum, original_length = self._analysis(noisy.squeeze(1), pad_end=pad_end)
        if spectrum.shape[-1] == 0:
            raise ValueError("At least one complete analysis window is required")
        magnitude = spectrum.abs().clamp_min(1e-7)
        compressed = magnitude.pow(self.magnitude_power)
        unit = spectrum / magnitude
        inputs = torch.stack(
            (compressed * unit.real, compressed * unit.imag, torch.log1p(magnitude)), dim=1
        )
        # torch.fft intentionally computes in FP32 under BF16. Cast only the
        # neural features back to the encoder dtype; reconstruction stays FP32.
        inputs = inputs.to(self.encoder.stem[0].weight.dtype)
        core, half_skip, detail_skip = self.encoder(inputs)
        current = core
        for block in self.core:
            current = block(current)
        raw_mask, detail = self.decoder(current, half_skip, detail_skip)
        real = 1.0 + self.mask_residual_limit * torch.tanh(raw_mask[:, 0])
        imaginary = self.mask_residual_limit * torch.tanh(raw_mask[:, 1])
        complex_mask = torch.complex(real.float(), imaginary.float())
        compressed_spectrum = torch.complex(
            compressed.float() * unit.real.float(), compressed.float() * unit.imag.float()
        )
        estimated_compressed = compressed_spectrum * complex_mask
        estimated_magnitude = estimated_compressed.abs().clamp_min(1e-7).pow(
            1.0 / self.magnitude_power
        )
        predicted_phase = torch.angle(estimated_compressed)
        enhanced_spectrum = torch.polar(estimated_magnitude, predicted_phase)
        enhanced = self._synthesis(enhanced_spectrum, original_length).unsqueeze(1)
        vq, indices, posterior, noise = self._empty_vq(
            enhanced, noisy.shape[0], spectrum.shape[-1]
        )
        magnitude_mask = estimated_magnitude / magnitude
        confidence = torch.ones_like(predicted_phase)
        zero = enhanced.new_tensor(0.0)
        return CausalPrototypeDualAxisMambaV9Output(
            enhanced=enhanced,
            estimated_magnitude=estimated_magnitude,
            predicted_phase=predicted_phase,
            phase_candidate=predicted_phase,
            phase_confidence=confidence,
            magnitude_mask=magnitude_mask,
            complex_mask=complex_mask,
            continuous_noise_state=noise,
            code_indices=indices,
            code_posterior=posterior,
            code_perplexity=zero,
            vq_adapter_strength=zero,
            noise_prediction=None,
            encoder_features=core,
            mamba_features=current,
            vq=vq,
        )

    def forward(self, noisy: torch.Tensor) -> CausalPrototypeDualAxisMambaV9Output:
        return self._forward_waveform(noisy, pad_end=True)

    def init_stream_state(
        self, batch_size: int, device: torch.device | str, dtype: torch.dtype
    ) -> V5StreamState:
        return V5StreamState(torch.empty(batch_size, 1, 0, device=device, dtype=dtype))

    @torch.no_grad()
    def forward_chunk(
        self, audio_chunk: torch.Tensor, state: V5StreamState
    ) -> tuple[torch.Tensor, V5StreamState]:
        if audio_chunk.shape[:2] != state.waveform.shape[:2]:
            raise ValueError("Chunk batch/channel dimensions do not match stream state")
        state.waveform = torch.cat((state.waveform, audio_chunk), dim=-1)
        if state.waveform.shape[-1] < self.win_length:
            return audio_chunk[..., :0], state
        complete_frames = 1 + (state.waveform.shape[-1] - self.win_length) // self.hop_length
        stable_samples = complete_frames * self.hop_length
        consumed = (complete_frames - 1) * self.hop_length + self.win_length
        result = self._forward_waveform(state.waveform[..., :consumed], pad_end=False)
        emitted = result.enhanced[..., state.emitted_samples:stable_samples]
        state.emitted_samples = stable_samples
        return emitted, state

    @torch.no_grad()
    def flush(self, state: V5StreamState) -> tuple[torch.Tensor, V5StreamState]:
        if state.waveform.shape[-1] == 0:
            return state.waveform, state
        result = self.forward(state.waveform)
        tail = result.enhanced[..., state.emitted_samples:]
        state.emitted_samples = state.waveform.shape[-1]
        return tail, state


__all__ = [
    "CausalPrototypeDualAxisMambaV9",
    "CausalPrototypeDualAxisMambaV9Output",
    "DualAxisBlockV9",
    "FrameChannelRMSNorm",
]
