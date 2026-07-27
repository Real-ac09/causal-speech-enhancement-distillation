from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .complex_cnvqg_model import TemporalStack
from .streaming_hybrid_v2 import CausalConv2d, EMANoiseVectorQuantizer


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


@dataclass
class AuxiliaryVQOutput:
    loss: torch.Tensor
    commitment_loss: torch.Tensor
    usage_kl: torch.Tensor
    reconstruction_loss: torch.Tensor
    perplexity: torch.Tensor
    active_fraction: torch.Tensor
    dead_fraction: torch.Tensor


@dataclass
class CausalAuxVQMambaV5Output:
    enhanced: torch.Tensor
    estimated_magnitude: torch.Tensor
    predicted_phase: torch.Tensor
    phase_candidate: torch.Tensor
    magnitude_mask: torch.Tensor
    continuous_noise_state: torch.Tensor
    code_indices: torch.Tensor
    code_posterior: torch.Tensor
    code_perplexity: torch.Tensor
    vq_adapter_strength: torch.Tensor
    noise_prediction: torch.Tensor
    encoder_features: torch.Tensor
    mamba_features: torch.Tensor
    vq: AuxiliaryVQOutput
    phase_confidence: torch.Tensor | None = None

    @property
    def estimated_phase(self) -> torch.Tensor:
        return self.predicted_phase

    @property
    def noise_state(self) -> torch.Tensor:
        return self.continuous_noise_state


@dataclass
class V5StreamState:
    """Reference streaming state.

    The state retains the current utterance and only releases samples whose
    overlap-add support is complete. This makes the public streaming contract
    exact for arbitrary chunk sizes. An exported frame kernel can replace the
    reference recomputation without changing the API.
    """

    waveform: torch.Tensor
    emitted_samples: int = 0


class OneStageCausalEncoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        stem_channels = channels // 2
        self.stem = nn.Sequential(
            CausalConv2d(3, stem_channels, (3, 3)),
            nn.GroupNorm(_groups(stem_channels), stem_channels),
            nn.SiLU(),
        )
        self.down = nn.Sequential(
            nn.Conv2d(
                stem_channels,
                channels,
                kernel_size=(4, 1),
                stride=(2, 1),
                padding=(1, 0),
            ),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        full = self.stem(x)
        return self.down(full), full


class FixedDualAxisMambaCell(nn.Module):
    """A tied refinement cell: causal in time and bidirectional in frequency."""

    def __init__(
        self,
        channels: int,
        noise_dim: int,
        use_mamba: bool,
        d_state: int,
        d_conv: int,
        expand: int,
    ) -> None:
        super().__init__()
        self.local = nn.Sequential(
            CausalConv2d(channels, channels, (3, 3), groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
        )
        kwargs = dict(
            dim=channels,
            layers=1,
            use_mamba=use_mamba,
            mamba_d_state=d_state,
            mamba_d_conv=d_conv,
            mamba_expand=expand,
        )
        self.time_mamba = TemporalStack(**kwargs)
        self.frequency_mamba = TemporalStack(**kwargs)
        self.condition = nn.Linear(noise_dim, channels * 2)
        self.time_scale = nn.Parameter(torch.tensor(0.1))
        self.frequency_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        batch, channels, bins, frames = x.shape
        x = x + self.local(x)

        sequence = x.permute(0, 2, 3, 1).reshape(batch * bins, frames, channels)
        temporal = self.time_mamba(sequence)
        temporal = temporal.view(batch, bins, frames, channels).permute(0, 3, 1, 2)
        x = x + torch.tanh(self.time_scale) * temporal

        sequence = x.permute(0, 3, 2, 1).reshape(batch * frames, bins, channels)
        forward = self.frequency_mamba(sequence)
        backward = self.frequency_mamba(sequence.flip(1)).flip(1)
        frequency = (forward + backward).mul(0.5)
        frequency = frequency.view(batch, frames, bins, channels).permute(0, 3, 2, 1)
        x = x + torch.tanh(self.frequency_scale) * frequency

        condition = self.condition(noise).transpose(1, 2)
        condition = F.interpolate(condition, size=frames, mode="nearest")
        scale, shift = condition.chunk(2, dim=1)
        return x * (1.0 + 0.1 * torch.tanh(scale[:, :, None])) + 0.1 * shift[:, :, None]


class FullSkipDualHeadDecoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        half = channels // 2
        self.latent = nn.Sequential(
            nn.Conv2d(channels, half, 1),
            nn.GroupNorm(_groups(half), half),
            nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            CausalConv2d(channels, half, (3, 3)),
            nn.GroupNorm(_groups(half), half),
            nn.SiLU(),
        )
        self.magnitude_branch = nn.Sequential(
            CausalConv2d(half, half, (3, 3), groups=half),
            nn.Conv2d(half, half, 1),
            nn.SiLU(),
        )
        self.phase_branch = nn.Sequential(
            CausalConv2d(half, half, (3, 3), groups=half),
            nn.Conv2d(half, half, 1),
            nn.SiLU(),
        )
        self.magnitude_head = nn.Conv2d(half, 1, 1)
        self.phase_head = nn.Conv2d(half, 2, 1)
        self.magnitude_to_phase = nn.Conv2d(half, half, 1, bias=False)
        self.phase_cross_scale = nn.Parameter(torch.tensor(0.0))
        self.mask_scale_logit = nn.Parameter(torch.tensor(0.0))

        nn.init.zeros_(self.magnitude_head.weight)
        nn.init.zeros_(self.magnitude_head.bias)
        nn.init.zeros_(self.phase_head.weight)
        nn.init.zeros_(self.phase_head.bias)
        with torch.no_grad():
            self.phase_head.bias[1] = 1.0

    def forward(
        self, latent: torch.Tensor, full_skip: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        up = F.interpolate(
            self.latent(latent), size=full_skip.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded = self.fuse(torch.cat((up, full_skip), dim=1))
        magnitude_features = decoded + self.magnitude_branch(decoded)
        mask_logits = self.magnitude_head(magnitude_features).squeeze(1)
        # Starts exactly at an identity mask while retaining bounded learnable range.
        maximum = 1.0 + 2.0 * torch.sigmoid(self.mask_scale_logit)
        mask = maximum * torch.sigmoid(mask_logits + torch.log(1.0 / (maximum - 1.0)))

        phase_features = decoded + self.phase_branch(decoded)
        phase_features = phase_features + 0.05 * torch.tanh(
            self.phase_cross_scale
        ) * self.magnitude_to_phase(magnitude_features)
        phase_vector = self.phase_head(phase_features)
        phase_vector = F.normalize(phase_vector, dim=1, eps=1e-7)
        return mask, phase_vector


class BoundedVQAdapter(nn.Module):
    def __init__(self, noise_dim: int, channels: int, rank: int = 8) -> None:
        super().__init__()
        self.down = nn.Linear(noise_dim, rank, bias=False)
        self.up = nn.Linear(rank, channels * 2, bias=False)
        self.strength_logit = nn.Parameter(torch.tensor(-8.0))
        nn.init.zeros_(self.up.weight)

    def forward(self, code: torch.Tensor, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        adapter = self.up(F.silu(self.down(code)))
        strength = 0.05 * torch.sigmoid(self.strength_logit)
        if training:
            keep = torch.empty(code.shape[:2] + (1,), device=code.device).bernoulli_(0.5)
            adapter = adapter * keep
        return strength * adapter, strength


class CausalAuxVQMambaV5(nn.Module):
    PRESETS = {
        # The requested 128/208 starting widths undershoot their envelopes.
        # These are the largest eight-channel search points below each cap for
        # the Mamba implementation used by this project.
        "student": {"channels": 208, "noise_dim": 64, "passes": 2, "cap": 1_100_000},
        "teacher": {"channels": 336, "noise_dim": 96, "passes": 3, "cap": 2_700_000},
    }

    def __init__(
        self,
        variant: Literal["student", "teacher"] = "student",
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 320,
        magnitude_power: float = 0.3,
        channels: int | None = None,
        noise_dim: int | None = None,
        refinement_passes: int | None = None,
        codebook_size: int = 32,
        vq_mode: Literal["train_only", "bounded_adapter"] = "train_only",
        vq_commitment_weight: float = 0.02,
        vq_usage_weight: float = 0.005,
        vq_reconstruction_weight: float = 0.05,
        use_mamba: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        enforce_parameter_cap: bool = True,
    ) -> None:
        super().__init__()
        if variant not in self.PRESETS:
            raise ValueError(f"Unknown V5 variant: {variant}")
        if n_fft < win_length or hop_length > win_length:
            raise ValueError("V5 requires n_fft >= win_length >= hop_length")
        if vq_mode not in {"train_only", "bounded_adapter"}:
            raise ValueError(f"Unknown VQ mode: {vq_mode}")
        preset = self.PRESETS[variant]
        self.variant = variant
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.magnitude_power = float(magnitude_power)
        self.channels = int(channels or preset["channels"])
        self.noise_dim = int(noise_dim or preset["noise_dim"])
        self.refinement_passes = int(refinement_passes or preset["passes"])
        self.parameter_cap = int(preset["cap"])
        self.vq_mode = vq_mode
        self.vq_usage_weight = float(vq_usage_weight)
        self.vq_reconstruction_weight = float(vq_reconstruction_weight)
        self.algorithmic_latency_samples = self.win_length

        self.register_buffer(
            "analysis_window", torch.hann_window(self.win_length, periodic=True), persistent=False
        )
        self.encoder = OneStageCausalEncoder(self.channels)
        self.noise_encoder = nn.Sequential(
            nn.Linear(self.channels, self.noise_dim), nn.SiLU(), nn.LayerNorm(self.noise_dim)
        )
        self.noise_vq = EMANoiseVectorQuantizer(
            codebook_size=codebook_size,
            code_dim=self.noise_dim,
            update_interval=1,
            commitment_weight=vq_commitment_weight,
        )
        self.cell = FixedDualAxisMambaCell(
            self.channels,
            self.noise_dim,
            use_mamba,
            mamba_d_state,
            mamba_d_conv,
            mamba_expand,
        )
        self.adapter = BoundedVQAdapter(self.noise_dim, self.channels)
        self.decoder = FullSkipDualHeadDecoder(self.channels)
        self.noise_predictor = nn.Sequential(
            nn.Linear(self.noise_dim, self.noise_dim),
            nn.SiLU(),
            nn.Linear(self.noise_dim, self.n_fft // 2 + 1),
            nn.Softplus(),
        )
        self.temporal = self.cell.time_mamba

        count = self.parameter_count()
        if enforce_parameter_cap and count > self.parameter_cap:
            raise ValueError(
                f"V5 {variant} has {count:,} parameters, exceeding cap {self.parameter_cap:,}. "
                "Reduce channels in increments of eight."
            )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def set_vq_mode(self, mode: Literal["train_only", "bounded_adapter"]) -> None:
        if mode not in {"train_only", "bounded_adapter"}:
            raise ValueError(f"Unknown VQ mode: {mode}")
        self.vq_mode = mode

    def _analysis(self, waveform: torch.Tensor, pad_end: bool) -> tuple[torch.Tensor, int]:
        original_length = waveform.shape[-1]
        if original_length < self.win_length:
            if not pad_end:
                return waveform.new_zeros(
                    waveform.shape[0], self.n_fft // 2 + 1, 0, dtype=torch.complex64
                ), original_length
            pad = self.win_length - original_length
        elif pad_end:
            pad = (self.hop_length - (original_length - self.win_length) % self.hop_length) % self.hop_length
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
        window_square = window.square()
        # Vectorised overlap-add. The previous Python loop launched one slice
        # update per frame (roughly 400 launches for a four-second crop), which
        # was measurable in every training forward and backward pass.
        output = F.fold(
            frames.transpose(1, 2),
            output_size=(1, output_length),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze(1).squeeze(1)
        weight = F.fold(
            window_square[None, :, None].expand(1, self.win_length, frame_count),
            output_size=(1, output_length),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()
        # The extreme Hann endpoints have near-zero overlap weight. Dividing
        # prefix-dependent FFT round-off by those tiny values can amplify a few
        # boundary samples even though the spectra agree. Treat samples with
        # less than 0.5% overlap support as undefined edge padding. This masks
        # at most the outer 27 samples (1.7 ms at 16 kHz), while keeping
        # whole-file and arbitrarily chunked synthesis numerically equivalent.
        minimum_overlap = 5e-3
        supported = weight >= minimum_overlap
        output = torch.where(
            supported[None],
            output / weight.clamp_min(minimum_overlap),
            torch.zeros_like(output),
        )
        return F.pad(output, (0, max(0, length - output.shape[-1])))[:, :length]

    def _rolling_noise(self, features: torch.Tensor) -> torch.Tensor:
        sequence = features.mean(dim=-2).transpose(1, 2)
        # Four 10-ms frames, current frame included; left replication is causal.
        sequence = F.pad(sequence.transpose(1, 2), (3, 0), mode="replicate")
        sequence = F.avg_pool1d(sequence, kernel_size=4, stride=1).transpose(1, 2)
        return self.noise_encoder(sequence)

    def _forward_waveform(self, noisy: torch.Tensor, pad_end: bool) -> CausalAuxVQMambaV5Output:
        if noisy.ndim != 3 or noisy.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, T], got {tuple(noisy.shape)}")
        spectrum, original_length = self._analysis(noisy.squeeze(1), pad_end=pad_end)
        if spectrum.shape[-1] == 0:
            raise ValueError("At least one complete analysis window is required")
        magnitude = spectrum.abs().clamp_min(1e-7)
        noisy_phase = torch.angle(spectrum)
        inputs = torch.stack(
            (magnitude.pow(self.magnitude_power), torch.cos(noisy_phase), torch.sin(noisy_phase)),
            dim=1,
        )
        features, full_skip = self.encoder(inputs)
        continuous_noise = self._rolling_noise(features)
        raw_vq = self.noise_vq(continuous_noise)
        probabilities = F.one_hot(raw_vq.indices, self.noise_vq.codebook_size).float().mean((0, 1))
        usage_kl = (probabilities * (probabilities.clamp_min(1e-8).log() + math.log(self.noise_vq.codebook_size))).sum()
        # This head is training-only: forcing the selected prototypes to
        # reconstruct the noise spectrum makes the codebook interpretable,
        # without putting it on the enhancement path.
        noise_prediction = self.noise_predictor(raw_vq.quantized).transpose(1, 2)
        flat_noise = continuous_noise.reshape(-1, self.noise_dim)
        codebook = self.noise_vq.codebook.to(flat_noise)
        distances = (
            flat_noise.square().sum(1, keepdim=True)
            - 2.0 * flat_noise @ codebook.transpose(0, 1)
            + codebook.square().sum(1)
        )
        code_posterior = torch.softmax(-distances, dim=-1).view(
            *continuous_noise.shape[:2], self.noise_vq.codebook_size
        )

        adapter_strength = continuous_noise.new_tensor(0.0)
        adapter_condition = None
        if self.vq_mode == "bounded_adapter":
            adapter_condition, adapter_strength = self.adapter(raw_vq.quantized, self.training)

        current = features
        for _ in range(self.refinement_passes):
            current = self.cell(current, continuous_noise)
            if adapter_condition is not None:
                scale, shift = adapter_condition.transpose(1, 2).chunk(2, dim=1)
                scale = F.interpolate(scale, size=current.shape[-1], mode="nearest")
                shift = F.interpolate(shift, size=current.shape[-1], mode="nearest")
                current = current * (1.0 + scale[:, :, None]) + shift[:, :, None]

        magnitude_mask, phase_vector = self.decoder(current, full_skip)
        estimated_magnitude = magnitude * magnitude_mask
        phase_residual = torch.atan2(phase_vector[:, 0], phase_vector[:, 1])
        predicted_phase = noisy_phase + phase_residual
        enhanced_spectrum = torch.polar(estimated_magnitude, predicted_phase)
        enhanced = self._synthesis(enhanced_spectrum, original_length).unsqueeze(1)
        zero = enhanced.new_tensor(0.0)
        vq = AuxiliaryVQOutput(
            loss=raw_vq.loss + self.vq_usage_weight * usage_kl,
            commitment_loss=raw_vq.commitment_loss,
            usage_kl=usage_kl,
            reconstruction_loss=zero,
            perplexity=raw_vq.perplexity,
            active_fraction=raw_vq.active_fraction,
            dead_fraction=raw_vq.dead_fraction,
        )
        return CausalAuxVQMambaV5Output(
            enhanced=enhanced,
            estimated_magnitude=estimated_magnitude,
            predicted_phase=predicted_phase,
            phase_candidate=predicted_phase,
            magnitude_mask=magnitude_mask,
            continuous_noise_state=continuous_noise,
            code_indices=raw_vq.indices,
            code_posterior=code_posterior,
            code_perplexity=raw_vq.perplexity,
            vq_adapter_strength=adapter_strength,
            noise_prediction=noise_prediction,
            encoder_features=features,
            mamba_features=current,
            vq=vq,
        )

    def forward(self, noisy: torch.Tensor) -> CausalAuxVQMambaV5Output:
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
        result = self._forward_waveform(state.waveform[..., : (complete_frames - 1) * self.hop_length + self.win_length], False)
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
    "AuxiliaryVQOutput",
    "CausalAuxVQMambaV5",
    "CausalAuxVQMambaV5Output",
    "V5StreamState",
]
