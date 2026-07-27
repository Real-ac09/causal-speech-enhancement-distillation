from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .causal_aux_vq_mamba_v5 import AuxiliaryVQOutput, CausalAuxVQMambaV5, V5StreamState
from .causal_aux_vq_mamba_v51 import FrameGroupNorm
from .complex_cnvqg_model import TemporalStack
from .streaming_hybrid_v2 import CausalConv2d, EMANoiseVectorQuantizer


def _norm_act(channels: int) -> nn.Sequential:
    return nn.Sequential(FrameGroupNorm(channels), nn.SiLU())


@dataclass
class PredictiveNoiseVQMambaV8Output:
    enhanced: torch.Tensor
    estimated_magnitude: torch.Tensor
    predicted_phase: torch.Tensor
    phase_candidate: torch.Tensor
    magnitude_mask: torch.Tensor
    magnitude_residual: torch.Tensor
    phase_confidence: torch.Tensor
    continuous_noise_state: torch.Tensor
    code_indices: torch.Tensor
    code_posterior: torch.Tensor
    code_perplexity: torch.Tensor
    vq_adapter_strength: torch.Tensor
    noise_prediction: torch.Tensor
    encoder_features: torch.Tensor
    mamba_features: torch.Tensor
    speech_spectrum: torch.Tensor
    noise_spectrum: torch.Tensor
    speech_mask: torch.Tensor
    noise_mask: torch.Tensor
    mixture_residual: torch.Tensor
    prototype_logits: torch.Tensor
    prototype_prediction_loss: torch.Tensor
    vq: AuxiliaryVQOutput

    @property
    def estimated_phase(self) -> torch.Tensor:
        return self.predicted_phase

    @property
    def noise_state(self) -> torch.Tensor:
        return self.continuous_noise_state


class SingleDownsampleCausalEncoder(nn.Module):
    """Detail-preserving encoder with exactly one frequency reduction."""

    def __init__(self, channels: int, time_kernel_size: int = 3) -> None:
        super().__init__()
        detail_channels = channels // 2
        self.detail = nn.Sequential(
            CausalConv2d(3, detail_channels, (5, time_kernel_size)),
            _norm_act(detail_channels),
            CausalConv2d(
                detail_channels,
                detail_channels,
                (3, time_kernel_size),
                groups=detail_channels,
            ),
            nn.Conv2d(detail_channels, detail_channels, 1),
            _norm_act(detail_channels),
        )
        self.down = nn.Sequential(
            nn.Conv2d(
                detail_channels,
                channels,
                kernel_size=(4, 1),
                stride=(2, 1),
                padding=(1, 0),
            ),
            _norm_act(channels),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        detail = self.detail(features)
        return self.down(detail), detail


class SelectiveTemporalBottleneck(nn.Module):
    """Local causal modelling followed by Mamba only along the time axis."""

    def __init__(
        self,
        channels: int,
        noise_dim: int,
        temporal_layers: int,
        use_mamba: bool,
        d_state: int,
        d_conv: int,
        expand: int,
        temporal_core: str | None = None,
        temporal_hidden_dim: int | None = None,
        time_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.local = nn.Sequential(
            CausalConv2d(
                channels,
                channels,
                (3, time_kernel_size),
                groups=channels,
            ),
            nn.Conv2d(channels, channels, 1),
            _norm_act(channels),
        )
        self.noise_condition = nn.Linear(noise_dim, channels * 2)
        self.temporal = TemporalStack(
            dim=channels,
            layers=temporal_layers,
            use_mamba=use_mamba,
            mamba_d_state=d_state,
            mamba_d_conv=d_conv,
            mamba_expand=expand,
            core=temporal_core,
            gru_hidden_dim=temporal_hidden_dim,
        )
        self.temporal_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, features: torch.Tensor, noise_state: torch.Tensor) -> torch.Tensor:
        current = features + self.local(features)
        condition = self.noise_condition(noise_state).transpose(1, 2)
        scale, shift = condition.chunk(2, dim=1)
        current = current * (1.0 + 0.05 * torch.tanh(scale[:, :, None]))
        current = current + 0.05 * shift[:, :, None]

        batch, channels, bins, frames = current.shape
        sequence = current.permute(0, 2, 3, 1).reshape(batch * bins, frames, channels)
        temporal = self.temporal(sequence)
        temporal = temporal.view(batch, bins, frames, channels).permute(0, 3, 1, 2)
        return current + torch.tanh(self.temporal_scale) * temporal


class MixtureConsistentComplexDecoder(nn.Module):
    """Legacy joint masks or a hybrid ratio/additive magnitude decoder."""

    def __init__(
        self,
        channels: int,
        mask_bound: float = 2.0,
        reconstruction_mode: str = "mixture_consistent_complex",
        magnitude_power: float = 0.3,
        magnitude_log_gain_bound: float = 4.0,
        magnitude_residual_bound: float = 2.0,
        scale_preserving_detail: bool = False,
        time_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        modes = {
            "mixture_consistent_complex",
            "hybrid_magnitude_residual",
            "direct_scalar_mask",
        }
        if reconstruction_mode not in modes:
            raise ValueError(
                f"Unknown reconstruction_mode={reconstruction_mode!r}; expected one of {sorted(modes)}"
            )
        detail_channels = channels // 2
        self.mask_bound = float(mask_bound)
        self.reconstruction_mode = reconstruction_mode
        self.magnitude_power = float(magnitude_power)
        self.magnitude_log_gain_bound = float(magnitude_log_gain_bound)
        self.magnitude_residual_bound = float(magnitude_residual_bound)
        self.scale_preserving_detail = bool(scale_preserving_detail)
        if not 0.0 < self.magnitude_power <= 1.0:
            raise ValueError("magnitude_power must be in (0, 1]")
        if self.magnitude_log_gain_bound <= 0.0 or self.magnitude_residual_bound <= 0.0:
            raise ValueError("hybrid magnitude bounds must be positive")
        self.latent = nn.Sequential(nn.Conv2d(channels, detail_channels, 1), _norm_act(detail_channels))
        self.fuse = nn.Sequential(
            CausalConv2d(
                detail_channels * 2,
                detail_channels,
                (5, time_kernel_size),
            ),
            _norm_act(detail_channels),
            CausalConv2d(
                detail_channels,
                detail_channels,
                (3, time_kernel_size),
                groups=detail_channels,
            ),
            nn.Conv2d(detail_channels, detail_channels, 1),
            _norm_act(detail_channels),
        )
        # Legacy: speech real/imag, noise real/imag, projection allocation.
        # Hybrid: log-ratio, additive compressed residual, phase real/imag.
        controls = {
            "mixture_consistent_complex": 5,
            "hybrid_magnitude_residual": 4,
            "direct_scalar_mask": 1,
        }[reconstruction_mode]
        self.mask_head = nn.Conv2d(detail_channels, controls, 1)
        nn.init.zeros_(self.mask_head.weight)
        nn.init.zeros_(self.mask_head.bias)
        self.scale_head = nn.Conv2d(3, controls, 1) if self.scale_preserving_detail else None
        if self.scale_head is not None:
            nn.init.zeros_(self.scale_head.weight)
            nn.init.zeros_(self.scale_head.bias)

    def _raw_controls(
        self, decoded: torch.Tensor, noisy_magnitude: torch.Tensor | None
    ) -> torch.Tensor:
        raw = self.mask_head(decoded).float()
        if self.scale_head is None:
            return raw
        if noisy_magnitude is None:
            raise ValueError("scale_preserving_detail requires noisy_magnitude")
        compressed = noisy_magnitude.float().clamp_min(1e-7).pow(self.magnitude_power)
        frame_level = compressed.mean(dim=-2, keepdim=True).clamp_min(1e-5)
        relative = compressed / frame_level
        scale_features = torch.stack(
            (
                torch.log1p(compressed),
                torch.log1p(frame_level).expand_as(compressed),
                torch.log1p(relative),
            ),
            dim=1,
        )
        return raw + self.scale_head(scale_features).float()

    def forward(
        self,
        latent: torch.Tensor,
        detail: torch.Tensor,
        noisy_magnitude: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        up = F.interpolate(self.latent(latent), size=detail.shape[-2:], mode="nearest")
        decoded = self.fuse(torch.cat((up, detail), dim=1))
        # PyTorch does not define complex-BF16 tensors. Keep the network in
        # mixed precision and promote only the five reconstruction controls at
        # the complex arithmetic boundary.
        raw = self._raw_controls(decoded, noisy_magnitude)
        bound = self.mask_bound
        speech_raw = torch.complex(
            1.0 + bound * torch.tanh(raw[:, 0]),
            bound * torch.tanh(raw[:, 1]),
        )
        noise_raw = torch.complex(
            bound * torch.tanh(raw[:, 2]),
            bound * torch.tanh(raw[:, 3]),
        )
        speech_share = torch.sigmoid(raw[:, 4])
        residual = 1.0 - speech_raw - noise_raw
        speech_mask = speech_raw + speech_share * residual
        noise_mask = noise_raw + (1.0 - speech_share) * residual
        return speech_mask, noise_mask, decoded

    def forward_hybrid(
        self,
        latent: torch.Tensor,
        detail: torch.Tensor,
        noisy_magnitude: torch.Tensor,
        noisy_phase: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Estimate magnitude without imposing a bounded multiplicative ceiling.

        The ratio branch handles attenuation and ordinary gain. The additive
        branch operates in the compressed-magnitude domain and can restore
        energy even where the noisy magnitude is close to zero. Both controls
        are zero-initialised, so the initial reconstruction is exactly the
        causal STFT identity path.
        """
        if self.reconstruction_mode != "hybrid_magnitude_residual":
            raise RuntimeError("forward_hybrid requires hybrid_magnitude_residual mode")
        up = F.interpolate(self.latent(latent), size=detail.shape[-2:], mode="nearest")
        decoded = self.fuse(torch.cat((up, detail), dim=1))
        raw = self._raw_controls(decoded, noisy_magnitude)

        compressed_noisy = noisy_magnitude.float().clamp_min(1e-7).pow(self.magnitude_power)
        log_gain = self.magnitude_log_gain_bound * torch.tanh(raw[:, 0])
        ratio_base = compressed_noisy * torch.exp(self.magnitude_power * log_gain)
        # A per-frame reference keeps the additive path scale-aware without
        # making it vanish in individual spectral nulls.
        frame_reference = compressed_noisy.mean(dim=-2, keepdim=True).clamp_min(1e-5)
        compressed_residual = (
            self.magnitude_residual_bound * frame_reference * torch.tanh(raw[:, 1])
        )
        estimated_compressed = (ratio_base + compressed_residual).clamp_min(1e-7)
        estimated_magnitude = estimated_compressed.pow(1.0 / self.magnitude_power)

        phase_real = 1.0 + self.mask_bound * torch.tanh(raw[:, 2])
        phase_imag = self.mask_bound * torch.tanh(raw[:, 3])
        phase_residual = torch.atan2(phase_imag, phase_real)
        candidate_phase = noisy_phase.float() + phase_residual
        magnitude_mask = estimated_magnitude / noisy_magnitude.float().clamp_min(1e-7)
        return (
            estimated_magnitude,
            candidate_phase,
            magnitude_mask,
            compressed_residual,
            decoded,
        )

    def forward_direct(
        self,
        latent: torch.Tensor,
        detail: torch.Tensor,
        noisy_magnitude: torch.Tensor,
        noisy_phase: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Predict the supervised scalar mask without complex intermediates."""
        if self.reconstruction_mode != "direct_scalar_mask":
            raise RuntimeError("forward_direct requires direct_scalar_mask mode")
        up = F.interpolate(self.latent(latent), size=detail.shape[-2:], mode="nearest")
        decoded = self.fuse(torch.cat((up, detail), dim=1))
        raw = self._raw_controls(decoded, noisy_magnitude)
        # Exactly one at zero logits, bounded in (0, 2), with a non-zero
        # derivative at initialization. The cap-1 oracle is fully contained.
        magnitude_mask = 2.0 * torch.sigmoid(-raw[:, 0])
        estimated_magnitude = noisy_magnitude.float() * magnitude_mask
        residual = magnitude_mask.new_zeros(magnitude_mask.shape)
        return estimated_magnitude, noisy_phase.float(), magnitude_mask, residual, decoded


class PredictiveNoiseVQMambaV8(nn.Module):
    """Causal complex enhancer with auxiliary predictive noise prototypes.

    Quantized states never enter the enhancement path. They supervise the
    continuous causal noise state through noise reconstruction, code usage,
    and multi-horizon prediction of upcoming noise prototypes.
    """

    PRESETS = {
        "student": {"channels": 232, "noise_dim": 64, "layers": 1, "cap": 1_100_000},
        "teacher": {"channels": 320, "noise_dim": 96, "layers": 2, "cap": 2_700_000},
    }

    def __init__(
        self,
        variant: str = "student",
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 320,
        magnitude_power: float = 0.3,
        channels: int | None = None,
        noise_dim: int | None = None,
        temporal_layers: int | None = None,
        codebook_size: int = 32,
        prediction_horizons: Sequence[int] = (1, 2, 4),
        auxiliary_vq: bool = True,
        vq_commitment_weight: float = 0.02,
        vq_usage_weight: float = 0.005,
        prototype_prediction_weight: float = 0.02,
        posterior_temperature: float = 0.25,
        prediction_label_smoothing: float = 0.02,
        phase_residual_scale: float = 1.0,
        use_mamba: bool = True,
        temporal_core: str | None = None,
        temporal_hidden_dim: int | None = None,
        time_kernel_size: int = 3,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mask_bound: float = 2.0,
        reconstruction_mode: str = "mixture_consistent_complex",
        magnitude_log_gain_bound: float = 4.0,
        magnitude_residual_bound: float = 2.0,
        scale_preserving_detail: bool = False,
        enforce_parameter_cap: bool = True,
    ) -> None:
        super().__init__()
        if variant not in self.PRESETS:
            raise ValueError(f"Unknown V8 variant: {variant}")
        if n_fft < win_length or hop_length > win_length:
            raise ValueError("V8 requires n_fft >= win_length >= hop_length")
        if time_kernel_size not in {1, 3}:
            raise ValueError("time_kernel_size must be 1 or 3")
        horizons = tuple(sorted({int(value) for value in prediction_horizons}))
        if not horizons or horizons[0] < 1:
            raise ValueError("prediction_horizons must contain positive frame offsets")
        preset = self.PRESETS[variant]
        self.variant = variant
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.magnitude_power = float(magnitude_power)
        self.channels = int(channels or preset["channels"])
        self.noise_dim = int(noise_dim or preset["noise_dim"])
        self.temporal_layers = int(temporal_layers or preset["layers"])
        self.parameter_cap = int(preset["cap"])
        self.algorithmic_latency_samples = self.win_length
        self.auxiliary_vq = bool(auxiliary_vq)
        self.prediction_horizons = horizons
        self.vq_usage_weight = float(vq_usage_weight)
        self.prototype_prediction_weight = float(prototype_prediction_weight)
        self.posterior_temperature = float(posterior_temperature)
        self.prediction_label_smoothing = float(prediction_label_smoothing)
        self.phase_residual_scale = float(phase_residual_scale)
        self.reconstruction_mode = str(reconstruction_mode)
        self.temporal_core = temporal_core or ("mamba" if use_mamba else "causal_conv")
        self.temporal_hidden_dim = int(temporal_hidden_dim or self.channels)
        self.time_kernel_size = int(time_kernel_size)
        if self.posterior_temperature <= 0.0:
            raise ValueError("posterior_temperature must be positive")
        if not 0.0 <= self.prediction_label_smoothing < 1.0:
            raise ValueError("prediction_label_smoothing must be in [0, 1)")
        if not 0.0 <= self.phase_residual_scale <= 1.0:
            raise ValueError("phase_residual_scale must be in [0, 1]")

        self.register_buffer(
            "analysis_window", torch.hann_window(self.win_length, periodic=True), persistent=False
        )
        self.encoder = SingleDownsampleCausalEncoder(
            self.channels, time_kernel_size=self.time_kernel_size
        )
        self.noise_encoder = nn.Sequential(
            nn.Linear(self.channels, self.noise_dim), nn.SiLU(), nn.LayerNorm(self.noise_dim)
        )
        self.bottleneck = SelectiveTemporalBottleneck(
            self.channels,
            self.noise_dim,
            self.temporal_layers,
            use_mamba,
            mamba_d_state,
            mamba_d_conv,
            mamba_expand,
            temporal_core=self.temporal_core,
            temporal_hidden_dim=self.temporal_hidden_dim,
            time_kernel_size=self.time_kernel_size,
        )
        self.decoder = MixtureConsistentComplexDecoder(
            self.channels,
            mask_bound,
            reconstruction_mode=self.reconstruction_mode,
            magnitude_power=self.magnitude_power,
            magnitude_log_gain_bound=magnitude_log_gain_bound,
            magnitude_residual_bound=magnitude_residual_bound,
            scale_preserving_detail=scale_preserving_detail,
            time_kernel_size=self.time_kernel_size,
        )
        self.noise_vq = EMANoiseVectorQuantizer(
            codebook_size=codebook_size,
            code_dim=self.noise_dim,
            update_interval=1,
            commitment_weight=vq_commitment_weight,
        )
        self.noise_predictor = nn.Sequential(
            nn.Linear(self.noise_dim, self.noise_dim),
            nn.SiLU(),
            nn.Linear(self.noise_dim, self.n_fft // 2 + 1),
            nn.Softplus(),
        )
        self.prototype_predictor = nn.Linear(
            self.noise_dim, len(self.prediction_horizons) * codebook_size
        )
        self.temporal = self.bottleneck.temporal

        count = self.parameter_count()
        if enforce_parameter_cap and count > self.parameter_cap:
            raise ValueError(
                f"V8 {variant} has {count:,} parameters, exceeding {self.parameter_cap:,}"
            )

    # Reuse the verified uncentred analysis, overlap-add, and correctness-first
    # streaming contract without inheriting the obsolete V5 architecture.
    _analysis = CausalAuxVQMambaV5._analysis
    _synthesis = CausalAuxVQMambaV5._synthesis
    init_stream_state = CausalAuxVQMambaV5.init_stream_state
    forward_chunk = CausalAuxVQMambaV5.forward_chunk
    flush = CausalAuxVQMambaV5.flush

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _rolling_noise(self, features: torch.Tensor) -> torch.Tensor:
        sequence = features.mean(dim=-2).transpose(1, 2)
        sequence = F.pad(sequence.transpose(1, 2), (3, 0), mode="replicate")
        sequence = F.avg_pool1d(sequence, kernel_size=4, stride=1).transpose(1, 2)
        return self.noise_encoder(sequence)

    def _prototype_prediction(
        self, noise_state: torch.Tensor, posterior: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, _ = noise_state.shape
        codes = self.noise_vq.codebook_size
        logits = self.prototype_predictor(noise_state).view(
            batch, frames, len(self.prediction_horizons), codes
        )
        losses = []
        for horizon_index, horizon in enumerate(self.prediction_horizons):
            if frames > horizon:
                prediction = logits[:, :-horizon, horizon_index].float()
                target = posterior[:, horizon:].detach().float()
                smoothing = self.prediction_label_smoothing
                target = (1.0 - smoothing) * target + smoothing / codes
                losses.append(-(target * F.log_softmax(prediction, dim=-1)).sum(-1).mean())
        loss = torch.stack(losses).mean() if losses else noise_state.new_tensor(0.0)
        return logits, loss

    def _forward_waveform(
        self, noisy: torch.Tensor, pad_end: bool
    ) -> PredictiveNoiseVQMambaV8Output:
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
        encoded, detail = self.encoder(inputs)
        continuous_noise = self._rolling_noise(encoded)
        current = self.bottleneck(encoded, continuous_noise)
        if self.reconstruction_mode == "hybrid_magnitude_residual":
            (
                raw_estimated_magnitude,
                candidate_phase,
                magnitude_mask,
                magnitude_residual,
                decoded,
            ) = self.decoder.forward_hybrid(current, detail, magnitude, noisy_phase)
            raw_speech_spectrum = torch.polar(raw_estimated_magnitude, candidate_phase)
        elif self.reconstruction_mode == "direct_scalar_mask":
            (
                raw_estimated_magnitude,
                candidate_phase,
                magnitude_mask,
                magnitude_residual,
                decoded,
            ) = self.decoder.forward_direct(current, detail, magnitude, noisy_phase)
            raw_speech_spectrum = torch.polar(raw_estimated_magnitude, candidate_phase)
        else:
            speech_mask, noise_mask, decoded = self.decoder(current, detail, magnitude)
            raw_speech_spectrum = speech_mask * spectrum
            candidate_phase = torch.angle(raw_speech_spectrum)
            magnitude_mask = speech_mask.abs()
            magnitude_residual = magnitude.new_zeros(magnitude.shape)
        phase_residual = torch.atan2(
            torch.sin(candidate_phase - noisy_phase),
            torch.cos(candidate_phase - noisy_phase),
        )
        predicted_phase = noisy_phase + self.phase_residual_scale * phase_residual
        speech_spectrum = torch.polar(raw_speech_spectrum.abs(), predicted_phase)
        # Preserve exact additivity after bounded phase selection.
        noise_spectrum = spectrum - speech_spectrum
        valid = spectrum.abs() > 1e-7
        speech_mask = torch.where(valid, speech_spectrum / spectrum, torch.zeros_like(spectrum))
        noise_mask = 1.0 - speech_mask
        mixture_residual = spectrum - speech_spectrum - noise_spectrum
        enhanced = self._synthesis(speech_spectrum, original_length).unsqueeze(1)
        estimated_magnitude = speech_spectrum.abs().clamp_min(1e-7)

        zero = enhanced.new_tensor(0.0)
        if self.auxiliary_vq:
            raw_vq = self.noise_vq(continuous_noise)
            flat = continuous_noise.reshape(-1, self.noise_dim)
            codebook = self.noise_vq.codebook.to(flat)
            distances = (
                flat.square().sum(1, keepdim=True)
                - 2.0 * flat @ codebook.transpose(0, 1)
                + codebook.square().sum(1)
            )
            posterior = torch.softmax(
                -distances / self.posterior_temperature, dim=-1
            ).view(
                *continuous_noise.shape[:2], self.noise_vq.codebook_size
            )
            soft_usage = posterior.mean((0, 1))
            usage_kl = (
                soft_usage
                * (soft_usage.clamp_min(1e-8).log() + math.log(self.noise_vq.codebook_size))
            ).sum()
            prototype_logits, prediction_loss = self._prototype_prediction(
                continuous_noise, posterior
            )
            noise_prediction = self.noise_predictor(raw_vq.quantized).transpose(1, 2)
            vq_loss = (
                raw_vq.loss
                + self.vq_usage_weight * usage_kl
                + self.prototype_prediction_weight * prediction_loss
            )
            vq = AuxiliaryVQOutput(
                loss=vq_loss,
                commitment_loss=raw_vq.commitment_loss,
                usage_kl=usage_kl,
                reconstruction_loss=zero,
                perplexity=raw_vq.perplexity,
                active_fraction=raw_vq.active_fraction,
                dead_fraction=raw_vq.dead_fraction,
            )
            indices = raw_vq.indices
        else:
            indices = torch.full(
                continuous_noise.shape[:2], -1, dtype=torch.long, device=noisy.device
            )
            posterior = continuous_noise.new_zeros(
                *continuous_noise.shape[:2], self.noise_vq.codebook_size
            )
            prototype_logits = continuous_noise.new_zeros(
                *continuous_noise.shape[:2], len(self.prediction_horizons), self.noise_vq.codebook_size
            )
            prediction_loss = zero
            noise_prediction = noise_spectrum.abs()
            vq = AuxiliaryVQOutput(zero, zero, zero, zero, zero, zero, zero)

        return PredictiveNoiseVQMambaV8Output(
            enhanced=enhanced,
            estimated_magnitude=estimated_magnitude,
            predicted_phase=predicted_phase,
            phase_candidate=candidate_phase,
            # Use the decoder's stable mask rather than recomputing it through
            # complex division at spectral nulls. The latter is numerically
            # harmless for inference after torch.where, but creates NaN
            # gradients when the mask itself is directly supervised.
            magnitude_mask=magnitude_mask,
            magnitude_residual=magnitude_residual,
            phase_confidence=torch.ones_like(estimated_magnitude),
            continuous_noise_state=continuous_noise,
            code_indices=indices,
            code_posterior=posterior,
            code_perplexity=vq.perplexity,
            vq_adapter_strength=zero,
            noise_prediction=noise_prediction,
            encoder_features=encoded,
            mamba_features=current,
            speech_spectrum=speech_spectrum,
            noise_spectrum=noise_spectrum,
            speech_mask=speech_mask,
            noise_mask=noise_mask,
            mixture_residual=mixture_residual,
            prototype_logits=prototype_logits,
            prototype_prediction_loss=prediction_loss,
            vq=vq,
        )

    def forward(self, noisy: torch.Tensor) -> PredictiveNoiseVQMambaV8Output:
        return self._forward_waveform(noisy, pad_end=True)


__all__ = ["PredictiveNoiseVQMambaV8", "PredictiveNoiseVQMambaV8Output", "V5StreamState"]
