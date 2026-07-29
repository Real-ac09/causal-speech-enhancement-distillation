from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from .auxiliary_gated_tf_mamba_v43 import AuxiliaryGatedTFMambaV43
from .causal_aux_vq_mamba_v5 import CausalAuxVQMambaV5, V5StreamState
from .noise_adaptive_tf_mamba_v41 import NoiseAdaptiveTFV41Output


class CausalSameConv2d(nn.Conv2d):
    """State-compatible Conv2d with left-only padding on time."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        freq_kernel, time_kernel = self.kernel_size
        freq_dilation, time_dilation = self.dilation
        total_frequency = freq_dilation * (freq_kernel - 1)
        frequency_left = total_frequency // 2
        frequency_right = total_frequency - frequency_left
        time_left = time_dilation * (time_kernel - 1)
        features = F.pad(
            features, (time_left, 0, frequency_left, frequency_right)
        )
        return F.conv2d(
            features,
            self.weight,
            self.bias,
            self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
        )


class CausalSameConvTranspose2d(nn.ConvTranspose2d):
    """State-compatible frequency upsampler with causal time alignment."""

    def forward(self, features: torch.Tensor, output_size=None) -> torch.Tensor:
        target_frames = features.shape[-1] * self.stride[1]
        output = F.conv_transpose2d(
            features,
            self.weight,
            self.bias,
            self.stride,
            padding=(self.padding[0], 0),
            output_padding=self.output_padding,
            groups=self.groups,
            dilation=self.dilation,
        )
        return output[..., :target_frames]


class CausalCumulativeGroupNorm(nn.GroupNorm):
    """GroupNorm over the current and preceding frames only.

    This retains the scale of V4.3's utterance GroupNorm much more closely
    than frame-local normalization while preventing future-frame leakage.
    """

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            return super().forward(features)
        batch, channels, bins, frames = features.shape
        grouped = features.reshape(
            batch, self.num_groups, channels // self.num_groups, bins, frames
        )
        frame_sum = grouped.sum(dim=(2, 3))
        frame_square_sum = grouped.square().sum(dim=(2, 3))
        cumulative_sum = frame_sum.cumsum(dim=-1)
        cumulative_square_sum = frame_square_sum.cumsum(dim=-1)
        values_per_frame = (channels // self.num_groups) * bins
        count = torch.arange(
            1, frames + 1, device=features.device, dtype=features.dtype
        ) * values_per_frame
        mean = cumulative_sum / count
        variance = (cumulative_square_sum / count - mean.square()).clamp_min(0.0)
        normalized = (grouped - mean[:, :, None, None, :]) * torch.rsqrt(
            variance[:, :, None, None, :] + self.eps
        )
        normalized = normalized.reshape(batch, channels, bins, frames)
        if self.weight is not None:
            normalized = normalized * self.weight[None, :, None, None]
            normalized = normalized + self.bias[None, :, None, None]
        return normalized


def _causalize(module: nn.Module) -> None:
    """Replace temporal leakage sources without changing state-dict keys."""
    for name, child in list(module.named_children()):
        replacement: nn.Module | None = None
        if type(child) is nn.Conv2d:
            replacement = CausalSameConv2d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                child.stride,
                padding=0,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                padding_mode="zeros",
            )
        elif type(child) is nn.ConvTranspose2d:
            replacement = CausalSameConvTranspose2d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                child.stride,
                padding=child.padding,
                output_padding=child.output_padding,
                groups=child.groups,
                bias=child.bias is not None,
                dilation=child.dilation,
            )
        elif type(child) is nn.GroupNorm:
            replacement = CausalCumulativeGroupNorm(
                child.num_groups,
                child.num_channels,
                eps=child.eps,
                affine=child.affine,
            )
        if replacement is not None:
            replacement.load_state_dict(copy.deepcopy(child.state_dict()))
            setattr(module, name, replacement)
        else:
            _causalize(child)


class CausalAuxiliaryGatedTFMambaV410(AuxiliaryGatedTFMambaV43):
    """Minimal real-time conversion of V4.3 with checkpoint-compatible weights.

    Frequency processing remains unrestricted within the current frame. Time
    convolutions, normalization, noise conditioning, and adaptive halting are
    made causal. Quantized states are retained for checkpoint compatibility but
    never enter enhancement.
    """

    def __init__(
        self,
        *args,
        win_length: int = 320,
        hop_length: int = 160,
        center: bool = False,
        deployment_vq: bool = False,
        **kwargs,
    ) -> None:
        if center:
            raise ValueError("V4.10 requires center=false")
        super().__init__(
            *args,
            win_length=win_length,
            hop_length=hop_length,
            center=False,
            **kwargs,
        )
        self.deployment_vq = bool(deployment_vq)
        if self.deployment_vq:
            raise ValueError("V4.10 baseline requires deployment_vq=false")
        self.algorithmic_latency_samples = self.win_length
        self.register_buffer(
            "analysis_window",
            torch.hann_window(self.win_length, periodic=True),
            persistent=False,
        )
        _causalize(self.encoder)
        _causalize(self.cell)
        _causalize(self.decoder)
        _causalize(self.magnitude_branch)
        _causalize(self.phase_branch)
        _causalize(self.magnitude_to_phase)
        _causalize(self.magnitude_head)
        _causalize(self.phase_head)

    _analysis = CausalAuxVQMambaV5._analysis
    _synthesis = CausalAuxVQMambaV5._synthesis
    init_stream_state = CausalAuxVQMambaV5.init_stream_state
    forward_chunk = CausalAuxVQMambaV5.forward_chunk
    flush = CausalAuxVQMambaV5.flush

    def _rolling_noise(self, features: torch.Tensor) -> torch.Tensor:
        sequence = features.mean(dim=-2).transpose(1, 2)
        history = self.noise_segment_frames
        sequence = F.pad(
            sequence.transpose(1, 2), (history - 1, 0), mode="replicate"
        )
        sequence = F.avg_pool1d(sequence, kernel_size=history, stride=1).transpose(1, 2)
        return self.noise_encoder(sequence)

    def _forward_waveform(
        self, noisy: torch.Tensor, pad_end: bool
    ) -> NoiseAdaptiveTFV41Output:
        if noisy.ndim != 3 or noisy.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, T], got {tuple(noisy.shape)}")
        spectrum, length = self._analysis(noisy.squeeze(1), pad_end=pad_end)
        if spectrum.shape[-1] == 0:
            raise ValueError("At least one complete analysis window is required")
        magnitude = spectrum.abs().clamp_min(1e-7)
        noisy_phase = torch.angle(spectrum)
        inputs = torch.stack(
            (
                magnitude.pow(self.magnitude_power),
                torch.cos(noisy_phase),
                torch.sin(noisy_phase),
            ),
            dim=1,
        )
        features, half_skip, full_skip = self.encoder(inputs)
        continuous_noise = self._rolling_noise(features)
        # VQ is intentionally auxiliary/off for the causal performance baseline.
        vq = self._empty_vq(continuous_noise)
        noise_states = continuous_noise
        vq_gate = continuous_noise.new_zeros(self.noise_dim)

        batch, _, _, frames = features.shape
        remaining = features.new_ones(batch, frames)
        accumulated = torch.zeros_like(features)
        halt_probabilities = []
        expected_iterations = features.new_zeros(batch, frames)
        current = features
        pooled_noise = noise_states
        for index in range(self.max_iterations):
            current = self.cell(current, noise_states)
            if self.adaptive_iterations and index < self.max_iterations - 1:
                pooled = current.mean(dim=-2).transpose(1, 2)
                halt = torch.sigmoid(
                    self.halting_head(torch.cat((pooled, pooled_noise), dim=-1)).squeeze(-1)
                )
                probability = remaining * halt
                remaining = remaining * (1.0 - halt)
            elif index == self.max_iterations - 1:
                probability = remaining
                remaining = torch.zeros_like(remaining)
            else:
                probability = torch.zeros_like(remaining)
            halt_probabilities.append(probability)
            accumulated = accumulated + probability[:, None, None, :] * current
            expected_iterations = expected_iterations + probability * float(index + 1)

        decoded = self.decoder(accumulated, half_skip, full_skip)
        magnitude_features = self.magnitude_branch(decoded)
        magnitude_mask = self.mask_max * torch.sigmoid(
            self.magnitude_head(magnitude_features).squeeze(1)
        )
        estimated_magnitude = magnitude * magnitude_mask
        phase_features = self.phase_branch(decoded)
        phase_features = phase_features + torch.tanh(
            self.phase_cross_scale
        ) * self.magnitude_to_phase(magnitude_features)
        phase_output = self.phase_head(phase_features)
        phase_residual = torch.atan2(phase_output[:, 0], phase_output[:, 1] + 1e-8)
        phase_candidate = noisy_phase + phase_residual
        confidence = (
            torch.sigmoid(phase_output[:, 2])
            if self.use_phase_confidence
            else torch.ones_like(phase_residual)
        )
        estimated_phase = noisy_phase + confidence * phase_residual
        enhanced_spectrum = torch.polar(estimated_magnitude, estimated_phase)
        enhanced = self._synthesis(enhanced_spectrum, length).unsqueeze(1)
        noise_prediction = self.noise_predictor(noise_states).transpose(1, 2)
        return NoiseAdaptiveTFV41Output(
            enhanced=enhanced,
            estimated_magnitude=estimated_magnitude,
            estimated_phase=estimated_phase,
            phase_candidate=phase_candidate,
            phase_confidence=confidence,
            magnitude_mask=magnitude_mask,
            noise_state=noise_states,
            code_indices=vq.indices,
            expected_iterations=expected_iterations.mean(dim=-1),
            halting_probabilities=torch.stack(halt_probabilities, dim=-1),
            noise_prediction=noise_prediction,
            vq=vq,
            vq_gate=vq_gate,
        )

    def forward(self, noisy: torch.Tensor) -> NoiseAdaptiveTFV41Output:
        return self._forward_waveform(noisy, pad_end=True)


__all__ = [
    "CausalAuxiliaryGatedTFMambaV410",
    "CausalCumulativeGroupNorm",
    "V5StreamState",
]
