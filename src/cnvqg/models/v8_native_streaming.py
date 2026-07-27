from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .complex_cnvqg_model import (
    ConvTemporalBlock,
    GRUTemporalBlock,
    MambaTemporalBlock,
)
from .predictive_noise_vq_mamba_v8 import PredictiveNoiseVQMambaV8
from .streaming_hybrid_v2 import CausalConv2d


@dataclass
class V8NativeStreamState:
    """Fixed-size state for frame-at-a-time V8 inference."""

    input_buffer: torch.Tensor
    ola_numerator: torch.Tensor
    ola_denominator: torch.Tensor
    convolution: dict[str, torch.Tensor] = field(default_factory=dict)
    temporal_convolution: dict[str, torch.Tensor] = field(default_factory=dict)
    mamba: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    gru: dict[str, torch.Tensor] = field(default_factory=dict)
    noise_history: torch.Tensor | None = None
    input_samples: int = 0
    emitted_samples: int = 0
    finished: bool = False

    def tensor_elements(self) -> int:
        """Return state size independently of the amount of audio processed."""

        tensors = [
            self.input_buffer,
            self.ola_numerator,
            self.ola_denominator,
            *self.convolution.values(),
            *self.temporal_convolution.values(),
            *self.gru.values(),
        ]
        if self.noise_history is not None:
            tensors.append(self.noise_history)
        for convolution, ssm in self.mamba.values():
            tensors.extend((convolution, ssm))
        return sum(tensor.numel() for tensor in tensors)


class V8NativeStreamer:
    """Constant-work streaming inference for ``PredictiveNoiseVQMambaV8``.

    The training/offline model remains unchanged. This adapter evaluates one
    complete causal STFT frame at a time and carries only the state needed by
    causal convolutions, the rolling noise encoder, temporal blocks, and
    overlap-add synthesis.
    """

    def __init__(self, model: PredictiveNoiseVQMambaV8) -> None:
        if type(model) is not PredictiveNoiseVQMambaV8:
            raise TypeError(
                "V8NativeStreamer currently supports PredictiveNoiseVQMambaV8 exactly; "
                "derived architectures need their own validated frame graph"
            )
        if model.training:
            raise ValueError("V8NativeStreamer requires model.eval()")
        self.model = model

    def init_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> V8NativeStreamState:
        model = self.model
        return V8NativeStreamState(
            input_buffer=torch.empty(batch_size, 1, 0, device=device, dtype=dtype),
            ola_numerator=torch.zeros(
                batch_size, model.win_length, device=device, dtype=torch.float32
            ),
            ola_denominator=torch.zeros(
                batch_size, model.win_length, device=device, dtype=torch.float32
            ),
        )

    @staticmethod
    def _empty_like(audio: torch.Tensor) -> torch.Tensor:
        return audio[..., :0]

    def _causal_conv(
        self,
        layer: CausalConv2d,
        features: torch.Tensor,
        state: V8NativeStreamState,
        key: str,
    ) -> torch.Tensor:
        padding = layer.time_padding
        if padding:
            history = state.convolution.get(key)
            if history is None:
                history = features.new_zeros(
                    features.shape[0], features.shape[1], features.shape[2], padding
                )
            context = torch.cat((history, features), dim=-1)
            state.convolution[key] = context[..., -padding:]
        else:
            context = features
        frequency_left, frequency_right = layer.frequency_padding
        context = F.pad(context, (0, 0, frequency_left, frequency_right))
        output = layer.conv(context)
        if output.shape[-1] != 1:
            raise RuntimeError(f"{key} produced {output.shape[-1]} streaming frames")
        return output

    def _sequence(
        self,
        module: nn.Module,
        features: torch.Tensor,
        state: V8NativeStreamState,
        key: str,
    ) -> torch.Tensor:
        if isinstance(module, CausalConv2d):
            return self._causal_conv(module, features, state, key)
        if isinstance(module, nn.Sequential):
            current = features
            for index, child in enumerate(module):
                current = self._sequence(child, current, state, f"{key}.{index}")
            return current
        if isinstance(module, nn.Conv2d) and module.kernel_size[1] != 1:
            raise RuntimeError(f"{key} is not frame-local and has no streaming cache")
        return module(features)

    def _rolling_noise(
        self, features: torch.Tensor, state: V8NativeStreamState
    ) -> torch.Tensor:
        pooled = features.mean(dim=-2)
        if state.noise_history is None:
            context = pooled.expand(-1, -1, 4)
        else:
            context = torch.cat((state.noise_history, pooled), dim=-1)
        state.noise_history = context[..., -3:]
        return self.model.noise_encoder(context.mean(dim=-1).unsqueeze(1))

    def _temporal_block(
        self,
        block: nn.Module,
        sequence: torch.Tensor,
        state: V8NativeStreamState,
        key: str,
    ) -> torch.Tensor:
        if isinstance(block, MambaTemporalBlock):
            normalized = block.norm(sequence)
            recurrent = state.mamba.get(key)
            if recurrent is None:
                recurrent = (
                    normalized.new_zeros(
                        normalized.shape[0], block.mamba.d_inner, block.mamba.d_conv
                    ),
                    normalized.new_zeros(
                        normalized.shape[0], block.mamba.d_inner, block.mamba.d_state
                    ),
                )
            convolution, ssm = recurrent
            mamba = block.mamba
            x, z = mamba.in_proj(normalized.squeeze(1)).chunk(2, dim=-1)
            convolution = torch.roll(convolution, shifts=-1, dims=-1)
            convolution[:, :, -1] = x
            x = (
                convolution * mamba.conv1d.weight.squeeze(1)
            ).sum(dim=-1)
            if mamba.conv1d.bias is not None:
                x = x + mamba.conv1d.bias
            x = F.silu(x)
            projected = mamba.x_proj(x)
            dt, b_term, c_term = torch.split(
                projected, (mamba.dt_rank, mamba.d_state, mamba.d_state), dim=-1
            )
            dt = F.softplus(F.linear(dt, mamba.dt_proj.weight, mamba.dt_proj.bias))
            a_term = -torch.exp(mamba.A_log.float()).to(x.dtype)
            transition = torch.exp(dt.unsqueeze(-1) * a_term.unsqueeze(0))
            ssm = transition * ssm + (
                dt.unsqueeze(-1) * b_term.unsqueeze(1) * x.unsqueeze(-1)
            )
            transformed = (ssm * c_term.unsqueeze(1)).sum(-1)
            transformed = transformed + mamba.D.to(x.dtype) * x
            transformed = mamba.out_proj(transformed * F.silu(z)).unsqueeze(1)
            state.mamba[key] = (convolution, ssm)
            return sequence + block.layer_scale * transformed

        if isinstance(block, GRUTemporalBlock):
            normalized = block.input_projection(block.norm(sequence))
            hidden = state.gru.get(key)
            if hidden is None:
                hidden = normalized.new_zeros(
                    1, normalized.shape[0], block.hidden_dim
                )
            transformed, hidden = block.gru(normalized, hidden)
            state.gru[key] = hidden
            return sequence + block.layer_scale * block.output_projection(transformed)

        if isinstance(block, ConvTemporalBlock):
            normalized = block.norm(sequence).transpose(1, 2)
            history = state.temporal_convolution.get(key)
            if history is None:
                history = normalized.new_zeros(
                    normalized.shape[0], normalized.shape[1], block.pad
                )
            context = torch.cat((history, normalized), dim=-1)
            state.temporal_convolution[key] = context[..., -block.pad:]
            transformed = block.net(context).transpose(1, 2)
            if transformed.shape[1] != 1:
                raise RuntimeError(f"{key} produced {transformed.shape[1]} temporal frames")
            return sequence + transformed

        raise TypeError(f"Unsupported temporal block: {type(block).__name__}")

    def _temporal(
        self, features: torch.Tensor, state: V8NativeStreamState
    ) -> torch.Tensor:
        batch, channels, bins, frames = features.shape
        if frames != 1:
            raise RuntimeError("Native V8 temporal inference expects exactly one frame")
        sequence = features.permute(0, 2, 3, 1).reshape(batch * bins, 1, channels)
        for index, block in enumerate(self.model.bottleneck.temporal.blocks):
            sequence = self._temporal_block(block, sequence, state, f"temporal.{index}")
        return sequence.view(batch, bins, 1, channels).permute(0, 3, 1, 2)

    def _decode(
        self,
        latent: torch.Tensor,
        detail: torch.Tensor,
        noisy_magnitude: torch.Tensor,
        noisy_phase: torch.Tensor,
        state: V8NativeStreamState,
    ) -> torch.Tensor:
        decoder = self.model.decoder
        up = self._sequence(decoder.latent, latent, state, "decoder.latent")
        up = F.interpolate(up, size=detail.shape[-2:], mode="nearest")
        decoded = self._sequence(
            decoder.fuse, torch.cat((up, detail), dim=1), state, "decoder.fuse"
        )
        raw = decoder._raw_controls(decoded, noisy_magnitude)

        if decoder.reconstruction_mode == "direct_scalar_mask":
            magnitude = noisy_magnitude.float() * (2.0 * torch.sigmoid(-raw[:, 0]))
            candidate_phase = noisy_phase.float()
        elif decoder.reconstruction_mode == "hybrid_magnitude_residual":
            compressed = noisy_magnitude.float().clamp_min(1e-7).pow(decoder.magnitude_power)
            log_gain = decoder.magnitude_log_gain_bound * torch.tanh(raw[:, 0])
            ratio = compressed * torch.exp(decoder.magnitude_power * log_gain)
            reference = compressed.mean(dim=-2, keepdim=True).clamp_min(1e-5)
            residual = decoder.magnitude_residual_bound * reference * torch.tanh(raw[:, 1])
            magnitude = (ratio + residual).clamp_min(1e-7).pow(
                1.0 / decoder.magnitude_power
            )
            phase_residual = torch.atan2(
                decoder.mask_bound * torch.tanh(raw[:, 3]),
                1.0 + decoder.mask_bound * torch.tanh(raw[:, 2]),
            )
            candidate_phase = noisy_phase.float() + phase_residual
        else:
            bound = decoder.mask_bound
            speech_raw = torch.complex(
                1.0 + bound * torch.tanh(raw[:, 0]),
                bound * torch.tanh(raw[:, 1]),
            )
            noise_raw = torch.complex(
                bound * torch.tanh(raw[:, 2]),
                bound * torch.tanh(raw[:, 3]),
            )
            share = torch.sigmoid(raw[:, 4])
            speech_mask = speech_raw + share * (1.0 - speech_raw - noise_raw)
            noisy_spectrum = torch.polar(noisy_magnitude.float(), noisy_phase.float())
            candidate = speech_mask * noisy_spectrum
            magnitude = candidate.abs()
            candidate_phase = torch.angle(candidate)

        phase_residual = torch.atan2(
            torch.sin(candidate_phase - noisy_phase),
            torch.cos(candidate_phase - noisy_phase),
        )
        predicted_phase = noisy_phase + self.model.phase_residual_scale * phase_residual
        return torch.polar(magnitude, predicted_phase).squeeze(-1)

    def _enhance_frame(
        self, audio_frame: torch.Tensor, state: V8NativeStreamState
    ) -> torch.Tensor:
        model = self.model
        window = model.analysis_window.to(device=audio_frame.device, dtype=audio_frame.dtype)
        spectrum = torch.fft.rfft(
            audio_frame.squeeze(1) * window, n=model.n_fft, dim=-1
        )
        magnitude = spectrum.abs().clamp_min(1e-7).unsqueeze(-1)
        noisy_phase = torch.angle(spectrum).unsqueeze(-1)
        inputs = torch.stack(
            (
                magnitude.squeeze(1).pow(model.magnitude_power),
                torch.cos(noisy_phase.squeeze(1)),
                torch.sin(noisy_phase.squeeze(1)),
            ),
            dim=1,
        )

        detail = self._sequence(model.encoder.detail, inputs, state, "encoder.detail")
        encoded = self._sequence(model.encoder.down, detail, state, "encoder.down")
        noise = self._rolling_noise(encoded, state)
        local = self._sequence(
            model.bottleneck.local, encoded, state, "bottleneck.local"
        )
        current = encoded + local
        condition = model.bottleneck.noise_condition(noise).transpose(1, 2)
        scale, shift = condition.chunk(2, dim=1)
        current = current * (1.0 + 0.05 * torch.tanh(scale[:, :, None]))
        current = current + 0.05 * shift[:, :, None]
        temporal = self._temporal(current, state)
        latent = current + torch.tanh(model.bottleneck.temporal_scale) * temporal
        return self._decode(latent, detail, magnitude, noisy_phase, state)

    def _synthesise_frame(
        self, spectrum: torch.Tensor, state: V8NativeStreamState
    ) -> torch.Tensor:
        model = self.model
        window = model.analysis_window.to(device=spectrum.device, dtype=torch.float32)
        frame = torch.fft.irfft(spectrum, n=model.n_fft, dim=-1)[..., : model.win_length]
        state.ola_numerator.add_(frame.float() * window)
        state.ola_denominator.add_(window.square().unsqueeze(0))
        weight = state.ola_denominator[..., : model.hop_length]
        output = torch.where(
            weight >= 5e-3,
            state.ola_numerator[..., : model.hop_length] / weight.clamp_min(5e-3),
            torch.zeros_like(weight),
        )
        state.ola_numerator = torch.cat(
            (
                state.ola_numerator[..., model.hop_length :],
                state.ola_numerator.new_zeros(
                    state.ola_numerator.shape[0], model.hop_length
                ),
            ),
            dim=-1,
        )
        state.ola_denominator = torch.cat(
            (
                state.ola_denominator[..., model.hop_length :],
                state.ola_denominator.new_zeros(
                    state.ola_denominator.shape[0], model.hop_length
                ),
            ),
            dim=-1,
        )
        return output.unsqueeze(1)

    def _process_frame(
        self, audio_frame: torch.Tensor, state: V8NativeStreamState
    ) -> torch.Tensor:
        spectrum = self._enhance_frame(audio_frame, state)
        return self._synthesise_frame(spectrum, state)

    @torch.inference_mode()
    def process_chunk(
        self, audio_chunk: torch.Tensor, state: V8NativeStreamState
    ) -> tuple[torch.Tensor, V8NativeStreamState]:
        if state.finished:
            raise RuntimeError("Cannot process audio after flush")
        if audio_chunk.ndim != 3 or audio_chunk.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, T], got {tuple(audio_chunk.shape)}")
        if audio_chunk.shape[:2] != state.input_buffer.shape[:2]:
            raise ValueError("Chunk batch/channel dimensions do not match stream state")

        state.input_samples += audio_chunk.shape[-1]
        state.input_buffer = torch.cat((state.input_buffer, audio_chunk), dim=-1)
        outputs = []
        while state.input_buffer.shape[-1] >= self.model.win_length:
            frame = state.input_buffer[..., : self.model.win_length]
            outputs.append(self._process_frame(frame, state))
            state.input_buffer = state.input_buffer[..., self.model.hop_length :]
        if not outputs:
            return self._empty_like(audio_chunk), state
        output = torch.cat(outputs, dim=-1).to(audio_chunk.dtype)
        state.emitted_samples += output.shape[-1]
        return output, state

    @torch.inference_mode()
    def flush(
        self, state: V8NativeStreamState
    ) -> tuple[torch.Tensor, V8NativeStreamState]:
        if state.finished:
            empty = state.input_buffer[..., :0]
            return empty, state
        if state.input_samples == 0:
            state.finished = True
            return state.input_buffer, state

        model = self.model
        remaining = state.input_samples - state.emitted_samples
        needs_padded_frame = (
            state.input_samples < model.win_length
            or (state.input_samples - model.win_length) % model.hop_length != 0
        )
        outputs = []
        if needs_padded_frame:
            padding = model.win_length - state.input_buffer.shape[-1]
            frame = F.pad(state.input_buffer, (0, padding))
            outputs.append(self._process_frame(frame, state))

        tail = torch.where(
            state.ola_denominator >= 5e-3,
            state.ola_numerator / state.ola_denominator.clamp_min(5e-3),
            torch.zeros_like(state.ola_denominator),
        )
        outputs.append(tail.unsqueeze(1))
        output = torch.cat(outputs, dim=-1)[..., :remaining]
        state.emitted_samples += output.shape[-1]
        state.finished = True
        state.input_buffer = state.input_buffer[..., :0]
        return output.to(state.ola_numerator.dtype), state


__all__ = ["V8NativeStreamer", "V8NativeStreamState"]
