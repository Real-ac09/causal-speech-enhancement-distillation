from __future__ import annotations

import torch
import torch.nn.functional as F

from .causal_oracle_residual_gate_v16 import CausalOracleResidualGateV16
from .v14_native_streaming import V14NativeStreamer
from .v8_native_streaming import V8NativeStreamer, V8NativeStreamState


class V16NativeStreamer(V14NativeStreamer):
    """Constant-work streamer for the V16 true waveform residual path."""

    LAST_STRENGTH_KEY = "v16_last_strength"

    def __init__(self, model: CausalOracleResidualGateV16) -> None:
        if type(model) is not CausalOracleResidualGateV16:
            raise TypeError(
                "V16NativeStreamer supports "
                "CausalOracleResidualGateV16 exactly"
            )
        if model.training:
            raise ValueError("V16NativeStreamer requires model.eval()")
        self.gated_model = model
        V8NativeStreamer.__init__(self, model.backbone)

    def _process_frame(
        self,
        audio_frame: torch.Tensor,
        state: V8NativeStreamState,
    ) -> torch.Tensor:
        base_spectrum, _, strength = self._base_spectrum_and_strength(
            audio_frame,
            state,
        )
        state.gru[self.LAST_STRENGTH_KEY] = strength
        base_output = self._synthesise_frame(base_spectrum, state)
        noisy_output = audio_frame[..., : self.model.hop_length]
        sample_strength = strength.unsqueeze(-1)
        return noisy_output + sample_strength * (
            base_output.to(noisy_output.dtype) - noisy_output
        )

    @torch.inference_mode()
    def flush(
        self,
        state: V8NativeStreamState,
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
            or (state.input_samples - model.win_length)
            % model.hop_length
            != 0
        )
        raw_buffer = state.input_buffer
        outputs = []
        if needs_padded_frame:
            padding = model.win_length - raw_buffer.shape[-1]
            frame = F.pad(raw_buffer, (0, padding))
            outputs.append(self._process_frame(frame, state))
            raw_tail = raw_buffer[..., model.hop_length :]
        else:
            raw_tail = raw_buffer

        base_tail = torch.where(
            state.ola_denominator >= 5e-3,
            state.ola_numerator
            / state.ola_denominator.clamp_min(5e-3),
            torch.zeros_like(state.ola_denominator),
        ).unsqueeze(1)
        raw_tail = F.pad(
            raw_tail,
            (0, max(0, base_tail.shape[-1] - raw_tail.shape[-1])),
        )[..., : base_tail.shape[-1]]
        strength = state.gru.get(self.LAST_STRENGTH_KEY)
        if strength is None:
            raise RuntimeError("V16 stream has no final gate strength")
        tail = raw_tail + strength.unsqueeze(-1) * (
            base_tail.to(raw_tail.dtype) - raw_tail
        )
        outputs.append(tail)
        output = torch.cat(outputs, dim=-1)[..., :remaining]
        state.emitted_samples += output.shape[-1]
        state.finished = True
        state.input_buffer = state.input_buffer[..., :0]
        return output.to(state.ola_numerator.dtype), state


__all__ = ["V16NativeStreamer"]
