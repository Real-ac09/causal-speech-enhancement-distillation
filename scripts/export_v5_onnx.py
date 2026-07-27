#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import math

import torch
import numpy as np

from cnvqg.models.factory import build_model


class V5WholeFileExport(torch.nn.Module):
    """ONNX-friendly deployment surface; state is the explicit audio history."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        frequencies = torch.arange(model.n_fft // 2 + 1, dtype=torch.float64)[:, None]
        samples = torch.arange(model.n_fft, dtype=torch.float64)[None, :]
        angle = 2.0 * math.pi * frequencies * samples / model.n_fft
        self.register_buffer("dft_cos", torch.cos(angle))
        self.register_buffer("dft_sin", torch.sin(angle))

    def _analysis(self, waveform: torch.Tensor):
        frames = waveform.squeeze(1).unfold(-1, self.model.win_length, self.model.hop_length)
        frames = (frames * self.model.analysis_window).double()
        frames = torch.nn.functional.pad(frames, (0, self.model.n_fft - self.model.win_length))
        real = frames @ self.dft_cos.transpose(0, 1)
        imag = -(frames @ self.dft_sin.transpose(0, 1))
        return real.transpose(1, 2).float(), imag.transpose(1, 2).float()

    def _synthesis(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        # Explicit real-valued inverse DFT avoids unsupported ONNX complex FFT ops.
        real, imag = real.double(), imag.double()
        interior_real = real[:, 1:-1].transpose(1, 2)
        interior_imag = imag[:, 1:-1].transpose(1, 2)
        samples = (
            real[:, :1].transpose(1, 2)
            + real[:, -1:].transpose(1, 2) * self.dft_cos[-1]
            + 2.0 * (interior_real @ self.dft_cos[1:-1]
                     - interior_imag @ self.dft_sin[1:-1])
        ) / self.model.n_fft
        frames = samples[..., : self.model.win_length] * self.model.analysis_window
        weight = self.model.analysis_window.square().clamp_min(1e-7)
        first = frames[:, 0, :160] / weight[:160]
        middle = (frames[:, 0, 160:] + frames[:, 1, :160]) / (weight[160:] + weight[:160])
        last = frames[:, 1, 160:] / weight[160:]
        return torch.cat((first, middle, last), dim=-1).unsqueeze(1).float()

    def forward(self, audio_history: torch.Tensor, audio_chunk: torch.Tensor):
        new_history = torch.cat((audio_history, audio_chunk), dim=-1)
        real, imag = self._analysis(new_history)
        magnitude = torch.sqrt(real.square() + imag.square()).clamp_min(1e-7)
        inputs = torch.stack(
            (magnitude.pow(self.model.magnitude_power), real / magnitude, imag / magnitude), dim=1
        )
        features, full_skip = self.model.encoder(inputs)
        noise = self.model._rolling_noise(features)
        vq = self.model.noise_vq(noise)
        current = features
        for _ in range(self.model.refinement_passes):
            current = self.model.cell(current, noise)
        mask, phase_vector = self.model.decoder(current, full_skip)
        noisy_phase = torch.atan2(imag, real)
        predicted_phase = noisy_phase + torch.atan2(phase_vector[:, 0], phase_vector[:, 1])
        estimated_magnitude = magnitude * mask
        enhanced = self._synthesis(
            estimated_magnitude * torch.cos(predicted_phase),
            estimated_magnitude * torch.sin(predicted_phase),
        )
        return (
            enhanced,
            mask,
            predicted_phase,
            noise,
            vq.indices,
            new_history,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export V5 with explicit stream-state tensors.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    wrapper = V5WholeFileExport(model)
    history = torch.zeros(1, 1, 320)
    chunk = torch.zeros(1, 1, 160)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (history, chunk),
        args.output,
        input_names=["audio_history", "audio_chunk"],
        output_names=["enhanced", "magnitude_mask", "predicted_phase", "noise_state",
                      "code_indices", "new_audio_history"],
        opset_version=args.opset,
        dynamo=False,
    )
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX was exported but verification requires requirements-optional.txt "
            "(onnxruntime)."
        ) from exc
    with torch.inference_mode():
        expected = wrapper(history, chunk)
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    actual = session.run(None, {
        "audio_history": history.numpy(), "audio_chunk": chunk.numpy()
    })
    errors = [float(np.max(np.abs(value.detach().numpy() - observed)))
              for value, observed in zip(expected, actual)]
    if max(errors) > 1e-4:
        raise RuntimeError(f"ONNX parity failed: max errors {errors}")
    print(f"ONNX parity max error: {max(errors):.3e}")
    print(f"Exported {args.output}")


if __name__ == "__main__":
    main()
