#!/usr/bin/env python3

from __future__ import annotations

import argparse

import torch

from cnvqg.models import CNVQGModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check CN-VQG model forward pass.")

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--sample-rate", type=int, default=16000)

    parser.add_argument("--no-mamba", action="store_true")
    parser.add_argument("--no-residual", action="store_true")
    parser.add_argument("--no-vq", action="store_true")
    parser.add_argument("--no-noise-conditioning", action="store_true")
    parser.add_argument("--no-temporal", action="store_true")

    return parser.parse_args()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_samples = int(args.seconds * args.sample_rate)

    model = CNVQGModel(
        use_mamba=not args.no_mamba,
        use_residual=not args.no_residual,
        use_vq=not args.no_vq,
        use_noise_conditioning=not args.no_noise_conditioning,
        use_temporal=not args.no_temporal,
    ).to(device)

    x = torch.randn(args.batch_size, 1, num_samples, device=device)

    with torch.no_grad():
        output = model(x)

    print("Device:", device)
    print("Uses Mamba:", model.temporal.uses_mamba)
    print("Use residual:", model.use_residual)
    print("Use VQ:", model.use_vq)
    print("Use noise conditioning:", model.use_noise_conditioning)
    print("Use temporal:", model.use_temporal)
    print("Parameter count:", count_parameters(model))
    print("Input shape:", x.shape)
    print("Enhanced shape:", output.enhanced.shape)
    print("Residual shape:", output.residual.shape)
    print("Speech latent shape:", output.speech_latent.shape)
    print("Noise latent shape:", output.noise_latent.shape)
    print("Quantized noise shape:", output.quantized_noise.shape)
    print("VQ loss:", float(output.vq.loss.detach().cpu()))
    print("VQ perplexity:", float(output.vq.perplexity.detach().cpu()))

    assert output.enhanced.shape == x.shape
    print("Model forward pass passed.")


if __name__ == "__main__":
    main()
