#!/usr/bin/env python3

from __future__ import annotations

import argparse

import torch

from cnvqg.models import CNVQGModel


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def parse_args():
    parser = argparse.ArgumentParser(description="Check CN-VQG model forward pass.")

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--no-mamba", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_samples = int(args.seconds * args.sample_rate)

    model = CNVQGModel(
        encoder_channels=64,
        latent_dim=128,
        speech_dim=128,
        noise_dim=64,
        codebook_size=256,
        temporal_hidden_dim=128,
        temporal_layers=4,
        use_mamba=not args.no_mamba,
    ).to(device)

    x = torch.randn(args.batch_size, 1, num_samples, device=device)

    with torch.no_grad():
        output = model(x)

    print("Device:", device)
    print("Uses Mamba:", model.temporal.uses_mamba)
    print("Parameter count:", count_parameters(model))
    print("Input shape:", x.shape)
    print("Enhanced shape:", output.enhanced.shape)
    print("Speech latent shape:", output.speech_latent.shape)
    print("Noise latent shape:", output.noise_latent.shape)
    print("Quantized noise shape:", output.quantized_noise.shape)
    print("VQ loss:", float(output.vq.loss.detach().cpu()))
    print("VQ perplexity:", float(output.vq.perplexity.detach().cpu()))

    assert output.enhanced.shape == x.shape

    print("Model forward pass passed.")


if __name__ == "__main__":
    main()
