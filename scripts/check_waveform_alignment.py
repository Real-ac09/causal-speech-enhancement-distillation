#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from cnvqg.models.streaming_hybrid_v2 import CausalTransposeDecoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Check decoder latent/sample alignment.")
    parser.add_argument("--latent-index", type=int, default=5)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=16)
    args = parser.parse_args()
    torch.manual_seed(0)
    decoder = CausalTransposeDecoder(args.latent_dim, args.base_channels).eval()
    for layer in decoder.layers:
        torch.nn.init.zeros_(layer.bias)
    latent = torch.zeros(1, args.latent_dim, args.latent_index + 8)
    latent[..., args.latent_index] = 1.0
    with torch.no_grad():
        output = decoder(latent)[0, 0]
    active = torch.nonzero(output.abs() > 1e-8).flatten()
    expected = args.latent_index * 16
    observed = int(active.min()) if len(active) else None
    print("expected first affected sample:", expected)
    print("observed first affected sample:", observed)
    print("aligned:", observed == expected)
    if observed != expected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
