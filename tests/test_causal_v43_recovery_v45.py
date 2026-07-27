from __future__ import annotations

import torch

from cnvqg.models import CausalV43RecoveryV45
from cnvqg.models.factory import build_model


def test_v45_factory_uses_minimal_defaults() -> None:
    model = build_model(
        {
            "architecture": "causal_v43_recovery_v45",
            "variant": "student",
            "channels": 64,
            "noise_dim": 32,
            "use_mamba": False,
        }
    )
    assert isinstance(model, CausalV43RecoveryV45)
    assert model.refinement_passes == 1
    assert model.vq_mode == "disabled"
    assert model.magnitude_mode == "bounded_mask"


def test_v45_vq_disabled_is_bitwise_codebook_independent() -> None:
    model = CausalV43RecoveryV45(
        variant="student",
        channels=64,
        noise_dim=32,
        use_mamba=False,
    ).eval()
    noisy = torch.randn(1, 1, 961)
    with torch.inference_mode():
        first = model(noisy)
        model.noise_vq.codebook.copy_(torch.randn_like(model.noise_vq.codebook) * 100.0)
        second = model(noisy)
    assert torch.equal(first.enhanced, second.enhanced)
    assert (first.code_indices == -1).all()
    assert first.code_perplexity.item() == 0.0


def test_v45_future_independence_and_streaming_equivalence() -> None:
    model = CausalV43RecoveryV45(
        variant="student",
        channels=64,
        noise_dim=32,
        use_mamba=False,
    ).eval()
    torch.manual_seed(45)
    model.decoder.magnitude_head.weight.data.normal_(0.0, 0.02)
    original = torch.randn(1, 1, 1753)
    changed = original.clone()
    changed[..., 1280:] = torch.randn_like(changed[..., 1280:])
    with torch.inference_mode():
        first = model(original).enhanced
        second = model(changed).enhanced
    torch.testing.assert_close(first[..., :960], second[..., :960], atol=1e-6, rtol=1e-6)

    state = model.init_stream_state(1, "cpu", torch.float32)
    pieces = []
    offset = 0
    for size in (73, 247, 1, 640, 511, 281):
        piece, state = model.forward_chunk(original[..., offset : offset + size], state)
        pieces.append(piece)
        offset += size
    tail, _ = model.flush(state)
    pieces.append(tail)
    torch.testing.assert_close(torch.cat(pieces, -1), first, atol=1e-4, rtol=1e-4)
