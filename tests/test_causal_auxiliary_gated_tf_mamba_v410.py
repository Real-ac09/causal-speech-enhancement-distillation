from __future__ import annotations

import copy

import pytest
import torch

from cnvqg.models import AuxiliaryGatedTFMambaV43
from cnvqg.models.causal_auxiliary_gated_tf_mamba_v410 import (
    CausalAuxiliaryGatedTFMambaV410,
)


def model() -> CausalAuxiliaryGatedTFMambaV410:
    return CausalAuxiliaryGatedTFMambaV410(
        variant="tiny",
        use_mamba=False,
        use_noise_codebook=True,
        win_length=320,
        hop_length=160,
        center=False,
    ).eval()


def test_v43_state_dict_is_strictly_compatible() -> None:
    legacy = AuxiliaryGatedTFMambaV43(variant="tiny", use_mamba=False)
    causal = model()
    causal.load_state_dict(legacy.state_dict(), strict=True)
    assert causal.state_dict().keys() == legacy.state_dict().keys()
    assert sum(p.numel() for p in causal.parameters()) == sum(
        p.numel() for p in legacy.parameters()
    )


def test_latency_and_future_independence() -> None:
    torch.manual_seed(1)
    network = model()
    waveform = torch.randn(1, 1, 1920)
    changed = waveform.clone()
    boundary = 1280
    changed[..., boundary:] = torch.randn_like(changed[..., boundary:])
    original_output = network(waveform).enhanced
    changed_output = network(changed).enhanced
    finalized = boundary - network.algorithmic_latency_samples
    assert network.algorithmic_latency_samples == 320
    assert torch.equal(
        original_output[..., :finalized], changed_output[..., :finalized]
    )


@pytest.mark.parametrize("chunks", [[37, 211, 503, 1169], [320, 160, 1440]])
def test_arbitrary_chunks_equal_whole_file(chunks: list[int]) -> None:
    torch.manual_seed(2)
    network = model()
    waveform = torch.randn(1, 1, sum(chunks))
    whole = network(waveform).enhanced
    state = network.init_stream_state(1, waveform.device, waveform.dtype)
    outputs = []
    offset = 0
    for size in chunks:
        output, state = network.forward_chunk(
            waveform[..., offset : offset + size], state
        )
        outputs.append(output)
        offset += size
    tail, state = network.flush(state)
    streamed = torch.cat((*outputs, tail), dim=-1)
    torch.testing.assert_close(streamed, whole, atol=1e-4, rtol=1e-4)


def test_enhancement_is_bitwise_codebook_independent() -> None:
    torch.manual_seed(3)
    network = model()
    changed = copy.deepcopy(network)
    changed.noise_vq.codebook.uniform_(-100.0, 100.0)
    waveform = torch.randn(1, 1, 960)
    assert torch.equal(network(waveform).enhanced, changed(waveform).enhanced)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 test requires CUDA")
def test_bf16_forward_backward_is_finite() -> None:
    network = CausalAuxiliaryGatedTFMambaV410(variant="tiny", use_mamba=True).cuda()
    waveform = torch.randn(1, 1, 960, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = network(waveform)
        loss = output.enhanced.square().mean()
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in network.parameters()
    )
