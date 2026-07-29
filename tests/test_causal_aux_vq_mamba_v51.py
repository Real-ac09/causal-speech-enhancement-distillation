from __future__ import annotations

import pytest
import torch

from cnvqg.models import CausalAuxVQMambaV51


def network(mode: str = "log_ratio") -> CausalAuxVQMambaV51:
    model = CausalAuxVQMambaV51(
        variant="student", magnitude_mode=mode, use_mamba=False
    ).eval()
    # Exercise the feature path; all reconstruction heads intentionally start
    # at identity and would otherwise hide future leakage in an untrained test.
    torch.manual_seed(11)
    model.decoder.magnitude_head.weight.data.normal_(0.0, 0.02)
    model.decoder.phase_head.weight.data.normal_(0.0, 0.02)
    return model


@pytest.mark.parametrize(
    "mode", ["bounded_mask", "log_ratio", "compressed_residual"]
)
def test_magnitude_modes_start_finite_and_shape_safe(mode: str) -> None:
    model = network(mode)
    output = model(torch.randn(1, 1, 961))
    assert output.enhanced.shape == (1, 1, 961)
    assert torch.isfinite(output.enhanced).all()
    assert torch.isfinite(output.magnitude_mask).all()


def test_future_suffix_cannot_change_finalized_prefix() -> None:
    model = network()
    torch.manual_seed(12)
    original = torch.randn(1, 1, 1920)
    changed = original.clone()
    boundary = 1280
    changed[..., boundary:] = 3.0 * torch.randn_like(changed[..., boundary:])
    with torch.no_grad():
        first = model(original).enhanced
        second = model(changed).enhanced
    finalized = boundary - model.algorithmic_latency_samples
    torch.testing.assert_close(
        first[..., :finalized], second[..., :finalized], atol=1e-6, rtol=1e-6
    )


def test_streaming_matches_whole_utterance() -> None:
    model = network("compressed_residual")
    waveform = torch.randn(1, 1, 1753)
    whole = model(waveform).enhanced
    state = model.init_stream_state(1, "cpu", torch.float32)
    outputs = []
    offset = 0
    for size in (73, 247, 1, 640, 511, 281):
        current, state = model.forward_chunk(waveform[..., offset : offset + size], state)
        outputs.append(current)
        offset += size
    tail, _ = model.flush(state)
    outputs.append(tail)
    torch.testing.assert_close(
        torch.cat(outputs, dim=-1), whole, atol=1e-4, rtol=1e-4
    )


def test_parameter_caps() -> None:
    assert CausalAuxVQMambaV51(variant="student", use_mamba=False).parameter_count() <= 1_100_000
    assert CausalAuxVQMambaV51(variant="teacher", use_mamba=False).parameter_count() <= 2_700_000


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 requires CUDA")
def test_bf16_gradients_are_finite() -> None:
    model = CausalAuxVQMambaV51(variant="teacher", use_mamba=True).cuda().train()
    waveform = torch.randn(1, 1, 960, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(waveform)
        loss = output.enhanced.square().mean() + output.vq.loss
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
