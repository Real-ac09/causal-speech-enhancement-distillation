from __future__ import annotations

import copy

import pytest
import torch

from cnvqg.models import CausalAuxVQMambaV5


def model(variant: str = "student") -> CausalAuxVQMambaV5:
    return CausalAuxVQMambaV5(variant=variant, use_mamba=False).eval()


def test_parameter_caps_and_fixed_passes() -> None:
    student = model("student")
    teacher = model("teacher")
    assert student.parameter_count() <= 1_100_000
    assert teacher.parameter_count() <= 2_700_000
    assert student.refinement_passes == 2
    assert teacher.refinement_passes == 3
    assert student.algorithmic_latency_samples == 320


@pytest.mark.parametrize("chunks", [[1, 79, 160, 317, 1043], [320, 320, 960]])
def test_arbitrary_chunks_equal_whole_file(chunks: list[int]) -> None:
    torch.manual_seed(1)
    network = model()
    waveform = torch.randn(1, 1, sum(chunks))
    whole = network(waveform).enhanced
    state = network.init_stream_state(1, waveform.device, waveform.dtype)
    outputs = []
    offset = 0
    for size in chunks:
        chunk, state = network.forward_chunk(waveform[..., offset : offset + size], state)
        outputs.append(chunk)
        offset += size
    tail, state = network.flush(state)
    outputs.append(tail)
    streamed = torch.cat(outputs, dim=-1)
    torch.testing.assert_close(streamed, whole, atol=1e-4, rtol=1e-4)


def test_future_frames_cannot_change_finalized_output() -> None:
    torch.manual_seed(2)
    network = model()
    prefix = torch.randn(1, 1, 800)
    state_a = network.init_stream_state(1, "cpu", torch.float32)
    emitted_a, _ = network.forward_chunk(prefix, state_a)
    state_b = network.init_stream_state(1, "cpu", torch.float32)
    emitted_b, state_b = network.forward_chunk(prefix, state_b)
    _, _ = network.forward_chunk(torch.randn(1, 1, 600), state_b)
    assert torch.equal(emitted_a, emitted_b)


def test_state_reset_prevents_cross_utterance_leakage() -> None:
    network = model()
    first = torch.randn(1, 1, 731)
    second = torch.randn(1, 1, 913)
    state = network.init_stream_state(1, "cpu", torch.float32)
    _, state = network.forward_chunk(first, state)
    _, state = network.flush(state)
    reset = network.init_stream_state(1, "cpu", torch.float32)
    output, reset = network.forward_chunk(second, reset)
    tail, reset = network.flush(reset)
    torch.testing.assert_close(
        torch.cat((output, tail), -1), network(second).enhanced, atol=1e-4, rtol=1e-4
    )


def test_train_only_enhancement_is_codebook_independent() -> None:
    torch.manual_seed(3)
    network = model()
    clone = copy.deepcopy(network)
    clone.noise_vq.codebook.normal_(mean=100.0, std=20.0)
    waveform = torch.randn(1, 1, 960)
    assert torch.equal(network(waveform).enhanced, clone(waveform).enhanced)


def test_bounded_adapter_starts_below_five_percent() -> None:
    network = CausalAuxVQMambaV5(
        variant="student", use_mamba=False, vq_mode="bounded_adapter"
    ).eval()
    output = network(torch.randn(1, 1, 640))
    assert 0.0 <= output.vq_adapter_strength.item() <= 0.05


def test_phase_representation_is_finite_at_wrap() -> None:
    network = model()
    output = network(torch.randn(1, 1, 640))
    wrapped = torch.atan2(torch.sin(output.predicted_phase), torch.cos(output.predicted_phase))
    assert torch.isfinite(wrapped).all()
    assert wrapped.min() >= -torch.pi and wrapped.max() <= torch.pi


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 test requires CUDA")
def test_bf16_forward_backward_is_finite() -> None:
    network = CausalAuxVQMambaV5(variant="student", use_mamba=True).cuda().train()
    waveform = torch.randn(1, 1, 960, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = network(waveform)
        loss = output.enhanced.square().mean() + output.vq.loss
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in network.parameters()
    )
