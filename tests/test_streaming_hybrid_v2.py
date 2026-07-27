from __future__ import annotations

import torch

from cnvqg.models.factory import build_model
from cnvqg.models.streaming_hybrid_v2 import (
    CausalConv1d,
    CausalConv2d,
    CausalTransposeDecoder,
    EMANoiseVectorQuantizer,
    StreamingHybridCNVQGV2,
)


def test_causal_conv1d_prefix_is_independent_of_future() -> None:
    torch.manual_seed(0)
    layer = CausalConv1d(2, 3, kernel_size=5)
    prefix = torch.randn(1, 2, 32)
    longer = torch.cat((prefix, torch.randn(1, 2, 16)), dim=-1)
    torch.testing.assert_close(layer(prefix), layer(longer)[..., :32])


def test_causal_conv2d_prefix_is_independent_of_future() -> None:
    torch.manual_seed(0)
    layer = CausalConv2d(2, 3, kernel_size=(3, 5))
    prefix = torch.randn(1, 2, 17, 24)
    longer = torch.cat((prefix, torch.randn(1, 2, 17, 8)), dim=-1)
    torch.testing.assert_close(layer(prefix), layer(longer)[..., :24])


def test_ema_vq_preserves_length_and_updates_codebook() -> None:
    torch.manual_seed(0)
    quantizer = EMANoiseVectorQuantizer(
        codebook_size=8,
        code_dim=4,
        update_interval=2,
        dead_code_threshold=0.0,
    ).train()
    before = quantizer.codebook.clone()
    output = quantizer(torch.randn(2, 11, 4))
    assert output.quantized.shape == (2, 11, 4)
    assert output.indices.shape == (2, 11)
    assert torch.isfinite(output.loss)
    assert not torch.equal(before, quantizer.codebook)


def test_factory_builds_all_v2_variants() -> None:
    expected_order = []
    for variant in ("tiny", "student", "teacher"):
        model = build_model(
            {
                "architecture": "streaming_hybrid_v2",
                "variant": variant,
                "use_mamba": False,
            }
        )
        expected_order.append(sum(parameter.numel() for parameter in model.parameters()))
        assert model.variant == variant
        assert model.temporal.uses_mamba is False
    assert expected_order[0] < expected_order[1] < expected_order[2]


def test_tiny_forward_output_and_identity_tf_initialization() -> None:
    torch.manual_seed(0)
    model = StreamingHybridCNVQGV2(variant="tiny", use_mamba=False).eval()
    noisy = torch.randn(1, 1, 4096)
    with torch.no_grad():
        output = model(noisy)
    assert output.enhanced.shape == noisy.shape
    assert output.base_enhanced.shape == noisy.shape
    assert output.waveform_residual.shape == noisy.shape
    assert len(output.speech_features) == 3
    torch.testing.assert_close(output.gain, torch.ones_like(output.gain))
    torch.testing.assert_close(output.phase_delta, torch.zeros_like(output.phase_delta))
    torch.testing.assert_close(output.enhanced, output.base_enhanced, atol=2e-5, rtol=2e-5)


def test_tiny_model_has_stable_finalized_prefix() -> None:
    torch.manual_seed(0)
    model = StreamingHybridCNVQGV2(variant="tiny", use_mamba=False).eval()
    prefix = torch.randn(1, 1, 4096)
    longer = torch.cat((prefix, torch.randn(1, 1, 2048)), dim=-1)
    with torch.no_grad():
        short_output = model(prefix).enhanced
        long_output = model(longer).enhanced
    # The last STFT window is not finalized until future input arrives.
    finalized = prefix.shape[-1] - model.stft.win_length
    torch.testing.assert_close(
        short_output[..., :finalized],
        long_output[..., :finalized],
        # FFT kernels may choose a different batched plan for a different
        # number of frames, producing small floating-point-only differences.
        atol=2e-4,
        rtol=2e-4,
    )


def test_reference_streaming_matches_direct_forward() -> None:
    torch.manual_seed(0)
    model = StreamingHybridCNVQGV2(variant="tiny", use_mamba=False).eval()
    waveform = torch.randn(1, 1, 2048)
    with torch.no_grad():
        direct = model(waveform).enhanced
        streamed = model.enhance_offline(waveform, chunk_samples=512)
    assert streamed.shape == direct.shape
    torch.testing.assert_close(streamed, direct, atol=2e-4, rtol=2e-4)


def test_causal_transpose_decoder_impulse_starts_at_aligned_sample() -> None:
    torch.manual_seed(0)
    decoder = CausalTransposeDecoder(in_dim=4, base_channels=8).eval()
    for layer in decoder.layers:
        torch.nn.init.zeros_(layer.bias)
    latent = torch.zeros(1, 4, 12)
    latent[..., 5] = 1.0
    output = decoder(latent)
    nonzero = torch.nonzero(output[0, 0].abs() > 1e-8).flatten()
    assert len(nonzero) > 0
    assert int(nonzero.min()) == 5 * 16


def test_waveform_only_transpose_model_prefix_is_causal() -> None:
    torch.manual_seed(0)
    model = StreamingHybridCNVQGV2(
        variant="tiny",
        use_mamba=False,
        decoder_type="causal_transpose",
        enable_tf_refiner=False,
    ).eval()
    prefix = torch.randn(1, 1, 2048)
    longer = torch.cat((prefix, torch.randn(1, 1, 512)), dim=-1)
    with torch.no_grad():
        prefix_output = model(prefix).enhanced
        longer_output = model(longer).enhanced[..., :prefix.shape[-1]]
    torch.testing.assert_close(prefix_output, longer_output, atol=1e-6, rtol=1e-6)


def test_waveform_only_mode_does_not_backpropagate_into_tf_refiner() -> None:
    model = StreamingHybridCNVQGV2(
        variant="tiny",
        use_mamba=False,
        decoder_type="causal_transpose",
        enable_tf_refiner=False,
    ).train()
    output = model(torch.randn(1, 1, 1024))
    output.enhanced.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.waveform_decoder.parameters())
    assert all(parameter.grad is None for parameter in model.tf_input.parameters())
