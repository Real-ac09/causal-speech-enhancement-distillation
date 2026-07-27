from __future__ import annotations

import torch

from cnvqg.models.auxiliary_gated_tf_mamba_v43 import AuxiliaryGatedTFMambaV43
from cnvqg.models.factory import build_model


def test_v43_factory_defaults_to_preservation_first_gate() -> None:
    model = build_model(
        {
            "architecture": "auxiliary_gated_tf_mamba_v43",
            "variant": "tiny",
            "use_mamba": False,
        }
    )
    assert isinstance(model, AuxiliaryGatedTFMambaV43)
    assert model.use_noise_codebook
    assert model.vq_gate_logits is not None
    assert not model.noise_vq.update_codebook
    torch.testing.assert_close(
        torch.sigmoid(model.vq_gate_logits),
        torch.full((model.noise_dim,), 0.01),
    )


def test_v43_gate_receives_gradient_with_frozen_enhancer() -> None:
    model = AuxiliaryGatedTFMambaV43(
        variant="tiny",
        use_mamba=False,
        max_iterations=1,
        noise_segment_frames=8,
    ).train()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name == "vq_gate_logits")
    noisy = torch.randn(1, 1, 4096)
    output = model(noisy)
    output.enhanced.square().mean().backward()
    assert output.code_indices.shape[1] > 1
    assert output.vq_gate.shape == (model.noise_dim,)
    assert model.vq_gate_logits.grad is not None
    assert torch.isfinite(model.vq_gate_logits.grad).all()
