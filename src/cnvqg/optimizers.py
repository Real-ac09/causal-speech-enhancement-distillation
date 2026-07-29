from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


class HybridMuonAdamW(torch.optim.Optimizer):
    """Muon for hidden matrices and AdamW for all auxiliary parameters."""

    def __init__(self, muon, adamw) -> None:
        self.muon = muon
        self.adamw = adamw
        parameters = [parameter for group in muon.param_groups + adamw.param_groups for parameter in group["params"]]
        super().__init__(parameters, defaults={})
        # Keep the children's dictionaries so schedulers update the real LRs.
        self.param_groups = muon.param_groups + adamw.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.muon.step(closure)
        self.adamw.step()
        return loss

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
        self.param_groups = self.muon.param_groups + self.adamw.param_groups


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model when using DataParallel or DDP."""
    return model.module if hasattr(model, "module") else model


def build_optimizer(
    model: nn.Module,
    training_config: Mapping[str, Any],
) -> torch.optim.Optimizer:
    """
    Build AdamW with either:

    1. One learning rate for every trainable parameter; or
    2. Separate learning rates for the waveform backbone and TF refiner.

    Differential learning rates are enabled when both
    `backbone_learning_rate` and `refiner_learning_rate`
    are present in the training configuration.
    """

    unwrapped_model = _unwrap_model(model)

    optimizer_config = training_config.get("optimizer", {})
    optimizer_name = str(optimizer_config.get("name", "adamw")).lower()

    default_lr = float(
        training_config.get("learning_rate", 1e-4)
    )
    weight_decay = float(
        training_config.get("weight_decay", 1e-5)
    )

    backbone_lr = training_config.get(
        "backbone_learning_rate"
    )
    refiner_lr = training_config.get(
        "refiner_learning_rate"
    )

    trainable_named_parameters = [
        (name, parameter)
        for name, parameter in unwrapped_model.named_parameters()
        if parameter.requires_grad
    ]

    if not trainable_named_parameters:
        raise RuntimeError(
            "The model has no trainable parameters."
        )

    if optimizer_name == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("This PyTorch build does not provide torch.optim.Muon")
        excluded_fragments = (
            "norm",
            "bias",
            "layer_scale",
            "residual_scale",
            "tf_output",
            "noise_predictor",
            # Mamba/SSM dynamics, scalar gates, VQ state, and quality models
            # retain AdamW as specified by the V5 optimisation programme.
            "mamba",
            "frequency_gru",
            "time_scale",
            "frequency_scale",
            "cross_scale",
            "strength_logit",
            "mask_scale_logit",
            "noise_vq",
            "codebook",
            "quality_discriminator",
        )
        muon_parameters = [
            parameter
            for name, parameter in trainable_named_parameters
            if parameter.ndim == 2
            and not any(fragment in name.lower() for fragment in excluded_fragments)
        ]
        muon_ids = {id(parameter) for parameter in muon_parameters}
        adamw_parameters = [
            parameter for _, parameter in trainable_named_parameters
            if id(parameter) not in muon_ids
        ]
        if not muon_parameters or not adamw_parameters:
            raise RuntimeError("Muon routing produced an empty parameter group")
        muon = torch.optim.Muon(
            [{"params": muon_parameters, "name": "muon_hidden"}],
            lr=float(optimizer_config.get("muon_learning_rate", 0.01)),
            momentum=float(optimizer_config.get("momentum", 0.95)),
            nesterov=bool(optimizer_config.get("nesterov", True)),
            ns_steps=int(optimizer_config.get("ns_steps", 5)),
            weight_decay=float(optimizer_config.get("muon_weight_decay", weight_decay)),
        )
        adamw = torch.optim.AdamW(
            [{"params": adamw_parameters, "name": "adamw_auxiliary"}],
            lr=float(optimizer_config.get("adamw_learning_rate", default_lr)),
            weight_decay=weight_decay,
        )
        print(
            "Optimizer: Muon/AdamW; "
            f"Muon={sum(p.numel() for p in muon_parameters):,}, "
            f"AdamW={sum(p.numel() for p in adamw_parameters):,} parameters"
        )
        return HybridMuonAdamW(muon, adamw)

    if optimizer_name != "adamw":
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Preserve the original single-LR behaviour for all existing configs.
    if backbone_lr is None or refiner_lr is None:
        parameters = [
            parameter
            for _, parameter in trainable_named_parameters
        ]

        print("Optimizer: AdamW with one parameter group")
        print(
            f"  Trainable parameters: "
            f"{sum(p.numel() for p in parameters):,}"
        )
        print(f"  Learning rate: {default_lr:.2e}")

        return torch.optim.AdamW(
            parameters,
            lr=default_lr,
            weight_decay=weight_decay,
        )

    backbone_lr = float(backbone_lr)
    refiner_lr = float(refiner_lr)

    # Integrated v2/v3 hybrids expose the TF refiner as several top-level
    # modules rather than a single waveform wrapper.  Classify those modules
    # explicitly so the encoder, waveform Mamba stack, VQ and decoder all keep
    # the conservative backbone learning rate.
    integrated_refiner_prefixes = (
        "tf_input",
        "tf_blocks",
        "tf_temporal",
        "cross_band",
        "tf_output",
        "noise_tf_projection",
    )
    is_integrated_hybrid = hasattr(unwrapped_model, "tf_input") and hasattr(
        unwrapped_model, "waveform_decoder"
    )
    if is_integrated_hybrid:
        refiner_parameters = [
            parameter
            for name, parameter in trainable_named_parameters
            if any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in integrated_refiner_prefixes
            )
        ]
        refiner_ids = {id(parameter) for parameter in refiner_parameters}
        backbone_parameters = [
            parameter
            for _, parameter in trainable_named_parameters
            if id(parameter) not in refiner_ids
        ]
        backbone_prefix = "integrated_waveform"
    else:
        backbone_parameters = []
        refiner_parameters = []
        backbone_prefix = None

    # Locate wrapped waveform backbones through their parameter-name prefix.
    candidate_prefixes = (
        "waveform_branch",
        "waveform_model",
        "waveform",
        "backbone",
        "base_model",
    )

    if not is_integrated_hybrid:
        for candidate in candidate_prefixes:
            if any(
                name == candidate
                or name.startswith(f"{candidate}.")
                for name, _ in trainable_named_parameters
            ):
                backbone_prefix = candidate
                break

    if backbone_prefix is None:
        top_level_modules = [
            name
            for name, _ in unwrapped_model.named_children()
        ]

        sample_names = [
            name
            for name, _ in trainable_named_parameters[:20]
        ]

        raise RuntimeError(
            "Differential learning rates were requested, but the "
            "waveform backbone could not be identified.\n"
            f"Top-level modules: {top_level_modules}\n"
            f"Example parameter names: {sample_names}"
        )

    if not is_integrated_hybrid:
        backbone_parameters = [
            parameter
            for name, parameter in trainable_named_parameters
            if name == backbone_prefix
            or name.startswith(f"{backbone_prefix}.")
        ]

        refiner_parameters = [
            parameter
            for name, parameter in trainable_named_parameters
            if not (
                name == backbone_prefix
                or name.startswith(f"{backbone_prefix}.")
            )
        ]

    if not backbone_parameters:
        raise RuntimeError(
            f"Backbone '{backbone_prefix}' has no trainable parameters. "
            "Set model.freeze_waveform to false."
        )

    if not refiner_parameters:
        raise RuntimeError(
            "No trainable refiner parameters were found."
        )

    backbone_count = sum(
        parameter.numel()
        for parameter in backbone_parameters
    )
    refiner_count = sum(
        parameter.numel()
        for parameter in refiner_parameters
    )

    print("Optimizer: AdamW with differential learning rates")
    print(
        f"  Backbone ({backbone_prefix}): "
        f"{backbone_count:,} parameters, "
        f"LR={backbone_lr:.2e}"
    )
    print(
        f"  TF refiner: "
        f"{refiner_count:,} parameters, "
        f"LR={refiner_lr:.2e}"
    )
    print(
        f"  Total trainable: "
        f"{backbone_count + refiner_count:,}"
    )

    parameter_groups = [
        {
            "params": backbone_parameters,
            "lr": backbone_lr,
            "name": "backbone",
        },
        {
            "params": refiner_parameters,
            "lr": refiner_lr,
            "name": "refiner",
        },
    ]

    return torch.optim.AdamW(
        parameter_groups,
        lr=refiner_lr,
        weight_decay=weight_decay,
    )


def optimizer_lr_summary(
    optimizer: torch.optim.Optimizer,
) -> dict[str, str]:
    """Return readable learning rates for every optimizer group."""

    return {
        group.get("name", f"group_{index}"): f"{group['lr']:.2e}"
        for index, group in enumerate(optimizer.param_groups)
    }
