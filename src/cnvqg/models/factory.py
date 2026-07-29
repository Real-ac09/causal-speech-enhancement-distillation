from __future__ import annotations

from typing import Any, Dict

from torch import nn

from .cnvqg_model import CNVQGModel
from .complex_cnvqg_model import ComplexCNVQGModel
from .hybrid_tf_refiner_model import HybridTFRefinerCNVQGModel
from .streaming_hybrid_v2 import StreamingHybridCNVQGV2
from .streaming_hybrid_v3 import StreamingHybridCNVQGV3
from .noise_adaptive_tf_mamba import NoiseAdaptiveTFMamba
from .noise_adaptive_tf_mamba_v41 import NoiseAdaptiveTFMambaV41
from .continuous_adaptive_tf_mamba_v42 import ContinuousAdaptiveTFMambaV42
from .auxiliary_gated_tf_mamba_v43 import AuxiliaryGatedTFMambaV43
from .causal_aux_vq_mamba_v5 import CausalAuxVQMambaV5
from .causal_aux_vq_mamba_v51 import CausalAuxVQMambaV51
from .causal_aux_vq_mamba_v52 import CausalAuxVQMambaV52
from .causal_complex_mamba_v6 import CausalComplexMambaV6
from .causal_multiscale_mamba_v7 import CausalMultiScaleMambaV7
from .causal_v43_recovery_v45 import CausalV43RecoveryV45
from .predictive_noise_vq_mamba_v8 import PredictiveNoiseVQMambaV8
from .causal_auxiliary_gated_tf_mamba_v410 import CausalAuxiliaryGatedTFMambaV410
from .causal_prototype_dual_axis_mamba_v9 import CausalPrototypeDualAxisMambaV9
from .causal_scale_preserving_mamba_v91 import CausalScalePreservingMambaV91
from .causal_single_scale_mamba_v92 import CausalSingleScaleMambaV92
from .causal_low_rank_frequency_mamba_v93 import CausalLowRankFrequencyMambaV93
from .causal_temporal_detail_mamba_v94 import CausalTemporalDetailMambaV94
from .causal_noise_conditioned_mamba_v95 import CausalNoiseConditionedMambaV95
from .causal_representation_norm_mamba_v96 import CausalRepresentationNormMambaV96
from .causal_global_frequency_mamba_v10 import CausalGlobalFrequencyMambaV10
from .causal_scale_aware_mamba_v11 import CausalScaleAwareMambaV11
from .causal_confidence_gate_v14 import CausalConfidenceGateV14
from .causal_oracle_residual_gate_v16 import CausalOracleResidualGateV16
from .causal_ordinal_residual_gate_v17 import CausalOrdinalResidualGateV17
from .causal_cumulative_ordinal_gate_v17 import (
    CausalCumulativeOrdinalGateV17,
)
from .causal_feature_cumulative_gate_v17 import (
    CausalFeatureCumulativeGateV17,
)
from .causal_utility_safety_gate_v17 import CausalUtilitySafetyGateV17
from .causal_statistics_utility_gate_v17 import (
    CausalStatisticsUtilityGateV17,
)
from .causal_two_stage_utility_gate_v18 import CausalTwoStageUtilityGateV18


def build_model(model_config: Dict[str, Any]) -> nn.Module:
    config = dict(model_config)
    architecture = config.pop("architecture", "waveform")

    if architecture in {"waveform", "cnvqg", "cnvqg_waveform"}:
        return CNVQGModel(**config)

    if architecture in {"complex_stft", "complex_cnvqg", "phase4"}:
        return ComplexCNVQGModel(**config)

    if architecture in {"hybrid_tf_refiner", "phase4c", "hybrid_cnvqg_tf"}:
        return HybridTFRefinerCNVQGModel(**config)

    if architecture in {
        "streaming_hybrid_v2",
        "causal_hybrid_v2",
        "cnvqg_v2",
    }:
        return StreamingHybridCNVQGV2(**config)

    if architecture in {"streaming_hybrid_v3", "efficient_hybrid_v3", "cnvqg_v3"}:
        return StreamingHybridCNVQGV3(**config)

    if architecture in {"noise_adaptive_tf_mamba", "tf_mamba_v4", "cnvqg_v4"}:
        return NoiseAdaptiveTFMamba(**config)

    if architecture in {"noise_adaptive_tf_mamba_v41", "tf_mamba_v41", "cnvqg_v41"}:
        return NoiseAdaptiveTFMambaV41(**config)

    if architecture in {
        "continuous_adaptive_tf_mamba_v42",
        "tf_mamba_v42",
        "cnvqg_v42",
    }:
        return ContinuousAdaptiveTFMambaV42(**config)

    if architecture in {
        "auxiliary_gated_tf_mamba_v43",
        "tf_mamba_v43",
        "cnvqg_v43",
    }:
        return AuxiliaryGatedTFMambaV43(**config)

    if architecture in {"causal_aux_vq_mamba_v5", "causal_mamba_v5", "cnvqg_v5"}:
        return CausalAuxVQMambaV5(**config)

    if architecture in {
        "causal_aux_vq_mamba_v51",
        "causal_v43_realtime_v44",
        "causal_mamba_v51",
        "cnvqg_v51",
        "cnvqg_v44",
    }:
        return CausalAuxVQMambaV51(**config)

    if architecture in {
        "causal_v43_recovery_v45",
        "causal_mamba_v45",
        "cnvqg_v45",
    }:
        return CausalV43RecoveryV45(**config)

    if architecture in {"causal_aux_vq_mamba_v52", "causal_mamba_v52", "cnvqg_v52"}:
        return CausalAuxVQMambaV52(**config)

    if architecture in {
        "causal_complex_mamba_v6",
        "causal_complex_mamba_v61",
        "causal_mamba_v6",
        "cnvqg_v6",
        "cnvqg_v61",
    }:
        return CausalComplexMambaV6(**config)

    if architecture in {"causal_multiscale_mamba_v7", "causal_mamba_v7", "cnvqg_v7"}:
        return CausalMultiScaleMambaV7(**config)

    if architecture in {
        "predictive_noise_vq_mamba_v8",
        "predictive_noise_vq_mamba_v83",
        "predictive_noise_vq_mamba_v84",
        "predictive_noise_vq_mamba_v89",
        "causal_predictive_noise_mamba_v8",
        "cnvqg_v8",
        "cnvqg_v83",
        "cnvqg_v84",
        "cnvqg_v89",
        "causal_temporal_core_v12",
        "cnvqg_v12",
    }:
        return PredictiveNoiseVQMambaV8(**config)

    if architecture in {
        "causal_auxiliary_gated_tf_mamba_v410",
        "causal_v43_minimal_v410",
        "cnvqg_v410",
    }:
        return CausalAuxiliaryGatedTFMambaV410(**config)

    if architecture in {
        "causal_prototype_dual_axis_mamba_v9",
        "causal_prototype_mamba_v9",
        "cnvqg_v9",
    }:
        return CausalPrototypeDualAxisMambaV9(**config)

    if architecture in {
        "causal_scale_preserving_mamba_v91",
        "causal_prototype_mamba_v91",
        "cnvqg_v91",
    }:
        return CausalScalePreservingMambaV91(**config)

    if architecture in {
        "causal_single_scale_mamba_v92",
        "causal_prototype_mamba_v92",
        "cnvqg_v92",
    }:
        return CausalSingleScaleMambaV92(**config)

    if architecture in {
        "causal_low_rank_frequency_mamba_v93",
        "causal_prototype_mamba_v93",
        "cnvqg_v93",
    }:
        return CausalLowRankFrequencyMambaV93(**config)

    if architecture in {
        "causal_temporal_detail_mamba_v94",
        "causal_prototype_mamba_v94",
        "cnvqg_v94",
    }:
        return CausalTemporalDetailMambaV94(**config)

    if architecture in {
        "causal_noise_conditioned_mamba_v95",
        "causal_prototype_mamba_v95",
        "cnvqg_v95",
    }:
        return CausalNoiseConditionedMambaV95(**config)

    if architecture in {
        "causal_representation_norm_mamba_v96",
        "causal_prototype_mamba_v96",
        "cnvqg_v96",
    }:
        return CausalRepresentationNormMambaV96(**config)

    if architecture in {
        "causal_global_frequency_mamba_v10",
        "causal_frequency_attention_mamba_v10",
        "cnvqg_v10",
    }:
        return CausalGlobalFrequencyMambaV10(**config)

    if architecture in {
        "causal_scale_aware_mamba_v11",
        "causal_contextual_scale_mamba_v11",
        "cnvqg_v11",
    }:
        return CausalScaleAwareMambaV11(**config)

    if architecture in {
        "causal_confidence_gate_v14",
        "cnvqg_v14",
    }:
        return CausalConfidenceGateV14(**config)

    if architecture in {
        "causal_oracle_residual_gate_v16",
        "cnvqg_v16",
    }:
        return CausalOracleResidualGateV16(**config)

    if architecture in {
        "causal_ordinal_residual_gate_v17",
        "cnvqg_v17",
    }:
        return CausalOrdinalResidualGateV17(**config)

    if architecture in {
        "causal_cumulative_ordinal_gate_v17",
        "cnvqg_v17_recipe3",
    }:
        return CausalCumulativeOrdinalGateV17(**config)

    if architecture in {
        "causal_feature_cumulative_gate_v17",
        "cnvqg_v17_recipe4",
    }:
        return CausalFeatureCumulativeGateV17(**config)

    if architecture in {
        "causal_utility_safety_gate_v17",
        "cnvqg_v17_recipe5",
    }:
        return CausalUtilitySafetyGateV17(**config)

    if architecture in {
        "causal_statistics_utility_gate_v17",
        "cnvqg_v17_recipe6",
    }:
        return CausalStatisticsUtilityGateV17(**config)

    if architecture in {
        "causal_two_stage_utility_gate_v18",
        "cnvqg_v18_recipe8",
    }:
        return CausalTwoStageUtilityGateV18(**config)

    raise ValueError(f"Unknown model architecture: {architecture}")
