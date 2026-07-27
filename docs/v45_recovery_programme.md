# V4.5 causal recovery programme

## Established baselines

The locked-400 evaluator scores complete utterances. The historical training
loop instead selected checkpoints on deterministic four-second centre crops.
The difference is material:

| Model | Whole utterance PESQ | Four-second PESQ | Crop delta |
|---|---:|---:|---:|
| V4.3 | 2.5023 | 2.4292 | -0.0731 |
| V4.4 epoch 17 | 2.3142 | 2.2240 | -0.0902 |

`scripts/evaluate.py` now records chunk length and model/EMA weight selection,
and `scripts/train.py` supports whole-utterance perceptual validation. Search
runs retain fast crop ranking and externally evaluate every candidate on whole
utterances; whole-utterance Mamba evaluation inside a fresh training process
has significant variable-shape compilation overhead.

## Causality decomposition

`scripts/audit_causality_components.py` measures output and intermediate-stage
future sensitivity, streaming equivalence, frontend latency, causal
convolutions, and frame-local normalization.

| Property | V4.3 | V4.4 |
|---|---:|---:|
| Algorithmic latency | 32 ms | 20 ms |
| Future output max difference | 0.02771 | 0.0 |
| Encoder future difference | 0.89784 | 0.0 |
| Core future difference | 8.72033 | 0.0 |
| Decoder future difference | 0.64664 | 0.0 |
| Streaming/whole max difference | unavailable | 1.28e-5 |

V4.3 is not a causal deployment baseline. Its advantage cannot be attributed
to one isolated noncausal component because the frontend, ordinary GroupNorm,
symmetric temporal convolutions, and core all consume future context.

## V4.5 model

Architecture name: `causal_v43_recovery_v45`.

The minimal variant retains the proven V4.4 causal analysis/synthesis,
frame-local normalization, dual-axis core, and high-resolution skip decoder.
It defaults to one tied pass and fully disabled VQ while retaining continuous
noise conditioning. Disabled VQ is bitwise independent of codebook contents.

Structural gates pass at 745,155 parameters, 20 ms algorithmic latency, zero
future sensitivity, exact reference streaming agreement in the structural
probe, and finite BF16 gradients.

Inference-only gates rejected two proposed simplifications before training:

| Probe | PESQ | SI-SDR | STOI | ESTOI |
|---|---:|---:|---:|---:|
| Two-pass, hop 160 control | 2.3610 | 15.188 | 0.89276 | 0.76926 |
| One pass | 1.9944 | 14.044 | 0.85584 | 0.71016 |
| Two-pass, hop 128 | 2.3606 | 15.230 | 0.88668 | 0.76663 |

One pass loses about 0.367 PESQ. Hop 128 is PESQ-neutral and violates the STOI
safeguard. Neither is permitted into the short training tournament.

## Active tournament

The guarded programme trains three four-epoch, 600-batch continuation runs
from the same V4.4 epoch-17 checkpoint:

1. Exact continuation control.
2. Phase-free objective matching noisy-phase deployment.
3. Phase-free objective with stronger magnitude and compressed-complex losses.

Every candidate receives a whole-utterance evaluation on the fixed 100-file
search set. A combined candidate is trained only from factors adding at least
0.005 PESQ without losing more than 0.15 dB SI-SDR, 0.002 STOI, or 0.003
ESTOI. Full scratch training is launched only when the final non-control winner
beats the control by at least 0.01 PESQ under the same safeguards.

The active log is `logs/v45/recovery_programme.log`. Final search decisions
are written to `results/v45/search/decision.json`; the scratch configuration is
written to `configs/v4/generated/v45/promoted_scratch.yaml`.

## Final outcome

The combined magnitude-focused objective won the 100-file search by 0.0069
PESQ but failed the 0.01 scratch gate. On locked-400 it scored 2.3134 PESQ
versus 2.3142 for V4.4 epoch 17 and lost 0.098 dB SI-SDR. The candidate was
rejected and scratch training was not started. See `results/v45/RUN_CONCLUDED.md`.
