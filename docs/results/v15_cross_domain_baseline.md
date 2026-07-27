# V15 cross-domain development baseline

## Scope

This is a development result, not an external-test result. The set was
deterministically selected from the DNS Challenge 1 training assets before
model evaluation and has no exact file-hash overlap with the downloaded DNS1
no-reverb test subset.

- Protocol: `configs/v15/frozen_cross_domain_dev_protocol.yaml`
- Models: fixed seed-1200 V13 and fixed-epoch-3 V14.2
- Execution: CPU, raw model weights, full utterances
- Set: 60 distinct clean readers and 60 distinct noise assets
- Conditions: five SNR levels from -5 to 15 dB and four clean RMS levels from
  -35 to -20 dBFS
- Uncertainty: 20,000 paired file-bootstrap samples

## Aggregate baseline

| Metric | Noisy | V13 | V14.2 | V14.2 gain over noisy | Harm rate |
|---|---:|---:|---:|---:|---:|
| PESQ | 1.3880 | 1.3944 | 1.4124 | +0.0244 | 30.0% |
| SI-SDR | 6.368 dB | 6.270 dB | 6.243 dB | -0.125 dB | 41.7% |
| STOI | 0.82685 | 0.78453 | 0.78199 | -0.04486 | 73.3% |
| ESTOI | 0.68008 | 0.66025 | 0.64985 | -0.03022 | 60.0% |

V14.2's gain over noisy has a 95% CI of [-0.04998, +0.08968] for PESQ,
[-3.454, +1.930] dB for SI-SDR, [-0.06550, -0.02558] for STOI, and
[-0.05525, -0.00552] for ESTOI. The intelligibility loss is therefore the
primary reproducible failure on this development set.

## V14.2 versus V13

| Metric | Mean paired delta | Paired 95% CI | Interpretation |
|---|---:|---:|---|
| PESQ | +0.01806 | [+0.00619, +0.03144] | V14.2 better |
| SI-SDR | -0.027 dB | [-0.235, +0.271] | inconclusive |
| STOI | -0.00254 | [-0.00657, +0.00205] | inconclusive negative mean |
| ESTOI | -0.01039 | [-0.01502, -0.00571] | V14.2 worse |

Privileged distillation is still transferring perceptual quality, but on this
set it also transfers a statistically supported ESTOI cost. A further
PESQ-only distillation-weight search is therefore not justified.

## Failure conditions

At -35 dBFS clean RMS, V14.2 loses 0.1084 PESQ, 6.144 dB SI-SDR, 0.08036
STOI, and 0.08311 ESTOI relative to the noisy input. STOI is harmed on 86.7%
of these items.

At -5 dB SNR, V14.2 loses 0.09305 STOI and 0.09421 ESTOI, with STOI harm on
91.7% of items. At 15 dB SNR, STOI is harmed on every item and SI-SDR falls
8.949 dB on average. This exposes two related but distinct problems:

1. absolute-level mismatch, especially quiet speech; and
2. over-processing when the input already contains highly usable speech.

These controlled development findings agree with the descriptive DNS1
post-test audit, where clean RMS was the strongest STOI-gain correlate and
the quietest quartile had an 86.8% STOI harm rate.

## V15 decision

Keep the 808,095-parameter causal graph unchanged for the first experiments.
The first candidate uses training-only clean-level augmentation at -35, -30,
-25, and -20 dBFS. The second adds 10% identity examples. A mild 1.10
speech-bin underestimation penalty is conditional on the second candidate
improving STOI without reaching the primary gate.

Do not immediately reuse the stronger 1.5 or 2.0 underestimation settings:
the earlier V11 screen improved STOI but caused unacceptable PESQ and SI-SDR
losses. A tiny causal preservation gate remains the fourth and final
seed-1200 option, only if the training-only candidates fail.

The exact sequence is recorded in
`configs/v15/preservation_ablation_program.yaml`; the promotion thresholds
were frozen before these baseline results in `configs/v15/promotion_gates.yaml`.
The first candidate config is `configs/v15/quiet_level_seed1200.yaml`.
