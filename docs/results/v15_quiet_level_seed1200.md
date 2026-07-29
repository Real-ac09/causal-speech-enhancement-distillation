# V15 candidate A: quiet-level augmentation

## Decision

Do not promote candidate A to three-seed replication. It passes 13 of 17
predeclared gates but fails the four absolute cross-domain STOI-preservation
gates. Continue to candidate B, which adds identity examples while retaining
the successful quiet-level augmentation.

This is a development decision. The DNS1 external test was not reused.

## Cross-domain development

| Metric | Noisy | V14.2 | Candidate A | A minus V14.2 |
|---|---:|---:|---:|---:|
| PESQ | 1.3880 | 1.4124 | 1.4717 | +0.0593 |
| SI-SDR | 6.368 dB | 6.243 dB | 6.769 dB | +0.526 dB |
| STOI | 0.82685 | 0.78199 | 0.79649 | +0.01450 |
| ESTOI | 0.68008 | 0.64985 | 0.67116 | +0.02130 |

The candidate-versus-V14.2 STOI improvement has a paired 95% CI of
[+0.00600, +0.02370], and the PESQ improvement has a paired 95% CI of
[+0.02774, +0.09624]. Quiet-level augmentation therefore improved quality,
distortion, STOI, and ESTOI together relative to V14.2.

The remaining failure is absolute preservation:

- STOI remains 0.03036 below noisy, with 70.0% harm.
- Quietest-level STOI remains 0.02784 below noisy, with 73.3% harm.
- The frozen thresholds require non-negative overall STOI gain, at most 40%
  harm, at least -0.005 quietest-level gain, and at most 50% quietest-level
  harm.

Compared with V14.2, the quietest-level STOI improved by 0.05252 and its harm
rate fell from 86.7% to 73.3%. The intervention is effective but insufficient.

## VoiceBank development safeguards

| Metric | A minus V14.2 | Gate | Result |
|---|---:|---:|---|
| PESQ | -0.00819 | >= -0.010 | pass |
| SI-SDR | +0.04896 dB | >= -0.050 dB | pass |
| STOI | +0.00030 | >= -0.001 | pass |
| ESTOI | +0.00053 | >= -0.001 | pass |

The deployed graph is unchanged at 808,095 parameters and 20 ms algorithmic
latency, so all deployment gates also pass.

## Next ablation

Candidate B retains the same level augmentation and adds clean-input identity
examples with probability 0.10. This directly targets the remaining
over-processing failure, including the high-SNR condition, without changing
the deployed graph or increasing runtime.
