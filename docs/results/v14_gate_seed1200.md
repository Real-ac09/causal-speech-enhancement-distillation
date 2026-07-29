# V14.1 causal confidence gate result

## Scope

V14.1 trains a 2,897-parameter causal residual-strength gate while keeping the
808,095-parameter V13 seed-1200 backbone frozen. Training used the VoiceBank
training partition, a 100-file epoch-selection partition, and the protected
400-file development screen. No standard-test files were used.

The run stopped at epoch 6 after five epochs without a PESQ improvement. Epoch
1 was the selected safeguard-eligible checkpoint.

## Paired development result

| Metric | V13 | V14.1 | Delta | Paired 95% CI |
|---|---:|---:|---:|---:|
| PESQ | 2.03871 | 2.04591 | +0.00720 | [+0.00545, +0.00919] |
| SI-SDR | 14.39278 dB | 14.36124 dB | -0.03155 dB | [-0.05023, -0.01562] |
| STOI | 0.88673 | 0.88760 | +0.00087 | [+0.00046, +0.00130] |
| ESTOI | 0.76666 | 0.76696 | +0.00030 | [+0.00007, +0.00055] |

PESQ improves on 97.25% of the 400 files. The selected checkpoint's mean gate
strength is approximately 0.946 on the full development set.

## Harm rates

Harm means that enhanced quality is below the corresponding noisy input.

| Metric | V13 harm rate | V14.1 harm rate | Change |
|---|---:|---:|---:|
| PESQ | 3.75% | 3.50% | -0.25 pp |
| SI-SDR | 1.25% | 1.25% | 0.00 pp |
| STOI | 11.75% | 11.25% | -0.50 pp |
| ESTOI | 6.25% | 6.50% | +0.25 pp |

Every harm-rate change is within the predeclared one-percentage-point limit.
The SI-SDR, STOI, and ESTOI mean safeguards also pass.

## Decision

Do not promote V14.1. Although its PESQ gain is statistically supported and
all no-harm safeguards pass, the +0.00720 mean gain is below the predeclared
+0.010 promotion threshold. The threshold is not changed after observing the
result.

The experiment nevertheless validates the core hypothesis: conservative
input-dependent residual control can improve PESQ and intelligibility together
without a material SI-SDR cost. Continue to V14.2 confidence-weighted
privileged distillation, retaining V14.1 as an ablation rather than the new
headline model.
