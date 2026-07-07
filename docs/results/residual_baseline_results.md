# CN-VQG Residual Baseline Results

## Model

- Model: CN-VQG residual baseline
- Temporal block: causal convolution fallback
- Mamba enabled: false
- Parameters: approximately 912k
- Dataset: VoiceBank + DEMAND
- Checkpoint: `checkpoints/cnvqg_residual_baseline/best.pt`

## Validation Results

| Metric | Noisy | Enhanced | Change |
|---|---:|---:|---:|
| SI-SDR | 6.22 dB | 13.72 dB | +7.51 dB |
| PESQ | 1.414 | 1.611 | +0.196 |
| STOI | 0.8549 | 0.8782 | +0.0233 |
| ESTOI | 0.6566 | 0.7221 | +0.0655 |

## Test Results

| Metric | Noisy | Enhanced | Change |
|---|---:|---:|---:|
| SI-SDR | 8.45 dB | 18.14 dB | +9.70 dB |
| PESQ | 1.968 | 2.133 | +0.165 |
| STOI | 0.9211 | 0.9281 | +0.0071 |
| ESTOI | 0.7867 | 0.8118 | +0.0251 |

## Interpretation

The residual CN-VQG baseline improves speech quality and intelligibility over the noisy input on both validation and test splits. The strongest improvement is in SI-SDR, showing that the model substantially reduces distortion/noise relative to the clean target. PESQ, STOI and ESTOI also improve, although the intelligibility gains are smaller because the noisy VoiceBank + DEMAND test speech is already relatively intelligible.
