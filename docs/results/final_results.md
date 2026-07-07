# Final CN-VQG Results

## Selected model

The selected model is the fixed-scale Mamba CN-VQG model.

- Checkpoint: `checkpoints/cnvqg_residual_mamba_fixed_scale/best.pt`
- Uses Mamba: yes
- Parameter count: 985,672
- Dataset: VoiceBank + DEMAND
- Test items: 824

## Official test results

| Metric | Noisy | Enhanced | Improvement |
|---|---:|---:|---:|
| SI-SDR | 8.4455 | 18.5014 | +10.0559 dB |
| PESQ | 1.9684 | 2.1161 | +0.1477 |
| STOI | 0.9211 | 0.9317 | +0.0107 |
| ESTOI | 0.7867 | 0.8213 | +0.0345 |

## Validation comparison

| Model | Params | SI-SDRi | PESQ | STOI | ESTOI |
|---|---:|---:|---:|---:|---:|
| Current causal-conv CN-VQG | 653,389 | +6.6776 | 1.5603 | 0.8725 | 0.7055 |
| Fixed-scale Mamba CN-VQG | 985,672 | +7.6473 | 1.6081 | 0.8783 | 0.7230 |
| No VQ | 653,389 | +6.5696 | 1.5561 | 0.8704 | 0.7036 |
| No noise conditioning | 653,389 | +6.6398 | 1.5880 | 0.8711 | 0.7040 |
| No temporal block | 518,217 | +5.8377 | 1.5209 | 0.8671 | 0.6926 |

## Notes

The first Mamba integration collapsed to an identity mapping because the learnable residual scale converged close to zero. The fixed-scale Mamba version used a non-learnable residual scale of 0.2, which prevented identity collapse and produced the best validation performance among the current-code models.

The official test set should only be used for the final selected model. Ablation comparisons are reported on the validation set.
