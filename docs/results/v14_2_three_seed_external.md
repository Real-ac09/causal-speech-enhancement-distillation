# V14.2 three-seed replication and external generalisation

## Scope and protocol

This report supersedes the single-seed uncertainty claims in
`v14_2_final_standard_test.md`. It does not change that frozen result.

The promoted V14.2 recipe was replicated from the V13 seed-1201 and
seed-1202 checkpoints with:

- the same teacher and confidence-weighted log-magnitude distillation;
- the same 0.05 distillation weight;
- the same AdamW settings and BF16 precision;
- exactly 600 training batches per epoch;
- exactly three epochs;
- no early stopping; and
- epoch 3 fixed before training, regardless of validation loss.

The standard-test and external evaluation protocol was frozen only after the
two replication checkpoints were written and hashed. No development,
standard-test, or external-test result was used to reselect an epoch,
checkpoint, recipe, loss, or architecture.

The external set is the official Microsoft DNS Challenge 1 INTERSPEECH 2020
synthetic/no-reverberation test condition: 150 paired 16 kHz clips, totalling
25 minutes. Every clean/noisy audio file is locked in the local manifest.

## VoiceBank+DEMAND standard test

There are 824 files per seed. Uncertainty uses 20,000 paired hierarchical
bootstrap samples over training seeds and files.

| Metric | V13 three-seed mean | V14.2 three-seed mean | Paired change | Paired 95% CI |
|---|---:|---:|---:|---:|
| PESQ | 2.54686 | **2.61755** | **+0.07069** | **[+0.05828, +0.08260]** |
| SI-SDR (dB) | 18.21348 | 18.20191 | -0.01157 | [-0.08164, +0.06566] |
| SI-SDRi (dB) | 9.76796 | 9.75639 | -0.01157 | [-0.08164, +0.06566] |
| STOI | 0.934304 | **0.936918** | **+0.002614** | **[+0.000822, +0.004773]** |
| ESTOI | 0.838635 | **0.841050** | **+0.002414** | **[+0.000317, +0.005628]** |

The PESQ change is consistent across all seeds:

| Base/training seed | PESQ change | SI-SDR change | STOI change | ESTOI change |
|---|---:|---:|---:|---:|
| 1200 | +0.07809 | +0.06841 dB | +0.000898 | +0.000594 |
| 1201 | +0.06905 | -0.08454 dB | +0.004873 | +0.005759 |
| 1202 | +0.06493 | -0.01859 dB | +0.002071 | +0.000890 |

The defensible interpretation is that V14.2 improves PESQ, STOI, and ESTOI.
SI-SDR is statistically neutral: its mean change is negligible and its
confidence interval spans zero. It would be incorrect to claim an SI-SDR
improvement from the multi-seed result, but there is also no evidence of a
material SI-SDR loss.

## DNS1 external test

### Absolute performance

| Metric | Noisy input | V13 three-seed mean | V14.2 three-seed mean |
|---|---:|---:|---:|
| PESQ | 1.58224 | 1.78264 | **1.84210** |
| SI-SDR (dB) | 9.22954 | 11.44691 | **11.57601** |
| STOI | **0.915185** | 0.898391 | 0.905349 |
| ESTOI | 0.809877 | 0.816654 | **0.823589** |

### V14.2 change relative to V13

| Metric | Paired mean change | Paired 95% CI | Interpretation |
|---|---:|---:|---|
| PESQ | **+0.05946** | **[+0.01214, +0.11677]** | significant improvement |
| SI-SDR | +0.12910 dB | [-0.32431, +0.49392] | statistically neutral |
| STOI | +0.006959 | [-0.004551, +0.020196] | statistically neutral |
| ESTOI | +0.006935 | [-0.007472, +0.023978] | statistically neutral |

The external PESQ improvement replicates, although its seed-to-seed size is
more variable (+0.0118, +0.1205, and +0.0461). V14.2 also sharply reduces
external seed variance relative to V13:

| Metric | V13 seed standard deviation | V14.2 seed standard deviation |
|---|---:|---:|
| PESQ | 0.06054 | **0.00719** |
| SI-SDR | 0.66312 dB | **0.33239 dB** |
| STOI | 0.01986 | **0.00739** |
| ESTOI | 0.02357 | **0.00719** |

This supports a useful secondary finding: privileged distillation appears to
stabilise the deployment model across base-model seeds on the external domain.

## Remaining generalisation problem

On DNS1, V14.2 improves PESQ by 0.260, SI-SDR by 2.35 dB, and ESTOI by 0.0137
relative to the noisy input. However, absolute STOI decreases by 0.00984
(0.91519 to 0.90535). V14.2 is less harmful than V13 on average, but neither
model preserves DNS1 STOI.

This is now the clearest technical limitation:

1. The model generalises its noise-removal and perceptual-quality behaviour.
2. It does not reliably preserve intelligibility on this different noise and
   speech distribution.
3. The VoiceBank-trained non-causal teacher does not provide a sufficient
   cross-domain intelligibility safeguard.

DNS1 must now remain a test-only holdout. Any attempted correction should use
training-domain augmentation or a separate development corpus, not tune
against these 150 DNS1 test files.

## Final verdict

The three-seed evidence supports the following dissertation claim:

> Fixed-recipe confidence-weighted privileged distillation improved a causal
> 808k-parameter speech-enhancement model by 0.071 PESQ on VoiceBank+DEMAND
> and 0.059 PESQ on the independent DNS1 test condition. The standard-test
> improvement was accompanied by higher STOI and ESTOI and statistically
> neutral SI-SDR, while retaining the existing 20 ms algorithmic latency and
> one-thread CPU real-time factor of 0.372.

The model is promoted over V13, but the external STOI reduction relative to
unprocessed noisy speech prevents a broad claim that enhancement is harmless
across domains.
