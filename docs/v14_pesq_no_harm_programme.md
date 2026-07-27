# V14 PESQ improvement under no-harm constraints

## Research boundary

V13 remains the frozen dissertation result. V14 is a separate exploratory
programme. The completed VoiceBank+DEMAND standard test may be used for error
description and listening examples, but not for architecture, hyperparameter,
checkpoint, or output-strength selection.

V14 screens use development metadata only. A final V14 claim requires a new
external holdout because all current internal VoiceBank validation partitions
have already participated in earlier development.

## Listening-set evidence

The fixed diagnostic set contains 24 cases:

- five largest PESQ regressions;
- five largest combined STOI/ESTOI regressions;
- three examples in each direction of metric disagreement;
- four seed-sensitive examples;
- four typical examples;
- four largest PESQ gains.

Every case contains noisy, clean, and enhanced audio from all three seeds, a
seed-1200 spectrogram/mask panel, and a blinded seed-1200 A/B comparison.

The visual and mask diagnostics reveal two failure regimes.

### Near-clean perturbation

Some high-SNR PESQ failures predict a mask close to one and obtain almost no
log-magnitude error reduction. Small, textured deviations from identity are
enough to reduce PESQ even though enhancement is weak. Examples include
`p257_304`, `p232_045`, and `p232_350`.

### Aggressive speech attenuation

The intelligibility-harm cases have a mean mask of 0.471 and suppress 25.5% of
clean-dominated bins below 0.7. The best-PESQ cases have a mean mask of 0.551
and suppress 13.7% of clean-dominated bins below 0.7. Examples such as
`p257_139` remove broadband noise successfully but also reduce visible speech
structure.

The 24 examples are deliberately stratified rather than randomly sampled, so
these mask percentages are diagnostic contrasts, not population estimates.

## Decision principle

V14 uses lexicographic selection:

1. reject candidates that violate SI-SDR, STOI, ESTOI, harm-rate, causality, or
   runtime gates;
2. among eligible candidates, select the largest statistically supported PESQ
   gain.

The development-screen gates are:

| Gate | Requirement |
|---|---:|
| PESQ mean gain | at least +0.010 |
| Paired PESQ 95% CI | lower bound above zero |
| Maximum SI-SDR drop | 0.10 dB |
| Maximum STOI drop | 0.001 |
| Maximum ESTOI drop | 0.002 |
| Maximum per-metric harm-rate increase | 1 percentage point |
| Algorithmic latency | no increase above 20 ms |
| p95 and p99 frame time | below 10 ms |

Final confirmation must use three seeds and hierarchical seed-and-file
uncertainty. Passing aggregate means alone is insufficient.

## Experiment order

### V14.0: constant residual-blend screen

Before adding a learned module, test whether the current output is globally too
strong:

`blended = noisy + strength * (enhanced - noisy)`

Strengths 0.80, 0.90, 0.95, and the 1.00 reference are evaluated from one model
forward pass per file. This adds no parameters or runtime branch and is a
falsification experiment:

- a safe winner supports insufficient identity preservation;
- no winner shows that a global strength cannot resolve the two failure
  regimes and motivates an input-dependent gate.

Prepare without loading the model:

```bash
LD_LIBRARY_PATH=/home/mohamedb/miniconda3/envs/cnvqg/lib \
PYTHONPATH=src \
/home/mohamedb/miniconda3/envs/cnvqg/bin/python \
  scripts/run_v14_blend_screen.py
```

Execute later:

```bash
LD_LIBRARY_PATH=/home/mohamedb/miniconda3/envs/cnvqg/lib \
PYTHONPATH=src \
/home/mohamedb/miniconda3/envs/cnvqg/bin/python \
  scripts/run_v14_blend_screen.py \
  --device cuda \
  --execute
```

The script refuses the standard-test metadata by digest.

#### Completed result

The seed-1200 development screen rejected every constant strength. Strength
`0.95` increased PESQ by 0.1151 (paired 95% CI 0.0956 to 0.1343) and improved
STOI by 0.00198, but reduced SI-SDR by 0.2279 dB and ESTOI by 0.00291. It
therefore failed the predeclared safeguards.

A diagnostic, non-deployable per-file oracle found that selective blending
could improve PESQ by 0.02420 while slightly improving SI-SDR, STOI, and
ESTOI. This supports an input-dependent gate but is not itself a candidate
result because it uses clean-reference metrics on the development files.

### V14.1: causal confidence residual gate

If no constant blend passes, add a small frame-level gate:

`final = noisy + g_t * (enhanced - noisy)`, with `0 <= g_t <= 1`.

The gate may consume only causal noisy-input summaries and the existing
continuous noise state. It must not consume clean SNR at inference. Start with
one scalar per frame, an EMA-smoothed state, fewer than 10,000 parameters, and
initial behaviour close to `g_t = 1`.

Train the gate with the V13 backbone frozen first. Include near-clean and clean
training examples and penalise deviation from identity when the training pair
shows little removable noise. Only consider a low-learning-rate joint
fine-tune after the frozen-backbone gate passes.

#### Completed result

The 2,897-parameter seed-1200 gate stopped at epoch 6 and selected epoch 1.
On the 400-file development screen it changed PESQ by +0.00720 (paired 95% CI
+0.00545 to +0.00919), SI-SDR by -0.03155 dB, STOI by +0.00087, and ESTOI by
+0.00030. All mean and harm-rate safeguards passed, but the PESQ gain did not
reach the predeclared +0.010 minimum. V14.1 is therefore retained as a positive
ablation and not promoted.

### V14.2: confidence-weighted privileged distillation

If gating preserves easy cases but does not increase PESQ enough, test
distillation from a teacher that is demonstrably better on the new development
set. Copy log-magnitude targets only where the teacher is closer to clean than
the noisy input.

This ordering is evidence-based: the repository's V4.6 experiment improved
PESQ, SI-SDR, STOI, and ESTOI simultaneously with confidence-weighted
distillation and did not change the deployment graph. Existing teachers must
still be re-audited; a teacher that does not beat V13 on the V14 development
set is ineligible.

### V14.3: constrained perceptual fine-tuning

A calibrated PESQ regressor is a later option, not the first response. Require
the existing calibration gate (Pearson at least 0.80 and MAE at most 0.12),
ramp a small generator weight from zero, retain at least half-strength
reconstruction losses, and apply the no-harm checkpoint safeguards every
epoch.

This stage has greater metric-gaming risk and requires blinded listening.

## Rejected or deferred directions

- Do not repeat asymmetric magnitude reweighting: it reduced PESQ and SI-SDR.
- Do not repeat the explicit scale adapter: its PESQ change was negligible.
- Do not repeat low-rank global-frequency attention unchanged: it failed its
  promotion gate.
- Do not place auxiliary VQ in the enhancement path.
- Do not add phase objectives without a new isolated reconstruction result;
  previous phase losses did not establish a reliable benefit.
- Do not use PCS as the headline model. PESQ-oriented target transforms may be
  reported separately but do not answer the no-harm question.

## Listening protocol

Use `results/v13/listening_set/blind/CASE###` first. Listen to the clean
reference, score candidates A and B for speech distortion, residual noise,
musical noise, consonant clarity, and overall preference, then open
`blind_key.csv`. Do not alter V14 selection rules based on these standard-test
examples.

The clear diagnostic directories contain the same case's named noisy, clean,
three-seed enhanced audio, and the spectrogram/mask panel.
