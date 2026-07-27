# V16 oracle-gate postmortem

## Scope

This is a post-hoc diagnosis of the frozen V16 seed-16040 candidate. V16 uses
the frozen Candidate-A enhancer, a 2,897-parameter causal controller, and a
true waveform residual blend:

`output = noisy + strength * (enhanced - noisy)`

The analysis uses the training and training-calibration oracle labels, the
same 60-file cross-domain development set and 400-file VoiceBank development
set used by the frozen promotion protocol, and the selected epoch-5
checkpoint. All additional inference was CPU-only.

- External test: not used.
- Candidate retuning: not performed.
- Status: descriptive diagnosis after the frozen 11/17 gate result.
- Promotion decision: do not promote or replicate V16.

## Verdict

V16 validates the waveform-residual hybrid architecture but not the learned
controller. It improves substantially over Candidate A on the cross-domain
development set while remaining real-time, but its predicted strength is
almost always near full enhancement and has almost no relationship to the
oracle target.

On the speaker-disjoint training-calibration set:

- target/prediction correlation is 0.0967 overall and -0.0337 on DNS;
- raw nearest-grid accuracy is 70.8%, but macro accuracy across the five
  targets is only 20.4%, approximately the chance level;
- DNS target mean is 0.592 while predicted mean is 0.951;
- target 0 is predicted as 0.964 and target 1 as 0.961;
- macro target MAE is 0.471, despite the misleading raw MAE of 0.139.

The controller therefore learned "use almost full enhancement" rather than
"predict when full enhancement is unsafe." The cross-domain gain comes from a
small global relaxation and a few transient low-strength frames, not reliable
input-dependent control.

## Frozen result

### Cross-domain development

| Metric | Noisy | Candidate A | V16 | V16 minus A | V16 minus noisy |
|---|---:|---:|---:|---:|---:|
| PESQ | 1.38804 | 1.41243 | 1.51468 | +0.10224 | +0.12664 |
| SI-SDR | 6.3684 dB | 6.2431 dB | 6.8620 dB | +0.6189 dB | +0.4936 dB |
| STOI | 0.826849 | 0.781990 | 0.815638 | +0.033648 | -0.011211 |
| ESTOI | 0.680075 | 0.649854 | 0.692160 | +0.042306 | +0.012085 |

V16 improves every metric relative to Candidate A. It nevertheless harms
STOI on 36/60 files, including 11/12 files at -5 dB SNR. The mean STOI harm is
the central reason it is not competitive.

### VoiceBank development

| Metric | Candidate A | V16 | V16 minus A | Frozen threshold | Result |
|---|---:|---:|---:|---:|---|
| PESQ | 2.15765 | 2.15197 | -0.00568 | >= -0.010 | pass |
| SI-SDR | 14.4334 dB | 14.3250 dB | -0.1084 dB | >= -0.050 | fail |
| STOI | 0.887730 | 0.888048 | +0.000319 | >= -0.001 | pass |
| ESTOI | 0.766911 | 0.764097 | -0.002815 | >= -0.001 | fail |

### Gate and deployment result

V16 passed 11/17 gates. The six failures were:

1. cross-domain mean STOI versus noisy: -0.01121, required >= 0;
2. cross-domain STOI harm rate: 60%, required <= 40%;
3. -35 dBFS mean STOI versus noisy: -0.01336, required >= -0.005;
4. -35 dBFS STOI harm rate: 60%, required <= 50%;
5. VoiceBank SI-SDR versus Candidate A: -0.1084 dB, required >= -0.05;
6. VoiceBank ESTOI versus Candidate A: -0.002815, required >= -0.001.

All deployment constraints passed:

- 810,992 total parameters;
- 20 ms algorithmic latency;
- 0.375 CPU streaming RTF;
- 3.85 ms one-thread CPU p95 per 10 ms hop.

## What the controller learned

| Diagnostic | Calibration | Cross-domain | VoiceBank |
|---|---:|---:|---:|
| Mean predicted strength | 0.959 | 0.951 | 0.959 |
| File-level strength range | 0.640–0.964 | — | — |
| File-level strength SD | — | 0.0343 | 0.0278 |
| Mean within-file SD | — | 0.0258 | 0.00781 |
| Frames below 0.90 | — | 4.13% | 1.96% |
| Frames below 0.75 | — | 2.29% | 0.88% |

Calibration predictions by oracle target make the collapse explicit:

| Oracle target | Items | Mean prediction | MAE | Nearest-grid accuracy |
|---:|---:|---:|---:|---:|
| 0.00 | 7 | 0.9637 | 0.9637 | 0% |
| 0.25 | 9 | 0.9579 | 0.7079 | 0% |
| 0.50 | 14 | 0.9389 | 0.4389 | 0% |
| 0.75 | 47 | 0.9562 | 0.2062 | 2.1% |
| 1.00 | 183 | 0.9612 | 0.0388 | 100% |

The model does not merely lack enough dynamic range. It predicts almost the
same value for every class and even assigns a slightly higher mean to target
0 than target 1.

Cross-domain harmed files receive a mean strength of 0.9445 versus 0.9603 for
unharmed files. The direction is weakly correct but the 0.016 difference is
far too small. The worst harmed examples still commonly receive mean
strengths above 0.95.

## Why the supervision collapsed

### 1. Smooth-L1 rewards the majority full-strength solution

The raw training labels are 79.9% strength 1. VoiceBank is 90.4% full
strength; DNS is only 27.5% full strength. The training sampler balances the
two domains, which lowers the expected full-strength fraction to 59.0%, but
that is still above one half. For an uninformative predictor, the median-like
Smooth-L1 optimum is therefore near strength 1.

The speaker-disjoint calibration set is not domain balanced: 200/260 items
are VoiceBank, and 70.4% of all calibration targets are strength 1. The
checkpoint objective is consequently dominated by the easy full-strength
class.

This bias is visible across epochs:

| Epoch | Validation supervision loss | Mean validation strength |
|---:|---:|---:|
| 1 | 0.13955 | 0.8933 |
| 2 | 0.12075 | 0.9442 |
| 3 | 0.11716 | 0.9543 |
| 4 | 0.11641 | 0.9585 |
| 5 | 0.11634 | 0.9591 |

The selected loss improves as the prediction saturates. The checkpoint rule
therefore measures majority-class fit rather than preservation-controller
skill.

### 2. An utterance target is applied to every causal frame

Each four-second mixture has one clean-reference oracle strength, but the
gate must predict that value at every frame using only current and past
mixture features. The oracle decision depends on whole-utterance PESQ,
SI-SDR, STOI, and ESTOI. Early causal frames cannot observe the same evidence,
and local acoustic conditions can change within an utterance.

Frame-wise replication turns an already imbalanced item target into thousands
of correlated majority-class targets. It also supplies no explicit speech
activity or valid-length mask.

### 3. Padded silence is labelled as full enhancement

VoiceBank utterances are padded to four seconds. On average, 29.2% of each
VoiceBank item is padding, and padding belonging to full-strength items is
estimated to occupy 26.6% of all VoiceBank frames. These frames are supervised
with the utterance's mostly-full target.

This creates a plausible shortcut: low-energy or silent frames imply strong
enhancement. The observed behavior is consistent with it. Cross-domain mean
gate strength is negatively correlated with noisy RMS (-0.474), and strength
is highest at the quietest level:

| Clean level | DNS training target mean | Cross predicted mean | Cross STOI gain |
|---:|---:|---:|---:|
| -35 dBFS | 0.546 | 0.964 | -0.01336 |
| -30 dBFS | 0.650 | 0.961 | -0.01706 |
| -25 dBFS | 0.625 | 0.954 | +0.00410 |
| -20 dBFS | 0.692 | 0.925 | -0.01851 |

The DNS labels ask for less enhancement at -35 dBFS than at -20 dBFS, but the
learned controller does the opposite on development data. Padding is not
proven to be the sole cause, but it is a concrete confound that must be
removed.

### 4. The input representation may be insufficient for the oracle task

Even within DNS calibration, where lower targets are common, target/prediction
correlation is -0.034 and MAE is 0.374. Rebalancing alone may prevent the
constant-full optimum, but it does not prove that the current frame summaries
and frozen noise state contain enough information to predict clean-reference
perceptual risk.

This is why V17 needs a training-domain learnability gate before another
development evaluation.

## What worked

V16 should not be described as a failed architecture:

1. The true waveform bypass fixed V15's non-identity zero-strength path.
2. Native and offline streaming match, and recurrent state remains bounded.
3. The hybrid retains the real-time and parameter budget.
4. A small relaxation improves all four cross-domain metrics relative to
   Candidate A.
5. The oracle label distribution confirms that scalar control remains
   relevant: DNS target strengths cover the full grid.

The failed component is the mapping from causal mixture evidence to the
desired scalar strength. A bandwise controller would add outputs without
fixing that mapping and is not yet justified.

## Bounded V17 recommendation

The V17 research hypothesis should be:

> A class-balanced causal risk router trained on local, valid speech regions
> can distinguish preservation-needed from full-enhancement frames and drive
> the existing true waveform residual path without sacrificing VoiceBank
> quality.

V17 should retain the frozen backbone and scalar waveform blend. The first
work should be a training-domain controller study, not a development-set
architecture sweep.

### Required changes

1. Replace scalar Smooth-L1 with five-way ordinal or categorical prediction
   over the frozen strength grid. Use class-balanced loss and convert class
   probabilities to expected strength for smooth deployment.
2. Balance batches over domain, oracle-strength class, clean level, and SNR.
   Do not let full-strength VoiceBank examples determine the constant optimum.
3. Use valid-frame and speech-activity masks. Padded frames must contribute
   neither gate supervision nor waveform losses.
4. Replace the four-second target replicated at every frame with local causal
   targets, preferably two-second windows with a fixed causal history and
   overlap. Alternatively supervise one pooled utterance decision only after
   sufficient causal context; do not label early frames with future-derived
   targets.
5. Select checkpoints by macro-averaged target performance across domains and
   strength classes, not raw item-averaged loss.
6. Keep the current waveform residual mixer, native streamer, parameter cap,
   and deployment gates unchanged.

### Learnability gate before development evaluation

Freeze thresholds before training. On the speaker-disjoint, domain-balanced
training-calibration split, require at minimum:

- DNS target/prediction correlation >= 0.30;
- macro nearest-grid accuracy >= 35%;
- DNS strength MAE <= 0.20;
- predicted mean strength monotonically increases across oracle classes;
- mean prediction for target 1 minus target 0 >= 0.30;
- VoiceBank MAE <= 0.10.

Current V16 values are -0.034, 20.4%, 0.374, non-monotonic, approximately
-0.003, and 0.068 respectively. If a candidate fails this training-domain
gate, it must not consume another development evaluation.

### Controlled dissertation ablation

Two training-calibration-only recipes are justified:

1. same controller with balanced ordinal loss, masking, and macro checkpoint
   selection;
2. the same recipe with local two-second oracle targets.

Select between them using only the learnability gate. Evaluate exactly one
frozen V17 candidate on the established development gates. If neither recipe
passes learnability, run a mixture-only feature separability audit before
changing architecture. A bandwise gate becomes justified only after scalar
target predictability is demonstrated but scalar deployment metrics remain
insufficient.

## Reproducible artifacts

- `scripts/analyze_v16_oracle_gate.py`
- `results/v16/oracle_gate_seed16040/postmortem/summary.json`
- `results/v16/oracle_gate_seed16040/postmortem/calibration_predictions.csv`
- `results/v16/oracle_gate_seed16040/postmortem/calibration_gate_diagnostics.csv`
- `results/v16/oracle_gate_seed16040/postmortem/cross_gate_diagnostics.csv`
- `results/v16/oracle_gate_seed16040/postmortem/voice_gate_diagnostics.csv`
- `results/v16/oracle_gate_seed16040/postmortem/cross_per_file.csv`
- `results/v16/oracle_gate_seed16040/postmortem/voice_per_file.csv`
- `results/v16/oracle_gate_seed16040/postmortem/cross_worst_stoi.csv`
- `results/v16/oracle_gate_seed16040/gate/gate_report.json`
- `docs/results/v16_oracle_gate_postmortem.md`
