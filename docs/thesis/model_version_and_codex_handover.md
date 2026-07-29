# Model-version and Codex handover

> Status: project handover for report writing. Model development was frozen on
> 27 July 2026. The authoritative final decision is
> [`final_research_freeze.md`](../results/final_research_freeze.md), not older
> files whose titles contain the word “final”.

## 1. The short answer

The final dissertation model is **V14.2, CN-VQG-GRU-T1-PD**. It is a causal,
808,095-parameter speech-enhancement model with:

- a 20 ms causal STFT window and 10 ms hop;
- local time-frequency convolutions;
- a four-frame continuous noise-context state;
- one 232-dimensional GRU temporal layer;
- a direct scalar magnitude mask;
- unchanged noisy phase;
- causal overlap-add reconstruction;
- no Mamba, VQ, attention, phase correction, or teacher at inference.

V14.2 has exactly the same deployed architecture as V13. The only difference
is that V14.2 was trained with confidence-weighted privileged distillation
from a stronger non-causal teacher. The teacher and clean-reference confidence
calculation exist only during training.

Do not start another training programme. The remaining work is dissertation
writing, reference checking, integration, and proofreading.

## 2. Important naming warning

The project uses three overlapping kinds of version name:

1. **Research-programme version:** V13, V14.2, V15, and so on describe an
   experiment or decision stage.
2. **Implementation filename:** the final network is implemented in
   `predictive_noise_vq_mamba_v8.py`, even though its VQ and Mamba options are
   disabled.
3. **Checkpoint architecture alias:** the final checkpoints record
   `causal_temporal_core_v12`, because V12 selected the temporal core.

Therefore, all of the following describe the same deployed network graph:

```text
Final research label:       V14.2 / CN-VQG-GRU-T1-PD
Undistilled reference:      V13 / CN-VQG-GRU-T1
Architecture alias:         causal_temporal_core_v12
Python implementation:      predictive_noise_vq_mamba_v8
```

The labels must not be interpreted as four nested models. V14.2 is a V12-selected
architecture expressed through the reusable V8 implementation and trained with
the V14.2 recipe.

## 3. Version history

### 3.1 Historical baseline and V1-era experiments

The original CN-VQG work used a convolutional residual enhancer, continuous or
quantised noise conditioning, and optional temporal Mamba. The first Mamba
integration could collapse towards the identity because a learnable residual
scale approached zero. A fixed residual scale of 0.2 prevented that collapse.
The historical fixed-scale Mamba model had 985,672 parameters and obtained
2.116 PESQ on the VoiceBank-DEMAND test.

This stage established several recurring ideas: identity-safe initialisation,
noise context, temporal modelling, and residual reconstruction. It is no
longer the final model. In particular,
[`docs/results/final_results.md`](../results/final_results.md) is a historical
snapshot and is superseded by the 27 July research freeze.

### 3.2 V2: streaming waveform-plus-TF hybrid

V2 was an ambitious two-stage streaming design. A causal strided waveform
encoder split into speech and noise paths. Temporal Mamba processed the speech
latent; an EMA vector quantiser represented the noise latent and conditioned
the speech path. A waveform decoder produced a base estimate, after which a
causal subband STFT refiner predicted bounded gain and phase corrections.

The implemented teacher, student, and tiny presets were approximately 24.9 M,
4.63 M, and 1.51 M parameters. The design was useful for defining streaming
interfaces, explicit recurrent state, distillation hooks, and identity-safe TF
correction. It was too complex and large for the eventual sub-million-parameter
research direction and was not promoted.

### 3.3 V3: quality-oriented hybrid revision

V3 retained the waveform-plus-TF concept but restored a learned
analysis/synthesis shape with bounded look-ahead. It compressed the VQ token
before broadcasting it across the TF grid, retained identity-initialised TF
refinement, and replaced unconstrained complex masks with bounded magnitude
gain and small phase residuals. It supplied roughly 1 M, 2.5 M, and 8.6 M
variants.

V3 helped expose the trade-off between quality-oriented learned waveform
frontends and strict causal deployment. It also motivated the later decision
to use a simpler, explicitly causal STFT-first system. It was not the final
architecture.

### 3.4 V4: recurrent noise-adaptive TF-Mamba

V4 moved to a TF-primary model. Power-compressed magnitude and phase features
entered a frequency encoder. One shared cell repeatedly applied local
convolution, temporal Mamba, and frequency Mamba. Separate heads estimated a
bounded magnitude mask, phase correction, and phase confidence. Optional EMA
noise codes modulated the recurrent dynamics, while adaptive computation could
choose the number of refinement iterations.

The main lesson was that the strong TF baseline mattered more than the
discrete noise code or adaptive mechanisms. The codebook did not demonstrate a
reliable test benefit.

### 3.5 V4.1–V4.3: stronger non-causal TF reference

V4.1 added full- and half-resolution skip connections, learned frequency
upsampling, separate magnitude and phase branches, segment-level noise codes,
and richer phase objectives. Later V4.2/V4.3 experiments refined the
curriculum, auxiliary VQ and perceptual training. V4.3 became a strong
approximately 1.15 M-parameter project reference and later served as the
privileged teacher.

V4.3 was not a valid final real-time baseline: it used a centred STFT,
symmetric temporal convolutions, and utterance-global noise context. Converting
these operations to causal equivalents caused a large quality loss. It should
be described as a non-causal teacher/reference, not a deployable competitor.

### 3.6 V4.4: causal conversion

V4.4 converted the V4.3 recipe to a causal 20 ms frontend and streaming-safe
operations. The selected epoch-17 checkpoint had 745,155 parameters. On the
locked 400-file validation set it scored 2.314 PESQ, 14.856 dB SI-SDR, 0.8969
STOI, and 0.7785 ESTOI. It was substantially below V4.3 on every metric.

This was an important negative result: matching parameter count and module
names did not preserve quality when future context and global statistics were
removed.

### 3.7 V4.5: causal recovery by loss reweighting

V4.5 removed unreliable phase objectives and increased magnitude and
compressed-complex loss weights. The short search gain did not transfer to the
locked 400-file set; PESQ changed by -0.00084 and SI-SDR fell by 0.098 dB.
Full scratch training was therefore not started.

This ruled out simple loss reweighting as a sufficient solution to the causal
quality gap.

### 3.8 V4.6: first successful privileged distillation

V4.6 used frozen non-causal V4.3 as a training-only teacher for causal V4.4.
Clean speech determined where the teacher was more trustworthy than the noisy
input. A confidence-weighted log-magnitude target improved all four protected
metrics without changing the deployed student graph. On the locked 400 files,
PESQ improved by 0.03096, SI-SDR by 0.1165 dB, STOI by 0.00122, and ESTOI by
0.00280.

V4.6 established the key training idea later reused successfully in V14.2.

### 3.9 V5–V5.2: fully causal auxiliary-VQ Mamba

V5 was redesigned around an explicitly causal STFT, one frequency downsample,
a full-resolution skip, tied dual-axis refinement, continuous causal noise
conditioning, and separate magnitude/phase heads. The student had 1.041 M
parameters and the teacher 2.629 M. VQ was either training-only or allowed
through a tightly bounded adapter.

V5 also established whole-file/chunked equivalence and an explicit streaming
state API. However, V5.2 reached only 2.2925 PESQ on the locked validation set,
0.210 below the non-causal V4.3 reference. Its complexity did not earn a
quality advantage.

### 3.10 V5.1/V5.2 diagnostics

The guarded V5.1 tournament and faster V5.2 work tested reconstruction,
optimisation and training-budget alternatives. These were controlled
diagnostics, not separate final models. Their main value was showing that
causal correctness alone was insufficient and that VQ should not be allowed to
control enhancement without direct evidence.

### 3.11 V6–V6.1: simpler causal complex model

V6 removed enhancement-path noise conditioning and iterative dual-axis
refinement. It used local causal convolutions around one temporal Mamba stack,
with the noise/VQ branch remaining auxiliary. Three reconstruction forms were
compared: Cartesian complex ratio, magnitude-only with noisy phase, and polar
residual with bounded phase correction.

V6.1's full-band polar search improved some metrics, but no version established
the required balanced advantage. The experiments helped show that elaborate
phase reconstruction was not the main limiting factor.

### 3.12 V7–V7.1: modular causal capacity

V7 tested a multiscale encoder/decoder, current-frame full-band frequency
modelling, one temporal Mamba block, and optional bounded phase detail. The
full-band operation was bidirectional only over frequency, not time. The
combined candidate produced only a small PESQ change and the cumulative
modules did not justify their complexity under the no-harm criteria.

V7.1 compared polar and multiscale reconstructions. The simpler V6 polar
candidate remained stronger in PESQ than the new multiscale variants. Neither
line was promoted.

### 3.13 V8: predictive noise prototypes and the final frontend family

V8 tested whether predicting transitions between auxiliary quantised noise
prototypes could help a continuous causal Mamba enhancer. It used a 20 ms
uncentred STFT, one frequency downsample, a full-resolution skip, a continuous
four-frame noise state, and temporal Mamba. Crucially, quantised states did not
enter the enhancement path.

The auxiliary prototype hypothesis did not earn inclusion. Subsequent V8.x
capacity and reconstruction diagnostics showed that the most effective
configuration was simpler:

- auxiliary VQ disabled;
- direct scalar magnitude mask;
- noisy phase retained;
- identity initialisation;
- magnitude-ratio supervision.

This simplified V8 graph became the frontend/decoder family retained by V12,
V13 and V14.2.

### 3.14 V9: prototype-conditioned dual-axis redesign

V9 proposed a larger causal-native multiscale system with untied time and
frequency Mamba blocks, multi-timescale noise state, direct complex
reconstruction, and auxiliary prototype VQ. It explicitly tried to address the
causal quality loss and V8 magnitude limitation.

Capacity and recovery checks did not justify a full promoted model. The design
was not carried into the final system. Its main contribution was clarifying
that adding more Mamba axes, phase losses, and prototype machinery was not
supported by the measured evidence.

### 3.15 V10: global-frequency modelling

V10 added low-rank full-band frequency attention to the active V8
direct-scalar path while preserving temporal causality. Neither the
32-dimensional nor 40-dimensional screen passed the +0.03 PESQ promotion
threshold. A later full D40 check changed PESQ by approximately -0.004 with a
confidence interval crossing zero.

Global-frequency attention was excluded from the final architecture.

### 3.16 V11: magnitude-recovery interventions

V11 diagnosed systematic magnitude-mask underestimation. Asymmetric
underestimation penalties improved STOI slightly but reduced PESQ and SI-SDR.
Explicit global-scale and frequency-context decoder adapters changed PESQ by
less than 0.001, with uncertainty intervals crossing zero.

Both the asymmetric loss and scale adapters were rejected. Phase work remained
deferred because the magnitude interventions had not passed their gate.

### 3.17 V12: temporal-core tournament

V12 retained the simplified V8 frontend, continuous noise context, scalar
mask, losses and data, and isolated the temporal/deployment structure:

| Candidate | Temporal core | TF time kernel | Parameters | Structural p95 |
|---|---|---:|---:|---:|
| Mamba control | Mamba, 232 | 3 | 1,067,471 | 10.38 ms |
| Matched GRU | GRU, 232 | 3 | 1,083,479 | 6.65 ms |
| **GRU-T1** | **GRU, 232** | **1** | **808,095** | **3.58 ms** |
| GRU128-T1 | GRU, 128 | 1 | 588,527 | 3.25 ms |

GRU-T1 was confirmed against the Mamba control on the untouched 437-file
internal reserve across three seeds. Its PESQ difference was -0.0300 with a
95% hierarchical interval of [-0.0438, -0.0157], above the predeclared -0.05
non-inferiority boundary. It therefore traded a bounded quality difference for
a large reduction in latency and persistent state.

This is the architecture-selection decision behind the final network.

### 3.18 V13: frozen supervised final hybrid

V13 renamed the selected V12 GRU-T1 model **CN-VQG-GRU-T1**, froze its three
checkpoints and selection policy, and evaluated it on the standard
VoiceBank-DEMAND test. “Hybrid” refers to the division between STFT local
spectral processing, continuous noise context, recurrent temporal memory, and
waveform overlap-add; it does not mean that several old models are ensembled.

V13 is the clean supervised reference for measuring V14.2's training-only
improvement.

### 3.19 V14.0: fixed residual-strength screen

V14.0 tested

```text
output = noisy + strength × (enhanced - noisy)
```

with fixed strengths 0.80, 0.90, 0.95, and 1.00. Strength 0.95 improved PESQ
but violated the SI-SDR and ESTOI safeguards. A single global strength could
not solve both near-clean perturbation and aggressive attenuation.

### 3.20 V14.1: learned causal residual gate

V14.1 added a 2,897-parameter frame-level strength gate around a frozen V13
backbone. It passed the harm safeguards but improved PESQ by only 0.0072, below
the predeclared +0.010 promotion threshold. It remains a positive ablation,
not the final model.

### 3.21 V14.2: final privileged-distilled model

V14.2 returned to the unchanged V13 deployment graph and applied the successful
V4.6 principle. A stronger non-causal V43 teacher generated training targets.
At each causal STFT bin, the clean target compared teacher error with noisy
error. The student copied teacher log magnitude strongly where the teacher was
better and weakly where it was worse.

The selected recipe used log-magnitude distillation weight 0.05, a two-epoch
warm-up, and fixed epoch 3. It was replicated for base seeds 1200, 1201, and
1202 without epoch or checkpoint reselection. **V14.2 is the final model.**

### 3.22 V15: quiet-level and preservation experiments

DNS development diagnostics showed that full enhancement could harm
intelligibility, especially for quiet and low-SNR speech. V15 investigated
quiet-level augmentation, identity targets, mild under-enhancement and a
causal preservation gate.

Candidate A, quiet-level training, improved the difficult quiet conditions.
Candidate B's identity-target intervention was rejected. Candidate D retained
VoiceBank quality and slightly improved cross-domain metrics relative to A,
but its learned strength was almost constant at 0.998. It had not learned
meaningful input-dependent preservation. V15 remained a development ablation.

### 3.23 V16: oracle-supervised waveform gate

V16 corrected an architectural problem in V15 by blending the enhanced output
with a truly delay-aligned noisy waveform, so strength zero became a genuine
bypass. A 2,897-parameter causal controller was trained from oracle strength
labels.

The waveform residual architecture worked and remained real-time, but the
controller collapsed towards full enhancement. It passed only 11 of 17 gates,
had weak target correlation, harmed cross-domain STOI, and failed VoiceBank
SI-SDR and ESTOI safeguards. It was not promoted or replicated.

### 3.24 V17: controller learnability programme

V17 retained the frozen enhancer and waveform blend and focused on the routing
problem. Recipes progressively introduced class-balanced ordinal targets,
local two-second oracles, domain/class balancing, richer causal features,
utility/safety auxiliary heads, causal summary statistics, burn-in, and
explicit one-second prefix targets.

Recipes 1–6 each failed their predeclared learnability or safety gate and
motivated the next controlled change. Recipe 7a, which used causal statistics
and burn-in-aware supervision, was the best balanced controller: its avoidable
violation rate was 0.2040, only three items above the 0.20 limit. Recipe 7b
reduced violation severity and intelligibility-violation rates but did not
close the primary avoidable-violation gate.

Recipe 7a is retained only as the strongest preservation-controller ablation.
It did not replace V14.2.

### 3.25 V18: two-stage controller

V18 factorised routing into:

1. full enhancement versus reduced enhancement; and
2. conditional selection among strengths 0, 0.25, 0.5, and 0.75.

It passed 10 of 12 training-domain checks, but failed the avoidable-violation
limit (0.2113 versus 0.20) and reduced-class macro-accuracy requirement
(0.2497 versus 0.30). The reduced selector still collapsed towards strength
zero. Development and external test sets were not used.

V18 ended controller training. It is a negative result for the current target
formulation, not a new final model.

## 4. Exact current architecture

### 4.1 End-to-end signal path

```text
16 kHz noisy waveform
        |
        v
causal STFT
FFT 512, Hann window 320 samples, hop 160, center=False
        |
        v
per-bin input [|Y|^0.3, cos(angle Y), sin(angle Y)]
        |
        v
full-resolution causal detail encoder
3 channels -> 116 channels
frequency kernels 5 and 3; time kernel 1
        |
        v
one frequency downsample
116 channels -> 232 channels
        |
        |-------------------------------|
        |                               |
        v                               v
local depthwise bottleneck       mean over frequency
                                four-frame causal average
                                232 -> 64 noise context
        |                               |
        |<--- bounded scale/shift ------|
        v
single GRU temporal block
hidden size 232, independently over each frequency trajectory
        |
        v
frequency upsample and concatenate full-resolution skip
232 -> 116 channels
        |
        v
one zero-initialised mask logit per TF bin
M = 2 sigmoid(-logit), so 0 < M < 2 and M=1 initially
        |
        v
enhanced magnitude = M |Y|
enhanced phase = noisy phase
        |
        v
causal inverse FFT and overlap-add
        |
        v
enhanced waveform
```

### 4.2 What is active

- local time-frequency encoder and decoder;
- one frequency reduction and full-resolution skip;
- continuous four-frame noise context;
- bounded noise-conditioned bottleneck modulation;
- one projected GRU with 232-dimensional state;
- direct scalar magnitude mask;
- noisy phase;
- causal overlap-add;
- constant-sized streaming state.

### 4.3 What is inactive

- Mamba;
- auxiliary vector quantisation;
- prototype prediction;
- global-frequency attention;
- scale-context decoder adapters;
- phase correction;
- group-delay and instantaneous-frequency objectives;
- asymmetric magnitude underestimation loss;
- residual preservation gate;
- privileged teacher at inference.

### 4.4 Offline versus real-time execution

Offline inference batches all frames of a complete utterance but remains
causal. It does not use future frames or centred STFT padding.

Real-time inference consumes 160 new samples every 10 ms, waits until a
320-sample frame is available, retains the GRU state, three previous noise
summary frames, analysis samples, and overlap-add buffers, and emits one hop.
The algorithmic latency is 20 ms. Offline and streaming modes implement the
same transformation through different execution strategies.

## 5. V13 versus V14.2

| Property | V13 | V14.2 |
|---|---|---|
| Deployed architecture | GRU-T1 | Same GRU-T1 |
| Parameters | 808,095 | 808,095 |
| Algorithmic latency | 20 ms | 20 ms |
| Teacher at inference | No | No |
| Initialisation | Fresh supervised run | Corresponding V13 checkpoint |
| Supervised loss | Waveform, SI-SDR, compressed complex, log magnitude, mask ratio | Same |
| Additional training loss | None | Confidence-weighted teacher log magnitude |
| Final role | Reference | Final model |

The correct causal inference claim is therefore:

> Privileged distillation improved the training of an unchanged real-time
> student; it did not increase inference-time capacity.

## 6. Frozen final evidence

### VoiceBank-DEMAND standard comparability test

Across three seeds and 824 utterances per seed:

| System | PESQ | SI-SDR | STOI | ESTOI |
|---|---:|---:|---:|---:|
| Noisy | 1.968 | 8.446 dB | 0.9211 | 0.7867 |
| V13 | 2.547 | 18.213 dB | 0.9343 | 0.8386 |
| **V14.2** | **2.618** | **18.202 dB** | **0.9369** | **0.8410** |

Paired V14.2-minus-V13 changes were +0.07069 PESQ, -0.01157 dB SI-SDR,
+0.002614 STOI, and +0.002414 ESTOI. The PESQ, STOI and ESTOI 95% intervals
excluded zero; SI-SDR was statistically neutral.

The standard test was reused during earlier project development. It is valid
for benchmark comparability but must not be called a project-wide pristine
holdout.

### DNS1 external test

Across three seeds and 150 utterances per seed, V14.2 obtained 1.842 PESQ,
11.576 dB SI-SDR, 0.9053 STOI, and 0.8236 ESTOI. Its paired PESQ gain over V13
was +0.05946 with a 95% interval of [+0.01214, +0.11677].

The important limitation is that V14.2's DNS1 STOI remained below the noisy
input (0.9053 versus 0.9152). Do not claim universal external intelligibility
preservation.

### Runtime

On one AMD Ryzen 7 7800X3D CPU thread:

- 20 ms algorithmic latency;
- 10 ms hop deadline;
- p95 frame time 4.077 ms;
- p99 frame time 4.184 ms;
- streaming real-time factor 0.372;
- persistent FP32 state 0.119 MiB.

## 7. Files the next Codex should trust

Read these first:

1. [`docs/results/final_research_freeze.md`](../results/final_research_freeze.md)
2. [`configs/v14/frozen_replication_protocol.yaml`](../../configs/v14/frozen_replication_protocol.yaml)
3. [`configs/v14/frozen_replication_evaluation.yaml`](../../configs/v14/frozen_replication_evaluation.yaml)
4. [`docs/thesis/methods_and_experimental_setup_draft.md`](methods_and_experimental_setup_draft.md)
5. [`docs/thesis/results_chapter_draft.md`](results_chapter_draft.md)
6. [`docs/v12_temporal_core_tournament.md`](../v12_temporal_core_tournament.md)
7. [`docs/v13_final_hybrid_protocol.md`](../v13_final_hybrid_protocol.md)
8. [`docs/results/v14_2_three_seed_external.md`](../results/v14_2_three_seed_external.md)

Exact final code:

- [`src/cnvqg/models/predictive_noise_vq_mamba_v8.py`](../../src/cnvqg/models/predictive_noise_vq_mamba_v8.py)
- [`src/cnvqg/models/v8_native_streaming.py`](../../src/cnvqg/models/v8_native_streaming.py)
- [`src/cnvqg/training/distillation.py`](../../src/cnvqg/training/distillation.py)
- [`src/cnvqg/losses/losses.py`](../../src/cnvqg/losses/losses.py)
- [`src/cnvqg/metrics/speech_metrics.py`](../../src/cnvqg/metrics/speech_metrics.py)

Final checkpoints:

```text
V14.2 seed 1200:
checkpoints/v14/distillation/mag_005/epoch_003.pt

V14.2 seed 1201:
checkpoints/v14/distillation/mag_005_seed1201_fixed_epoch3/epoch_003.pt

V14.2 seed 1202:
checkpoints/v14/distillation/mag_005_seed1202_fixed_epoch3/epoch_003.pt
```

Run `python scripts/verify_final_research_freeze.py` before relying on local
artifacts if anything has changed.

## 8. Files that can mislead a new session

- `docs/results/final_results.md` describes an obsolete early “final” Mamba
  checkpoint.
- V4.3 has stronger internal quality but is non-causal and must not replace the
  final deployment model.
- V13 is the frozen supervised reference, not the latest final model.
- V15–V18 are preservation ablations and negative results, not promoted
  systems.
- The class name and filename contain VQ/Mamba for historical compatibility;
  neither is active in V14.2.
- Do not select the best of the three V14.2 seeds using test performance. The
  canonical deployment checkpoint is seed 1200, epoch 3, while final quality
  claims aggregate all three seeds.

## 9. Recommended next task

The next task is report writing, beginning with the Discussion chapter. It
should explain:

1. why GRU-T1 was selected by quality-efficiency non-inferiority;
2. why privileged distillation improved perceptual metrics without changing
   deployment;
3. why SI-SDR remained neutral while PESQ/STOI/ESTOI improved;
4. why external DNS1 supports limited generalisation but not universal
   intelligibility preservation;
5. why VQ, Mamba, attention, explicit scale correction, phase correction, and
   learned preservation controllers were excluded;
6. what the V15–V18 negative results reveal about causal risk prediction.

Do not launch more training unless report writing reveals a critical missing
verification that cannot be resolved from the frozen artifacts.

## 10. Copyable prompt for another Codex session

```text
You are continuing a frozen master's dissertation project on causal real-time
speech enhancement. Work only from repository evidence. First read:

1. docs/thesis/model_version_and_codex_handover.md
2. docs/results/final_research_freeze.md
3. docs/thesis/methods_and_experimental_setup_draft.md
4. docs/thesis/results_chapter_draft.md

V14.2 (CN-VQG-GRU-T1-PD) is the final model. Do not train or reselect models.
The final student has 808,095 parameters, 20 ms algorithmic latency, a causal
STFT/local-convolution/four-frame-noise-state/GRU/scalar-mask architecture,
and no active Mamba or VQ. V14.2 differs from V13 only through training-time
privileged distillation. VoiceBank-DEMAND is a historically reused standard
comparability test; DNS1 is the independent external test. V15–V18 are
unpromoted preservation ablations.

The immediate task is dissertation writing in Coventry University APA 7th
edition style. Preserve frozen numerical claims, distinguish development from
test evidence, use primary citations, and do not invent missing results.
```
