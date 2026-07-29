# Methods and experimental setup draft

> Writing status: implementation-traced draft using Coventry University's
> author-date APA 7th edition referencing convention. Replace provisional
> chapter, section, table, and equation numbers after integration into the
> dissertation template, and apply that template's fonts, spacing, margins,
> pagination, and heading styles. Apply a 1.27 cm hanging indent to every
> reference-list entry in the final typeset document.
> The final systems and evaluation procedures described here are frozen in
> [`configs/v13/frozen_final_protocol.yaml`](../../configs/v13/frozen_final_protocol.yaml),
> [`configs/v14/frozen_replication_protocol.yaml`](../../configs/v14/frozen_replication_protocol.yaml),
> and
> [`configs/v14/frozen_replication_evaluation.yaml`](../../configs/v14/frozen_replication_evaluation.yaml).

## 1. Experimental objective and design

This work investigates whether a sub-million-parameter speech-enhancement
system can improve noisy speech while meeting a strict causal real-time
constraint. The final deployment model was required to process 16 kHz speech
using a 10 ms hop, introduce no more than 20 ms of algorithmic latency, and
complete both its 95th- and 99th-percentile frame computations within the
10 ms hop deadline on one CPU thread. The study also investigated whether a
stronger non-causal model could improve this causal student during training
without increasing deployment cost.

The final experimental comparison contains two systems:

- **V13 (CN-VQG-GRU-T1):** the selected causal temporal-core model trained with
  paired noisy and clean speech.
- **V14.2 (CN-VQG-GRU-T1-PD):** the same student architecture and inference
  graph, fine-tuned using confidence-weighted privileged distillation.

The identical inference architecture is important to the experimental design.
V14.2 does not use its teacher at test time and has no additional parameters,
state, look-ahead, or algorithmic latency. Differences between V13 and V14.2
can therefore be attributed to the training procedure rather than increased
deployment capacity.

Architecture development, final model selection, and final evaluation were
separated. Candidate architectures were screened using designated validation
subsets. V13 was then frozen after confirmation on an internal reserve set.
The V14.2 recipe and epoch were selected on development data and subsequently
replicated with a fixed recipe across three seeds. The VoiceBank-DEMAND
standard test and the external DNS1 test were not used to reselect an
architecture, loss, recipe, epoch, or checkpoint after the relevant protocol
was frozen.

## 2. Datasets and partitioning

### 2.1 VoiceBank-DEMAND

The main corpus was the paired VoiceBank-DEMAND speech-enhancement dataset
(Valentini-Botinhao, 2017). Each metadata row associated a noisy waveform with
its clean reference. Table 1 gives the partitions used by the final programme.
Durations were calculated from the local processed waveforms.

**Table 1**

*VoiceBank-DEMAND Partitions Used in the Final Experiments*

| Partition | Role | Utterances | Speakers | Duration |
|---|---|---:|---:|---:|
| Training | Parameter optimisation | 10,235 | 25 | 8.306 h |
| Epoch selection | Training diagnostics and early stopping for V13 | 100 | 3 | 0.085 h |
| Architecture selection | Candidate comparison | 400 | 3 | 0.317 h |
| Internal reserve | Frozen V13 confirmation | 437 | 3 | 0.353 h |
| Standard test | Final benchmark comparison | 824 | 2 | 0.576 h |

The three development partitions were disjoint at utterance level and were
derived from the available validation material using a frozen manifest. The
epoch-selection subset contained 100 utterances, the architecture-selection
subset contained 400 utterances, and the remaining 437 utterances formed the
internal reserve. Their roles were kept separate to reduce repeated adaptation
to one validation sample.

The standard VoiceBank-DEMAND test set contained 824 utterances from speakers
`p232` and `p257`. It had been evaluated earlier in the wider project and was
therefore retained as a standard comparability benchmark rather than described
as a project-wide pristine holdout. Its results were not used to change the
frozen final systems.

### 2.2 External DNS1 evaluation set

Cross-domain generalisation was evaluated on 150 paired utterances from the
Microsoft DNS Challenge 1 synthetic no-reverberation test condition (Reddy et
al., 2020). This material totalled 0.417 h and was acquired from source commit
`70f19285c36cca4df2338f9248775ddc50980c6b`. It was not used for training,
development, recipe selection, epoch selection, or checkpoint selection, and
it had not been evaluated before the V14.2 protocol was frozen. The
no-reverberation condition was declared as the primary external condition
before evaluation.

### 2.3 Waveform preparation

Audio was loaded as 32-bit floating point. Multi-channel files were converted
to mono, and files not already sampled at 16 kHz were resampled to 16 kHz.
Samples were clipped to the interval \([-1,1]\), but no utterance-level peak
normalisation was applied. Noisy and clean files were aligned by truncating
both to the shorter length.

During training, examples were randomly cropped or zero-padded to 4 s. The
validation loader used a deterministic centred 4 s crop or pad. Final quality
evaluation was performed on complete utterances, one utterance at a time,
without chunk truncation. Model output and clean reference were aligned to
their common minimum length before metric calculation.

## 3. Causal enhancement model

### 3.1 Analysis and synthesis

Let \(y[\tau]=s[\tau]+n[\tau]\) denote noisy speech, where \(s\) is clean
speech and \(n\) is additive noise. The system used an uncentred short-time
Fourier transform (STFT) with sample rate 16 kHz, FFT size 512, Hann analysis
window of 320 samples, and hop size 160 samples. A frame therefore required
20 ms of input and a new frame was processed every 10 ms. The FFT was
zero-padded from 320 to 512 samples; it did not add future context. With no
centred padding or look-ahead, the algorithmic latency was 20 ms.

For each complete frame, the input feature at frequency \(f\) and time \(t\)
was

\[
\mathbf{x}_{f,t} =
\left[
  |Y_{f,t}|^{0.3},
  \cos(\angle Y_{f,t}),
  \sin(\angle Y_{f,t})
\right].
\tag{1}
\]

The network estimated a real scalar magnitude mask while retaining the noisy
phase. If \(a_{f,t}\) denotes the decoder output before activation, the mask
and enhanced spectrum were

\[
M_{f,t}=2\sigma(-a_{f,t}), \qquad
\widehat{S}_{f,t}=M_{f,t}|Y_{f,t}|
\exp(j\angle Y_{f,t}).
\tag{2}
\]

Thus \(M_{f,t}\in(0,2)\), allowing both attenuation and limited amplification.
The mask head was zero-initialised, making the initial mask equal to one and
the initial system an identity mapping. No phase correction was used in the
final model.

Waveforms were reconstructed by inverse real FFT followed by causal
overlap-add. The same periodic Hann window was applied during analysis and
synthesis, and the summed waveform was divided by the overlap-add window
energy. Samples with less than 0.5% overlap support at the extreme signal
boundaries were set to zero to avoid amplifying numerical error at near-zero
Hann endpoints.

### 3.2 Encoder and continuous noise context

The causal encoder operated locally in time and frequency. Its detail branch
used 116 channels and a \(5\times1\) causal convolution, followed by a
depthwise \(3\times1\) convolution and pointwise projection. A separate
\(4\times1\), frequency-stride-two convolution produced a 232-channel
latent representation. All temporal kernel sizes were one, so these
convolutions introduced no temporal look-ahead or growing temporal cache.
Frame-wise group normalisation and SiLU activations were used.

A continuous noise-context vector was computed from the encoded features.
Features were averaged across frequency, and the current frame was averaged
with the preceding three frames. At an utterance boundary, the first frame was
replicated to supply the missing history. A linear projection mapped this
four-frame rolling summary from 232 to 64 dimensions, followed by SiLU and
layer normalisation.

The 64-dimensional context modulated the bottleneck through learned scale and
shift terms. For bottleneck activation \(\mathbf{h}_{f,t}\), the modulation
had the form

\[
\widetilde{\mathbf{h}}_{f,t}
=
\mathbf{h}_{f,t}
\odot
\left[1+0.05\tanh(\boldsymbol{\gamma}_{t})\right]
+0.05\boldsymbol{\beta}_{t}.
\tag{3}
\]

This bounded initial influence allowed the network to adapt to recent acoustic
conditions without depending on a discrete noise label. Although the
historical implementation contains an auxiliary vector-quantisation module,
auxiliary VQ was disabled in the final V13 and V14.2 systems and did not
contribute to either enhancement or the training objective.

### 3.3 Temporal core and decoder

The temporal core was a single projected gated recurrent unit (GRU; Cho et al.,
2014). Each frequency trajectory was treated as a temporal sequence. The
232-dimensional input was layer-normalised and passed through a single-layer
GRU with hidden size 232, followed by a 232-dimensional output projection. A
residual connection used a learnable layer scale initialised to 0.1. A second
bounded residual scale joined the temporal output to the bottleneck.

The decoder projected the bottleneck from 232 to 116 channels, upsampled along
frequency using nearest-neighbour interpolation, and concatenated the result
with the full-resolution encoder detail features. A \(5\times1\) causal
convolution, depthwise \(3\times1\) convolution, and pointwise output head
produced the scalar mask in Equation 2.

The final network contained 808,095 trainable parameters. Mamba blocks, global
frequency attention, phase correction, the scale-decoder adapter, auxiliary
VQ, and asymmetric magnitude loss were all disabled. This detail prevents the
historical source filename, `predictive_noise_vq_mamba_v8.py`, from being
mistaken for a description of the selected inference graph.

### 3.4 Offline and streaming inference

Offline inference analysed all complete frames of an utterance in one tensor
operation and padded the final incomplete frame when required. Streaming
inference received exactly 160 new samples per call, retained the samples
needed to form the next 320-sample analysis frame, and emitted one 160-sample
hop through causal overlap-add. The streamer maintained the GRU hidden state,
the three preceding noise-summary frames, input samples, and overlap-add
buffers.

Both modes implemented the same causal transformation. Offline processing did
not enable centred STFT padding or future context, while streaming processing
used constant-sized state rather than recomputing the full utterance history.
Consequently, the offline mode was a batched execution route for the causal
model, not a separate non-causal enhancement model.

## 4. Supervised V13 training

### 4.1 Objective function

V13 was trained using a weighted sum of four active objectives:

\[
\mathcal{L}_{\mathrm{sup}}
=0.05\mathcal{L}_{\mathrm{wav}}
+0.01\mathcal{L}_{\mathrm{SI\text{-}SDR}}
+0.50\mathcal{L}_{\mathrm{cSTFT}}
+0.25\mathcal{L}_{\log |S|}
+1.00\mathcal{L}_{\mathrm{ratio}}.
\tag{4}
\]

The waveform term was a Charbonnier penalty with
\(\epsilon=10^{-3}\):

\[
\mathcal{L}_{\mathrm{wav}}
=
\frac{1}{N}\sum_{\tau}
\sqrt{(\widehat{s}[\tau]-s[\tau])^2+\epsilon^2}.
\tag{5}
\]

The SI-SDR term was the negative mean scale-invariant signal-to-distortion
ratio, calculated after removing the temporal mean from the estimate and
target (Le Roux et al., 2019).

The compressed-complex STFT term used an STFT with FFT size 512, hop 160, and
window 320. If
\(\mathcal{C}_{p}(Z)=|Z|^p Z/(|Z|+\varepsilon)\), with \(p=0.3\), then

\[
\mathcal{L}_{\mathrm{cSTFT}}
=
\frac{
  \operatorname{mean}
  |\mathcal{C}_{0.3}(\operatorname{STFT}(\widehat{s}))
   -\mathcal{C}_{0.3}(\operatorname{STFT}(s))|
}{
  \operatorname{mean}
  |\mathcal{C}_{0.3}(\operatorname{STFT}(s))|
}.
\tag{6}
\]

The log-magnitude loss was

\[
\mathcal{L}_{\log |S|}
=
\operatorname{mean}
\left|
\log(1+|\widehat{S}|)
-\log(1+|S|)
\right|.
\tag{7}
\]

Finally, the magnitude-ratio target was the phase-agnostic ideal amplitude
mask capped at one:

\[
R_{f,t}=\min\left(\frac{|S_{f,t}|}{|Y_{f,t}|+\varepsilon},1\right).
\tag{8}
\]

Its L1 error was weighted towards bins containing clean speech:

\[
\mathcal{L}_{\mathrm{ratio}}
=\operatorname{mean}_{f,t}
\left[
  w_{f,t}|M_{f,t}-R_{f,t}|
\right],
\tag{9}
\]

where the weights were proportional to \(|S_{f,t}|^{0.3}\), normalised to
mean one within an example, and clipped to \([0.1,5]\). Noise-only regions
therefore retained a non-zero learning signal. All VQ, mel, explicit
noise-prediction, phase, group-delay, instantaneous-frequency, and compute
penalties had zero weight.

### 4.2 Optimisation

Three V13 models were trained using seeds 1200, 1201, and 1202. AdamW
(Loshchilov & Hutter, 2019) used an initial learning rate of \(10^{-4}\),
weight decay \(10^{-5}\), and gradient-norm clipping at 1.0. Training used
bfloat16, batch size 6, and a maximum of 40 epochs. The learning rate was
warmed up for three epochs and then reduced using cosine decay to 5% of its
initial value. Early stopping used patience 8 and minimum improvement 0.003.
The selected checkpoint maximised full-utterance validation PESQ on at most
100 designated epoch-selection utterances.

## 5. Privileged-distillation V14.2 training

### 5.1 Teacher eligibility and student initialisation

Each V14.2 student was initialised from the corresponding trained V13 seed.
The teacher was a larger, non-causal V43 enhancement model used only during
training. Before distillation, it was audited against V13 on the protected
400-utterance development subset and improved all four protected metrics:
PESQ, SI-SDR, STOI, and ESTOI. This audit established that the teacher provided
useful privileged targets rather than merely a different prediction.

The teacher received the noisy waveform to generate its enhanced target. Clean
speech was used to calculate how trustworthy the teacher was at each
time-frequency location. Neither the teacher nor clean reference was available
to the deployed model.

### 5.2 Confidence-weighted distillation

Teacher and noisy-speech errors were measured against clean speech on the
student's causal STFT grid in \(0.3\)-compressed magnitude. Let
\(e^{(T)}_{f,t}\) and \(e^{(Y)}_{f,t}\) be the teacher and noisy errors. The
detached confidence was

\[
c_{f,t}
=
0.05+0.95
\sigma\left(
\frac{e^{(Y)}_{f,t}-e^{(T)}_{f,t}}{0.05}
\right).
\tag{10}
\]

Confidence approached one where the teacher was better than the noisy input
and approached a floor of 0.05 where it was worse. Detaching the confidence
prevented optimisation from changing the weighting mechanism itself.

The selected distillation term matched student and teacher log magnitudes:

\[
\mathcal{L}_{\mathrm{PD}}
=
0.05\operatorname{mean}_{f,t}
\left[
c_{f,t}
\left|
\log(|\widehat{S}^{(S)}_{f,t}|+\varepsilon)
-\log(|\widehat{S}^{(T)}_{f,t}|+\varepsilon)
\right|
\right].
\tag{11}
\]

The full V14.2 objective was

\[
\mathcal{L}_{\mathrm{V14.2}}
=
\mathcal{L}_{\mathrm{sup}}
+r_e\mathcal{L}_{\mathrm{PD}},
\tag{12}
\]

where the two-epoch warm-up gave \(r_e=0.5\) in epoch 1 and \(r_e=1\) from
epoch 2 onward. No waveform or compressed-complex teacher loss was active in
the selected recipe.

### 5.3 Recipe selection and fixed replication

Distillation log-magnitude weights 0.02 and 0.05 and candidate epochs 1--4
were screened using development data. Weight 0.05 at epoch 3 was selected.
Against V13 on the protected 400-utterance development set, it improved PESQ
by 0.11894 with a paired 95% confidence interval of
\([0.10395,0.13358]\), while satisfying the predeclared SI-SDR, STOI, ESTOI,
per-file harm, latency, and runtime gates.

After this selection, the recipe was frozen. Seeds 1200, 1201, and 1202 used
distillation random-number seeds 1420, 1421, and 1422, respectively. Training
used AdamW with learning rate \(2.5\times10^{-5}\), weight decay \(10^{-5}\),
gradient clipping at 2.0, bfloat16, batch size 4, and at most 600 training
batches per epoch. Every replication ran exactly three epochs and used
`epoch_003`, regardless of validation loss. Early stopping and
development-driven checkpoint reselection were disabled.

## 6. Model-selection protocol

The temporal-core tournament compared four causal structural candidates:
a Mamba control, a parameter-matched GRU, the selected one-layer
time-kernel-one GRU (GRU-T1), and a smaller GRU128-T1. Candidates first had to
meet the one-thread 10 ms frame deadline. The Mamba control exceeded the
deadline, whereas GRU-T1 used 808,095 parameters, required 0.119 MiB of
persistent FP32 state, and achieved a structural p95 frame time of 3.577 ms.

The selected GRU-T1 model was then compared with the Mamba quality control on
the untouched 437-utterance internal reserve across three paired seeds.
GRU-T1's PESQ difference was \(-0.03004\), with hierarchical 95% confidence
interval \([-0.04382,-0.01574]\). Because the lower bound remained above the
predeclared non-inferiority margin of \(-0.05\) PESQ, GRU-T1 passed the
quality criterion. This supports a claim of non-inferiority within the stated
margin, not quality superiority over Mamba.

## 7. Evaluation metrics

Quality was evaluated with wideband PESQ at 16 kHz (Rix et al., 2001), SI-SDR
(Le Roux et al., 2019), STOI (Taal et al., 2011), and extended STOI (ESTOI;
Jensen & Taal, 2016). PESQ estimates perceptual speech quality. STOI and ESTOI
estimate intelligibility, with ESTOI intended to remain informative under a
wider range of temporal and spectral distortions. SI-SDR measures waveform
fidelity after allowing a scale projection of the clean reference.

For estimate \(\widehat{\mathbf{s}}\) and zero-mean reference
\(\mathbf{s}\), the implementation computed

\[
\mathbf{s}_{\mathrm{target}}
=
\frac{\langle\widehat{\mathbf{s}},\mathbf{s}\rangle}
{\|\mathbf{s}\|^2+\varepsilon}\mathbf{s},
\qquad
\mathrm{SI\text{-}SDR}
=10\log_{10}
\frac{\|\mathbf{s}_{\mathrm{target}}\|^2+\varepsilon}
{\|\widehat{\mathbf{s}}-\mathbf{s}_{\mathrm{target}}\|^2+\varepsilon}.
\tag{13}
\]

SI-SDR improvement was enhanced SI-SDR minus noisy-input SI-SDR for the same
utterance. All metrics were calculated on full aligned waveforms. PESQ used
wideband mode; STOI used the standard mode and ESTOI used the extended mode of
the same implementation.

## 8. Statistical analysis

Absolute final-model uncertainty was estimated with a hierarchical bootstrap
over training seed and utterance. Metric values were arranged as
seed-by-file matrices. For each of 20,000 bootstrap iterations, three seeds
were sampled with replacement and utterances were sampled with replacement
within each selected seed. The mean of this resample formed one bootstrap
estimate. The 2.5th and 97.5th percentiles gave the 95% confidence interval.
Absolute-score bootstrapping used seed 14204 in the final V14.2 replication
evaluation.

V14.2-minus-V13 comparisons used a paired hierarchical bootstrap. Differences
were first calculated between corresponding model seeds and utterances; seeds
and files were then hierarchically resampled from this paired difference
matrix. This preserved both common evaluation material and the correspondence
between each V14.2 model and its V13 initialisation. The paired bootstrap used
20,000 samples and seed 14205. Paired win rate was the proportion of
seed-utterance differences greater than zero. Across-seed standard deviations
used the sample definition with \(n-1\) degrees of freedom.

## 9. Real-time benchmark

Runtime was measured through the native streaming interface on an AMD Ryzen 7
7800X3D CPU with one PyTorch intra-operation and one inter-operation thread.
The benchmark supplied one 160-sample hop per call. It used 100 warm-up frames
(1 s), followed by 3,000 measured frames (30 s). Each call was timed with a
nanosecond-resolution monotonic performance counter.

The benchmark reported mean, median, p95, p99, and maximum frame time. Real-time
factor was the sum of processing time divided by the 30 s audio duration.
Both p95 and p99 had to remain below the 10 ms hop deadline. Persistent-state
memory was calculated from the number of stored tensor elements assuming
32-bit floating point. The final recorded environment used PyTorch
2.11.0+cu128; quality evaluation used CUDA, while the deployment timing claim
is based on the stated single-thread CPU protocol.

## 10. Reproducibility and methodological safeguards

The final protocols record configuration, checkpoint, metadata, source-file,
and result hashes. Final evaluation used raw model weights rather than an
exponential moving average. The V14.2 replications used a fixed epoch and did
not permit early stopping, test-set inspection, or metric-driven reselection.
The external DNS1 set was acquired and frozen before any final V14.2 checkpoint
was evaluated on it.

Three limitations bound the interpretation. First, the VoiceBank-DEMAND
standard test was historically reused during the broader project, so it
provides benchmark comparability rather than pristine selection evidence.
Second, three training seeds quantify some optimisation variability but do not
fully characterise the distribution over random initialisation and data
ordering. Third, DNS1 is an independently sourced synthetic paired condition,
not a complete test of real recordings, reverberation, unseen devices, or all
noise domains.

## 11. Repository evidence map

The principal reproducibility artefacts are:

- V13 final protocol:
  [`configs/v13/frozen_final_protocol.yaml`](../../configs/v13/frozen_final_protocol.yaml)
- V14.2 fixed-recipe replication:
  [`configs/v14/frozen_replication_protocol.yaml`](../../configs/v14/frozen_replication_protocol.yaml)
- V14.2 final evaluation:
  [`configs/v14/frozen_replication_evaluation.yaml`](../../configs/v14/frozen_replication_evaluation.yaml)
- V13 seed-1200 training configuration:
  [`configs/v12/generated/full/gru_matched_time1_seed1200.yaml`](../../configs/v12/generated/full/gru_matched_time1_seed1200.yaml)
- V14.2 selected training configuration:
  [`configs/v14/generated/distillation/mag_005.yaml`](../../configs/v14/generated/distillation/mag_005.yaml)
- Student model:
  [`src/cnvqg/models/predictive_noise_vq_mamba_v8.py`](../../src/cnvqg/models/predictive_noise_vq_mamba_v8.py)
- Native streaming implementation:
  [`src/cnvqg/models/v8_native_streaming.py`](../../src/cnvqg/models/v8_native_streaming.py)
- Supervised loss implementation:
  [`src/cnvqg/losses/losses.py`](../../src/cnvqg/losses/losses.py)
- Distillation implementation:
  [`src/cnvqg/training/distillation.py`](../../src/cnvqg/training/distillation.py)
- Metric implementation:
  [`src/cnvqg/metrics/speech_metrics.py`](../../src/cnvqg/metrics/speech_metrics.py)
- Final Results chapter draft:
  [`docs/thesis/results_chapter_draft.md`](results_chapter_draft.md)

## References

Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F.,
Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN
encoder-decoder for statistical machine translation. In *Proceedings of the
2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*
(pp. 1724–1734). Association for Computational Linguistics.
https://doi.org/10.3115/v1/D14-1179

Jensen, J., & Taal, C. H. (2016). An algorithm for predicting the
intelligibility of speech masked by modulated noise maskers. *IEEE/ACM
Transactions on Audio, Speech, and Language Processing, 24*(11), 2009–2022.
https://doi.org/10.1109/TASLP.2016.2585871

Le Roux, J., Wisdom, S., Erdogan, H., & Hershey, J. R. (2019). SDR—Half-baked
or well done? In *ICASSP 2019—2019 IEEE International Conference on Acoustics,
Speech and Signal Processing (ICASSP)* (pp. 626–630). IEEE.
https://doi.org/10.1109/ICASSP.2019.8683855

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. In
*International Conference on Learning Representations*.
https://openreview.net/forum?id=Bkg6RiCqY7

Reddy, C. K. A., Gopal, V., Cutler, R., Beyrami, E., Cheng, R., Dubey, H.,
Matusevych, S., Aichner, R., Aazami, A., Braun, S., Rana, P., Srinivasan, S.,
& Gehrke, J. (2020). The INTERSPEECH 2020 deep noise suppression challenge:
Datasets, subjective testing framework, and challenge results. In *Interspeech
2020* (pp. 2492–2496). International Speech Communication Association.
https://doi.org/10.21437/Interspeech.2020-3038

Rix, A. W., Beerends, J. G., Hollier, M. P., & Hekstra, A. P. (2001).
Perceptual evaluation of speech quality (PESQ)—A new method for speech quality
assessment of telephone networks and codecs. In *2001 IEEE International
Conference on Acoustics, Speech, and Signal Processing. Proceedings* (Vol. 2,
pp. 749–752). IEEE. https://doi.org/10.1109/ICASSP.2001.941023

Taal, C. H., Hendriks, R. C., Heusdens, R., & Jensen, J. (2011). An algorithm
for intelligibility prediction of time-frequency weighted noisy speech. *IEEE
Transactions on Audio, Speech, and Language Processing, 19*(7), 2125–2136.
https://doi.org/10.1109/TASL.2011.2114881

Valentini-Botinhao, C. (2017). *Noisy speech database for training speech
enhancement algorithms and TTS models* [Data set]. University of Edinburgh.
https://doi.org/10.7488/ds/2117
