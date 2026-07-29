# V9 causal prototype-conditioned dual-axis Mamba

## Purpose

V9 is a causal-native replacement for the V4.10 retrofit and the
magnitude-limited V8 line. It is designed around one dissertation question:

> Can auxiliary discrete noise prototypes improve a continuous, lightweight,
> causal Mamba speech enhancer without becoming an information bottleneck?

The encoder-decoder and complex reconstruction path are deliberately treated
as established engineering foundations. The proposed contribution is the
multi-timescale continuous noise state, its bounded conditioning of temporal
Mamba, and auxiliary prototype regularisation with a formal no-harm gate.

## Evidence incorporated

V9 must address the failures already measured in this repository:

- V4.3 direct VQ conditioning lost up to 0.043 PESQ; quantized features must
  not be required by the enhancement path.
- Converting V4.3's centred STFT, symmetric time convolutions, and utterance
  normalization after training caused a catastrophic zero-shot regression.
  V9 must be trained causally from its first optimizer step.
- V8 showed that mask range was not the limiting oracle, but its learned
  magnitude estimate remained weak even after long capacity tests. V9 must
  add full-band frequency interaction, a stronger multiscale decoder, and a
  direct reconstruction target without the joint speech/noise projection.
- V4.10 phase, group-delay, and instantaneous-frequency losses remained almost
  constant while magnitude and SI-SDR improved. Phase refinement is therefore
  optional and cannot sit on the critical path of the first foundation run.
- V4.10 recovery plateaued despite improving SI-SDR. More epochs on a causal
  retrofit are not a substitute for causal-native features and normalization.

## Deployment contract

- Sample rate: 16 kHz.
- Analysis window: 320 samples (20 ms), periodic Hann.
- Hop: 160 samples (10 ms).
- FFT: 512 samples with zero padding.
- Centring: disabled.
- Algorithmic latency: exactly 320 samples / 20 ms.
- Arbitrary input chunk sizes and final partial frames are supported.
- Whole-utterance causal and streaming outputs agree within `1e-4`.
- Student parameter cap: 1.10M.
- Teacher parameter cap: 2.70M.
- Deployment VQ mode defaults to `train_only`.

## Signal representation

For noisy spectrum `X`, define the power-compressed complex spectrum

`X_p = |X|^0.3 exp(j angle(X))`.

The model input has three channels:

1. `real(X_p)`;
2. `imag(X_p)`;
3. `log1p(|X|)`.

This retains amplitude-dependent reliability in the phase-bearing channels.
It replaces V8/V4's unit sine/cosine features, which give random phase in
near-silent bins the same apparent scale as energetic speech bins.

## Causal-native multiscale encoder

All encoder and decoder convolutions have temporal kernel size one. Temporal
memory belongs exclusively to Mamba.

The encoder has two frequency-only reductions and preserves both skips:

```text
3 input channels
  -> full-resolution detail stem       [F = 257]
  -> frequency stride 2 + projection   [F = 129]
  -> frequency stride 2 + projection   [F = 65]
```

Each stage uses a dense `1x1` channel projection plus a depthwise frequency
convolution. There is no time padding, time cache, or transposed time
convolution. The decoder uses nearest frequency upsampling followed by
frequency-only refinement and explicit full/half-resolution skip fusion.

Normalization is `FrameChannelRMSNorm`: RMS statistics are computed over the
channel dimension independently for every time-frequency bin. It never pools
over time, does not depend on batch statistics, and has fixed behaviour at the
start and end of an utterance.

## Untied dual-axis core

V9 uses independent blocks rather than V4's tied recurrent refinement:

```text
pre-norm
  -> bidirectional frequency Mamba within the current frame
  -> residual
pre-norm
  -> unidirectional temporal Mamba
  -> bounded continuous-noise adapter
  -> residual
```

Frequency processing may inspect all 65 bins of the current frame without
violating causality. The same frequency Mamba parameters process low-to-high
and flipped high-to-low sequences; their outputs are concatenated and fused
with a learned projection. Blocks are untied across depth.

Temporal Mamba is strictly unidirectional and exposes recurrent cache tensors.
Mamba settings initially remain `d_state=16`, `d_conv=4`, and `expand=2`.

Initial depth and widths:

| Variant | Detail | Half | Core | Noise | Blocks |
|---|---:|---:|---:|---:|---:|
| Student | 40 | 80 | 152 | 64 | 2 |
| Teacher | 64 | 128 | 208 | 96 | 3 |

Widths are searched in increments of eight at implementation time. The
largest configuration below each parameter cap is selected; depth is not
silently reduced to satisfy a cap.

A structural parameter estimate gives approximately 0.94M for the student and
2.47M for the teacher before optional phase/VQ adapters. The remaining margin
is intentional. Exact counts are enforced against instantiated models rather
than relying on this estimate.

## Multi-timescale continuous noise state

The noise state is derived from eight causal pooled frequency bands of the
half-resolution encoder feature. It has two paths:

- fast state: rolling mean over the current and previous three frames (40 ms);
- slow state: causal EMA with an initial time constant of 32 frames (320 ms).

The concatenated states are projected to `noise_dim` and normalized per frame.
The fast path captures transitions; the slow path captures the background
noise identity. Neither path consumes clean targets or future frames.

Each temporal block receives the continuous state through a zero-initialized
low-rank scale/shift adapter. Its residual contribution is capped at 10%.
The unconditioned backbone is therefore exactly represented at initialization.

## Direct complex reconstruction

The decoder predicts one direct compressed-domain complex-ratio mask:

```text
M_r = 1 + a * tanh(r)
M_i =     a * tanh(i)
S_p = (M_r + j M_i) * X_p
```

The head is zero-initialized, giving exact identity reconstruction. There is
no joint speech/noise allocation and no mixture-projection branch. The clean
magnitude is recovered by power decompression of `S_p`; phase is its angle.

An optional phase-detail head predicts a further wrapped residual capped at
0.25 radians. It is zero-initialized and energy-gated so near-silent bins
cannot dominate. It is disabled in the first foundation control and retained
only if a paired reconstruction ablation improves PESQ without harming SI-SDR,
STOI, or ESTOI.

## Auxiliary prototype VQ

The continuous noise state always drives enhancement. VQ cannot replace it.

- 32 EMA-updated codes.
- Commitment weight: 0.02.
- Usage KL/entropy weight: 0.005.
- Eight-band noise-spectrum reconstruction weight: 0.05.
- Optional transition prediction at 10, 20, and 40 ms is a separate ablation,
  not part of the first VQ experiment.

Modes:

- `off`: no VQ objectives;
- `train_only`: prototypes supervise the noise state but cannot affect speech;
- `bounded_adapter`: prototype embedding enters a zero-initialized rank-eight
  adapter capped at 5%, with 50% branch dropout.

`bounded_adapter` is retained only for a locked-validation improvement of at
least 0.01 PESQ whose paired bootstrap 95% interval excludes zero. VQ-off
enhancement must be bitwise independent of codebook contents.

## Foundation losses

The first control intentionally uses a small objective set:

| Objective | Weight |
|---|---:|
| Compressed complex L1 | 1.00 |
| Log-magnitude L1 | 0.50 |
| Waveform Charbonnier | 0.10 |
| SI-SDR | 0.10 |
| STFT consistency | 0.10 |

Wrapped phase, group-delay, IF, mel, MetricGAN, PCS, and VQ objectives are off
in the first control. A gradient-cosine audit is required before expanding the
loss set.

After the foundation is competitive, the perceptual stage may add a calibrated
PESQ regressor with a generator weight ramped from zero to 0.02. Foundation
losses remain at 50% strength and metric safeguard limits remain mandatory.

## Training programme

All architecture selection uses the student first to reduce turnaround time.

1. Shape/causality/finite-gradient smoke test.
2. Sixteen-example overfit test; require a large mask-oracle gap reduction.
3. Non-VQ student foundation control from scratch.
4. Add bidirectional frequency Mamba only if its matched control improves.
5. Add continuous noise conditioning only if its matched control improves.
6. Add `train_only` VQ and apply the no-harm gate.
7. Train the teacher only after the student architecture is frozen.
8. Distil the final student from the teacher.

Foundation optimization:

- BF16 on RTX 3090.
- Effective batch size 48 via gradient accumulation.
- AdamW control first: LR 3e-4, weight decay 1e-5.
- 2,000-step linear warm-up followed by cosine decay.
- Maximum 60 epochs; early-stop patience 8, minimum PESQ delta 0.003.
- Muon is introduced only as a matched optimizer ablation after the
  architecture passes the performance gate.

The headline model uses only the standard VoiceBank+DEMAND training split.
An external-data robustness model is reported separately.

## Ablation ladder and claim boundary

The cumulative ladder is fixed before test-set evaluation:

| ID | Backbone | Frequency Mamba | Continuous noise | Auxiliary VQ |
|---|---|---|---|---|
| A | causal complex U-Net + temporal Mamba | off | off | off |
| B | A | on | off | off |
| C | B | on | on | off |
| D | C | on | on | `train_only` |
| E | C | on | on | `bounded_adapter` |

The dissertation claim is supported if D improves representation/noise
prediction without harming enhancement, or if E passes the statistical
no-harm gate. If neither occurs, the negative result remains reportable, but
the deployment model is C.

## Interfaces

Configuration:

```yaml
architecture: causal_prototype_dual_axis_mamba_v9
variant: student | teacher
vq_mode: off | train_only | bounded_adapter
phase_detail: false
```

Streaming API:

```python
state = model.init_stream_state(batch_size, device, dtype)
output, state = model.forward_chunk(audio_chunk, state)
tail, state = model.flush(state)
```

The output contains enhanced audio, compressed complex mask, estimated
magnitude and phase, fast/slow/combined noise states, code indices, code
perplexity, VQ adapter strength, and intermediate features required for
distillation.

## Initial acceptance gates

Before any full teacher run:

- future input cannot affect finalized output;
- arbitrary chunks match whole-file causal inference within `1e-4`;
- BF16 forward/backward gradients are finite;
- student and teacher parameter caps are enforced;
- identity initialization reconstructs the causal frontend within `1e-4`;
- non-VQ student exceeds V4.10 on the same locked validation protocol;
- full 400-file validation occurs only for new 100-file leaders;
- the official test set remains untouched until architecture selection freezes.

V9 is not claimed as “a new U-Net” or “Mamba plus VQ.” Its claim is the
bounded, causal integration and empirical evaluation of continuous
multi-timescale noise conditioning with auxiliary discrete prototypes.
