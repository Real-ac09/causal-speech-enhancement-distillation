# Streaming Hybrid CN-VQG v2 Design

## 1. Objective

Design a causal, stateful speech-enhancement family that preserves the defining
parts of CN-VQG:

- a waveform enhancement path;
- Mamba temporal modelling;
- a vector-quantised noise-state codebook;
- a lightweight time-frequency refinement path;
- bounded residual corrections.

The family has one large teacher and two deployment students. All variants use
the same causal information boundary and the same streaming API so that teacher
targets remain attainable by the students.

This design is additive. Existing waveform, complex-STFT, and hybrid models stay
available for reproducing earlier experiments.

## 2. Model family

| Variant | Target parameters | Primary use |
|---|---:|---|
| `streaming_hybrid_v2_teacher` | 24--32M | Distillation teacher and quality ceiling |
| `streaming_hybrid_v2_student` | 4.5--6.5M | Main desktop/mobile deployment model |
| `streaming_hybrid_v2_tiny` | 1.4--2.8M | Highly constrained deployment target |

Parameter counts are budgets rather than names. CI must calculate the actual
count and reject configurations outside their declared budget.

The initial implemented presets contain 24,946,123, 4,634,923, and 1,513,923
parameters respectively when Mamba is enabled.

### 2.1 Initial dimensions

| Parameter | Teacher | Student | Tiny |
|---|---:|---:|---:|
| Waveform encoder base channels | 256 | 128 | 80 |
| Shared latent dimension | 512 | 256 | 160 |
| Speech latent dimension | 512 | 256 | 160 |
| Noise latent dimension | 256 | 128 | 80 |
| Waveform Mamba layers | 10 | 6 | 4 |
| VQ codebook size | 256 | 128 | 64 |
| TF hidden dimension | 64 | 32 | 24 |
| TF Mamba layers | 4 | 2 | 2 |
| Frequency subbands | 16 | 8 | 8 |

Codebook size may be increased only when code utilization, perplexity, and
dead-code measurements show that the current codebook is capacity-limited.

## 3. Signal path

```text
waveform chunk + state
    |
    v
causal strided waveform encoder
    |-------------------------------|
    v                               v
speech projection              noise projection
    |                               |
causal waveform Mamba          temporal pooling
    |                               |
    |                         EMA vector quantizer
    |                               |
    |<---------- FiLM/gating -------|
    v
causal waveform residual decoder
    |
    v
base enhanced waveform
    |
    v
causal STFT analysis (center=False)
    |
    v
subband TF encoder -> shared causal Mamba -> cross-band mixer
    |
    v
bounded gain and phase residual
    |
    v
streaming overlap-add synthesis
    |
    v
enhanced chunk + updated state
```

The TF path refines the waveform estimate; it does not replace it. Its final
projection is zero-initialized, producing an identity correction at the start
of training.

## 4. Causality and latency contract

### 4.1 Required causal behavior

- Waveform convolutions use explicit left padding only.
- Transposed convolutions must not emit samples that depend on future input.
- STFT and all spectral losses use `center=False` when testing streaming parity.
- TF convolutions use symmetric frequency padding and left-only time padding.
- Mamba blocks expose and consume inference state; they never recompute an
  unbounded utterance history during streaming inference.
- VQ state updates use only current and previous frames.
- Offline inference is implemented by repeatedly calling the streaming path.

### 4.2 Initial framing

| Setting | Value |
|---|---:|
| Sample rate | 16,000 Hz |
| STFT FFT size | 512 samples |
| Window length | 512 samples / 32 ms |
| Hop length | 128 samples / 8 ms |
| External input chunk | 128 or 256 samples |
| Target algorithmic latency | at most 48 ms |

The exact latency must be derived from receptive fields and verified with an
impulse test. It must not be inferred from hop length alone.

### 4.3 Streaming state

`StreamingHybridV2State` contains, at minimum:

- waveform encoder convolution caches;
- waveform Mamba states for every layer;
- decoder/upsampling caches;
- analysis sample buffer;
- TF causal-convolution caches;
- subband Mamba states;
- synthesis overlap-add buffer;
- most recent noise code and optional persistence counters.

State is explicit, batch-aware, device-aware, and serializable. No module may
hide mutable inference state internally.

## 5. Waveform branch

The waveform branch retains the current residual-enhancement concept while
removing symmetric padding.

### 5.1 Encoder

- Four stride-2 causal convolution stages; total downsampling factor 16.
- Normalization must not aggregate future time positions.
- PReLU or SiLU activations are both permitted and must be ablated once on the
  student, not independently at every size.

### 5.2 Speech path

- Project the shared encoder representation to `speech_dim`.
- Apply pre-normalized residual Mamba blocks.
- Retain small layer scales for stable initialization.
- Expose early, middle, and final features for optional distillation.

### 5.3 Noise path and separation

Speech and noise latents originate from the shared encoder, but the noise path
must have an explicit learning target. It predicts a clean/noisy spectral-noise
proxy in addition to feeding the codebook.

Recommended auxiliary objectives:

- noise magnitude reconstruction from `abs(STFT(noisy) - STFT(clean))` or a
  stable magnitude-domain approximation;
- speech-latent consistency across different noise mixtures of the same clean
  utterance when such paired augmentation is available;
- low-weight cross-covariance penalty between pooled speech and noise latents.

These objectives make the labels "speech" and "noise" testable rather than
depending only on projection names.

### 5.4 Residual decoder

The decoder predicts a bounded waveform residual. The residual scale is fixed
during initial waveform training or parameterized with a nonzero lower bound so
that it cannot collapse to an exact identity mapping.

## 6. Noise-state VQ

### 6.1 Quantizer

Use an EMA-updated codebook with:

- straight-through gradients to the encoder;
- EMA cluster counts and code vectors;
- Laplace-smoothed cluster sizes;
- dead-code replacement from recent encoder samples;
- commitment loss;
- optional low-weight entropy/usage regularization.

The codebook is updated by EMA, not by the main optimizer. Its embedding must
therefore be excluded from AdamW and Muon parameter groups.

### 6.2 Noise update rate

Noise codes operate slower than the waveform latent. Pool causal latent frames
to an initial update interval of 32 ms. Hold or interpolate the quantized state
between updates. Compare 32, 64, and 96 ms only after the 32 ms baseline works.

An optional persistence penalty discourages rapid code switching:

```text
mean(1 - cosine(q[t], q[t-1]))
```

It must be kept weak enough to track genuinely changing noises.

### 6.3 Required diagnostics

Every training and validation epoch records:

- perplexity;
- active-code fraction;
- dead-code fraction;
- minimum, median, and maximum assignment count;
- code-switch rate per second;
- commitment loss;
- noise-proxy reconstruction loss.

## 7. TF refiner

### 7.1 Inputs

The refiner consumes causal features derived from:

- noisy log magnitude;
- base-enhanced log magnitude;
- base-minus-noisy log-magnitude difference;
- base phase represented by sine and cosine;
- projected VQ noise state broadcast over frequency;
- optional projected waveform speech context.

Conditioning on waveform states lets the TF stage reuse information instead of
behaving as a second independent enhancer.

### 7.2 Subband temporal modelling

Divide the 257 FFT bins into contiguous subbands. A small 2D encoder produces
features per band. The same Mamba stack is applied over time to each band by
folding the band dimension into the batch dimension. Shared weights control the
parameter and memory cost.

A pointwise or grouped cross-band mixer follows the temporal stack. No operation
may mix future time frames.

### 7.3 Outputs

The refiner predicts:

```text
gain = exp(gain_scale * tanh(gain_logits))
phase_delta = phase_scale * tanh(phase_logits)
```

Initial bounds:

- `gain_scale = 0.20`;
- `phase_scale = 0.08` radians.

Phase loss is weighted by target magnitude so that near-silent bins do not
dominate training.

## 8. Public interfaces

### 8.1 Training output

```python
@dataclass
class StreamingHybridV2Output:
    enhanced: Tensor
    base_enhanced: Tensor
    waveform_residual: Tensor
    gain: Tensor
    phase_delta: Tensor
    speech_features: tuple[Tensor, ...]
    noise_latent: Tensor
    quantized_noise: Tensor
    noise_prediction: Tensor
    code_indices: Tensor
    vq_metrics: VQMetrics
```

Heavy intermediate fields may be disabled for normal inference but must be
available for distillation without hooks.

### 8.2 Streaming inference

```python
state = model.init_streaming_state(batch_size, device, dtype)
output, state = model.stream_step(audio_chunk, state)
tail, state = model.flush_stream(state)
```

`stream_step` accepts only a documented set of chunk sizes. `flush_stream`
emits delayed samples and resets no state implicitly.

### 8.3 Offline inference

```python
enhanced = model.enhance_offline(waveform, chunk_samples=256)
```

This method is a convenience loop over the streaming API. A separate full-
utterance implementation may exist only as a test oracle, not as the release
inference path.

## 9. Distillation contract

The teacher is frozen and receives the same causal input chunks and initial
state as the student.

Distillation targets include:

1. enhanced waveform and multi-resolution spectrum;
2. TF gain and magnitude-weighted phase correction;
3. early, middle, and final speech features through training-only projections;
4. projected quantized-noise embeddings and temporal similarity matrices;
5. noise-proxy prediction.

Student code IDs are never matched directly to teacher code IDs because their
codebooks differ in size and permutation.

Clean supervision remains dominant. Initial aggregate weighting:

| Term | Weight |
|---|---:|
| Clean waveform | 1.00 |
| Clean MR-STFT | 1.00 |
| Clean SI-SDR | 0.30 |
| Teacher waveform | 0.30 |
| Teacher spectrum | 0.30 |
| Gain | 0.10 |
| Magnitude-weighted phase | 0.05 |
| Speech features | 0.10 |
| Noise features | 0.10 |
| VQ commitment | 0.10 |
| Noise prediction | 0.05--0.10 |

Distillation weights warm up over the first 5--10% of optimizer steps. A final
short clean-target-only fine-tune is required to avoid preserving teacher
artifacts.

## 10. Optimizer design

AdamW is the reference optimizer. Muon is an optional controlled ablation, not
a dependency of the architecture.

For a Muon run:

- native 2D hidden Mamba/linear weights may use Muon;
- convolution kernels use Muon only through a tested convolution-aware adapter;
- biases, normalization parameters, layer scales, residual scales, first input
  layer, output heads, and distillation projections use AdamW;
- EMA VQ codebooks are excluded from both optimizers.

Muon and AdamW experiments use matched examples, schedules, seeds, and stopping
criteria. Optimizer selection is based on validation quality and time to target,
not training loss alone.

## 11. Training plan

### 11.1 Teacher

1. Train causal waveform branch and EMA-VQ from scratch for 40--60 epochs.
2. Freeze waveform branch and train TF refiner for 15--20 epochs.
3. Jointly fine-tune with a 5--10x smaller waveform learning rate for 5--10
   epochs.
4. Freeze and register the teacher; do not select it using the final holdout.

### 11.2 Main student

1. Train the waveform student with clean and teacher supervision for 30--50
   epochs.
2. Train the TF student for 10--15 epochs.
3. Jointly fine-tune for 5--10 epochs.
4. Fine-tune against clean targets only for 2--5 epochs.

### 11.3 Tiny student

Train only after confirming that distillation improves the main student over a
same-size non-distilled baseline. Direct teacher-to-tiny distillation is the
default; progressive teacher-to-student-to-tiny distillation is a fallback.

## 12. Verification and release gates

### 12.1 Functional tests

- Output length is exact for every supported chunk size and arbitrary utterance
  lengths.
- Chunked and offline-wrapper outputs match within a stated numerical tolerance.
- An input prefix produces identical output regardless of future suffix.
- Batched streams do not share or leak state.
- Reset and flush behavior is deterministic.
- Serialization and checkpoint resume preserve optimizer, scheduler, EMA, and
  VQ state.

### 12.2 Model tests

- All variants fall inside parameter budgets.
- No requested Mamba path silently falls back to convolution.
- Codebook utilization diagnostics are finite and populated.
- Dead codes are replaced without introducing NaNs.
- The residual and TF correction heads start at their documented identity
  behavior.

### 12.3 Deployment gates

For the main student on the selected target CPU:

- algorithmic latency at most 48 ms;
- real-time factor below 0.5 for one stream;
- no unbounded state growth;
- report peak resident memory and per-chunk p50/p95/p99 latency;
- report quality from the actual streaming implementation;
- evaluate on an untouched external holdout.

## 13. Implementation order

1. Add causal convolution, causal STFT/overlap-add, and streaming-state
   primitives with unit tests.
2. Implement EMA noise VQ and its diagnostics independently.
3. Implement the waveform-only v2 model and streaming parity tests.
4. Train a small smoke model to validate learning and codebook activity.
5. Implement the subband TF refiner and shared conditioning.
6. Add teacher/student outputs and distillation losses.
7. Add teacher, student, and tiny configs with enforced parameter budgets.
8. Benchmark the untrained and smoke-trained student inference paths before
   committing to full teacher training.
9. Train teacher, non-distilled student baseline, and distilled student.
10. Run the Muon ablation only after the AdamW recipe is stable.

## 14. Decisions intentionally deferred

- Whether phase prediction beats magnitude-only refinement under causal
  constraints.
- Whether 32, 64, or 96 ms is the best VQ update interval.
- Whether shared subband Mamba is sufficient or needs limited band-specific
  adapters.
- Whether Muon helps this model family.
- Final quantization format and runtime backend.

These are experiments, not assumptions embedded in the first implementation.
