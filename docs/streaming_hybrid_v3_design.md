# Streaming Hybrid V3

## Goals

V3 targets higher PESQ/STOI/ESTOI than v2 while remaining small enough for
practical low-latency inference. It deliberately does not assume that a larger
parameter count improves enhancement.

## Architecture decisions

- Restore the original model's learned analysis/synthesis shape with bounded
  lookahead. Strict-causal configurations remain available through
  `encoder_type: causal` and `decoder_type: upsample`.
- Use EMA VQ for stable noise tokens, but project the token from 64--128
  channels to 8--12 channels before broadcasting it over the TF grid.
- Keep the TF correction identity-initialised so joint refinement cannot
  destroy a pretrained waveform model at initialization.
- Predict bounded magnitude gain and a small phase residual. Large unconstrained
  complex masks were rejected because they can introduce audible artifacts.
- Provide approximately 1M, 2.5M, and 8.6M variants. Scale only after the
  teacher beats the original 1M/hybrid baselines.

## Metric strategy

- SI-SDR: direct SI-SDR loss and SI-SDR-selected checkpoints.
- PESQ: log-mel and multi-resolution spectral reconstruction, followed by a
  very low-weight MetricGAN-lite stage. PESQ itself is non-differentiable and
  should be measured on fixed validation subsets, not optimized directly.
- STOI/ESTOI: preserve speech envelopes with waveform L1, SI-SDR, mel loss,
  moderate enhancement gains, and conservative phase correction.
- Artifact control: identity initialization, bounded gain/phase, complex STFT
  consistency, and evaluation of per-noise/per-SNR failures.

## Training sequence

1. Train the teacher waveform path in BF16 with hybrid Muon/AdamW.
2. Initialize the TF refiner from the waveform checkpoint and fine-tune in
   BF16 with a low learning rate.
3. Evaluate full validation SI-SDR, PESQ, STOI, and ESTOI.
4. Only if PESQ lags, run the short FP32 MetricGAN-lite quality stage.
5. Distil the best teacher into student and tiny models and fine-tune each on
   clean targets. Do not accept a distilled model based on distillation loss
   alone.

## Muon policy

Muon is applied to eligible hidden 2-D matrices. AdamW remains responsible for
biases, normalization gains, input/output heads, convolution tensors unsupported
by `torch.optim.Muon`, VQ-related auxiliary parameters, and perceptual fine-tunes.
Muon and AdamW learning rates must be tuned separately.
