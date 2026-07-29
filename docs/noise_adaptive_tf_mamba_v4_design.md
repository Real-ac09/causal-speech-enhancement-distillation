# Noise-State Adaptive Recurrent TF-Mamba (V4)

V4 is a TF-primary magnitude/phase model. A single dual-axis Mamba cell is
reused for progressive refinement, avoiding the parameter growth of stacked
blocks. The baseline deliberately disables discrete noise adaptation and
adaptive depth so that each proposed contribution can be measured.

## Main path

1. Power-compressed magnitude, phase cosine, and phase sine enter a compact
   frequency-downsampling encoder.
2. One shared cell performs local convolution, time Mamba, and frequency
   Mamba refinement up to four times.
3. A magnitude head predicts a mask in `[0, 2]`.
4. A phase head predicts a wrapped residual and a confidence head controls how
   strongly it replaces noisy phase.
5. Complex iSTFT reconstructs the waveform.

## Noise-adaptive path

An EMA codebook quantizes a global noise state. The selected state conditions
the recurrent cell's scale and shift. An ACT-style halting head assigns a
probability to every refinement depth, producing a differentiable expected
iteration count. Training executes all iterations; optimized single-stream
inference can stop once cumulative halting probability reaches a threshold.

## Required ablations

- TF baseline: no VQ, no conditioning, fixed four iterations.
- Shared recurrence at one, two, and four iterations.
- Continuous conditioning without quantization.
- VQ without conditioned dynamics.
- VQ with conditioned dynamics and fixed depth.
- Full adaptive-depth system.
- Phase correction without and with confidence.

The baseline must approach competitive PESQ before the adaptive mechanisms are
treated as useful. Novelty is claimed from noise states controlling recurrent
dynamics and compute, not from using Mamba, VQ, or magnitude/phase decoding in
isolation.
