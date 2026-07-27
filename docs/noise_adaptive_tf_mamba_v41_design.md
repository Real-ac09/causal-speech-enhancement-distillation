# Noise-adaptive TF-Mamba V4.1

V4.1 targets the bottlenecks identified by the V4 ablations. V4 established a
strong sub-million-parameter TF-Mamba baseline, but its bilinear frequency
restoration limited spectral detail and its global 128-entry VQ codebook had no
measurable test benefit.

## Architecture

- A two-level frequency encoder retains full- and half-resolution skip features.
- Transposed convolutions learn frequency restoration, with odd-bin alignment
  handled only at skip boundaries.
- Magnitude and phase use dedicated refinement branches.
- A zero-initialized magnitude-to-phase adapter allows controlled branch
  interaction without disturbing initial phase training.
- The shared recurrent cell still applies temporal and frequency Mamba using one
  parameter set across refinement iterations.
- Noise states are extracted per 32 latent frames rather than once per utterance.
- A 16-entry EMA codebook is the default for the independent VQ stage.
- Segment codes produce time-varying scale and shift for recurrent dynamics.

The XL preset has approximately 1.147M trainable parameters with real Mamba
(approximately 1.105M with the convolutional fallback used by CPU tests).

## Phase objectives

The phase branch exposes an ungated candidate phase as well as the synthesized
phase. Confidence is trained to predict whether the candidate phase is close to
the clean phase. Speech-weighted group-delay and instantaneous-frequency losses
supplement the direct circular phase loss.

## Training sequence

1. Train the fixed-two, no-VQ skip/dual-decoder baseline using partial transfer
   of compatible V4 Mamba weights.
2. Select checkpoints using PESQ on a deterministic 100-item validation subset.
3. Only after the decoder baseline is established, enable the 16-entry
   segmentwise VQ, code conditioning, and adaptive depth.
4. Confirm every checkpoint-level result with an equal-budget retrained control.
