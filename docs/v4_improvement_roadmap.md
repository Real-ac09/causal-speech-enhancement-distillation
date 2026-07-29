# V4 performance improvement roadmap

## Current reference

The current reference is the 956,760-parameter adaptive V4 checkpoint trained in
`noise_adaptive_tf_mamba_v4_noise_adaptive`. Its held-out VoiceBank+DEMAND score
is PESQ 2.7818, SI-SDR 18.8600 dB, STOI 0.9401, and ESTOI 0.8523.

## Measurement before further training

1. Run checkpoint-level inference ablations for VQ, code conditioning, adaptive
   weighting, and fixed depths one, two, and four.
2. Add calibrated hard-exit inference and measure quality, latency, real-time
   factor, and memory at several thresholds.
3. Stratify results by input SNR, noise type, speaker, utterance duration, and
   baseline difficulty.
4. Select checkpoints using a fixed perceptual validation subset rather than
   composite loss alone.

Checkpoint-level toggles measure dependency and deployment trade-offs. Any
promising result must subsequently be confirmed with a separately trained
ablation to support a causal claim.

## Architecture changes

1. Replace bilinear frequency restoration with learned upsampling and
   full-resolution encoder skip connections.
2. Split the output head into dedicated magnitude and phase decoders with
   controlled cross-branch interaction.
3. Test one frequency downsampling stage versus the current two stages to retain
   harmonic and consonant detail.
4. Preserve the shared recurrent dual-axis Mamba cell and noise-controlled
   dynamics as the central architecture.

## Phase and perceptual objectives

1. Replace the current confidence target with one derived from local SNR or
   observed phase error.
2. Add speech-weighted phase, group-delay, and complex-consistency losses.
3. Add a short low-learning-rate PESQ-proxy or MetricGAN-style fine-tuning stage
   with SI-SDR and distortion safeguards.
4. Track PESQ, STOI, ESTOI, and SI-SDR on a fixed validation subset each epoch.

## Noise-state codebook

1. Compare codebook sizes 8, 16, and 32; 128 is currently underused.
2. Add usage balancing or entropy regularization and report assignment
   perplexity rather than relying on dead-code replacement statistics.
3. Compare random/EMA initialization with reservoir or k-means initialization.
4. Compare one global utterance code with causal segmentwise noise states.
5. Compare mixture-derived states with states derived from the estimated noise
   residual.

## Adaptive computation

1. Train useful multi-pass refinement before ramping the compute penalty.
2. Calibrate hard stopping against the current soft four-pass mixture.
3. Report average and tail iteration counts alongside actual wall-clock latency.
4. Explore stochastic hard exits or straight-through halting only after the
   soft controller has been characterized.

## Scaling, distillation, and optimization

1. Fix the spectral decoder and codebook before increasing model size.
2. Train a 2–3M parameter teacher using the improved architecture.
3. Distil magnitude, phase, intermediate TF features, noise states, and final
   waveform output into a roughly 1M deployment model.
4. Compare AdamW against hybrid Muon from a common initialization. Use Muon only
   for suitable matrix parameters; retain AdamW for normalization, biases,
   scalar gates, codebook state, and incompatible Mamba parameters.

## Data and robustness

1. Add SNR-balanced sampling and a difficult-example curriculum.
2. Add unseen noise, reverberation, bandwidth, clipping, and level variation.
3. Keep the final external holdout locked until architecture and checkpoint
   selection are frozen.
