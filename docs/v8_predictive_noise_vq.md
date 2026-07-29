# V8 predictive noise-prototype Mamba

V8 tests one focused dissertation hypothesis: auxiliary prediction of
quantized **noise transitions** can improve a continuous, causal Mamba speech
enhancer without placing VQ in the speech reconstruction path.

## Architecture contract

- 16 kHz, uncentred 20 ms analysis window and 10 ms hop.
- Exactly one frequency-downsampling stage.
- Full-resolution causal detail skip.
- Mamba operates only along time at the bottleneck.
- Joint complex speech/noise masks are projected so their sum is exactly one.
- The enhancement path uses continuous features only.
- A 32-entry EMA codebook quantizes the causal noise state for auxiliary noise
  reconstruction and 10/20/40 ms prototype prediction.
- The model starts as identity enhancement and zero estimated noise.

## Claim boundary

The novelty claim is not “Mamba plus VQ”. Existing systems already combine
Mamba with U-Net, time/frequency processing, and quantized speech features.
The investigated contribution is the combination of (a) mixture-consistent
complex noise-residual estimation, (b) non-bottleneck noise prototypes, and
(c) multi-horizon prediction of upcoming noise prototypes from a causal
continuous Mamba state.

The claim must remain conditional until a full systematic literature review.

## Required ablation ladder

1. Continuous causal control (`auxiliary_vq: false`).
2. Auxiliary noise VQ without prototype prediction.
3. Auxiliary VQ plus prototype prediction.
4. GRU/causal-convolution versus Mamba at matched size.
5. Learned EMA prototypes versus fixed/random assignments.
6. Optional bounded prototype adapter only after a no-harm gate.

Control and predictive configurations intentionally use AdamW first. Muon,
perceptual fine-tuning, and distillation are deferred until the architectural
hypothesis is resolved.
