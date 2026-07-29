# V7 modular causal enhancement design

V7 adds structure only where V6 measurements indicate missing capacity. Its
enhancement path remains a single feed-forward chain:

```text
causal multiscale encoder
  -> frame-local full-band frequency block
  -> one causal temporal Mamba block
  -> multiscale complex decoder
  -> optional bounded phase detail
```

The full-band block is bidirectional only over frequency bins belonging to the
current frame, so it adds no future-time context. Mamba is used once for
sub-band temporal memory instead of defining every operation in the model.

The complex mask is identity-initialised. Adaptive phase refinement is a
separate zero-initialised bounded residual with a maximum gate of 0.85. The
high-resolution encoder skip preserves consonant and harmonic detail.

VQ is disabled in the initial search. When enabled later, its state is derived
from bottleneck features but cannot modify the enhancement path.

The controlled search tests four cumulative configurations:

1. Multiscale encoder/decoder only.
2. Multiscale plus full-band frequency modelling.
3. Multiscale plus adaptive phase refinement.
4. Both additions.

Every candidate has the same seed, data, optimizer and training budget. An
addition is retained only if it improves PESQ without violating SI-SDR, STOI
or ESTOI no-harm limits.
