# V6 causal complex enhancement rework

V6 is a deliberately simpler causal baseline created after V5.2 reached
2.2925 PESQ but remained 0.210 below the V4.3 locked-validation reference.

## Signal path

- 20 ms causal STFT window and 10 ms hop, with `center=false` behaviour.
- Compressed magnitude plus unit real/imaginary input features.
- One frequency downsampling stage and a full-resolution skip.
- Local causal gated convolutions surrounding one projected temporal Mamba
  stack. There is no enhancement-path noise conditioning or iterative
  refinement.
- An identity-initialised complex mask. The model starts as noisy-signal
  reconstruction and learns only a bounded correction.
- The noise representation and optional EMA VQ branch are auxiliary only and
  cannot change enhanced audio.

Teacher defaults are approximately 1.96M parameters with Mamba. Student
defaults are approximately 1.01M parameters. Both retain the V5 streaming API.

## Reconstruction search

The initial controlled search compares three otherwise identical models:

1. `complex_ratio`: bounded Cartesian complex-mask residual.
2. `magnitude_only`: log-magnitude mask with noisy phase.
3. `polar_residual`: log-magnitude mask plus bounded noisy-phase correction.

Each candidate uses the same seed, data order, architecture width, optimiser
and simple foundation losses. Search results select the representation used by
a fresh full training run; search weights are not continued into that run.

Run structural checks and the search tournament with:

```bash
bash scripts/run_v6_rework.sh search
```

After inspecting `results/v6/search_winner.json`, launch the fresh full run:

```bash
bash scripts/run_v6_rework.sh full
```

Use `all` only when automatic promotion into full training is intended.
