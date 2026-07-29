# V10 causal global-frequency Mamba programme

V10 tests one primary change relative to V8: low-rank full-band attention
within each current time frame. The separation between temporal recurrence and
within-frame frequency attention is motivated by FastEnhancer (Ahn et al.,
ICASSP 2026, <https://arxiv.org/abs/2509.21867>); the bottlenecked residual
implementation and its integration with V8 are implemented locally. It does
not add future audio context. The two disabled V8 prediction heads are omitted;
they do not participate in enhancement and this releases 27,105 parameters.
Phase correction is deferred until the isolated global-frequency result is known.

## Experiment order

1. `control`: the active V8 direct-scalar-mask path expressed through V10.
2. `global_d32`: 32-channel frequency bottleneck.
3. `global_d40`: 40-channel frequency bottleneck near the 1.10M cap.
4. Three fresh full-training seeds are generated only after promotion.

The programme uses the 100-file search set for checkpoint selection and the
locked 400-file validation set for paired promotion. It never reads the test
metadata. Promotion requires the configured PESQ gain, SI-SDR/STOI/ESTOI
safeguards, and a positive paired-bootstrap 95% PESQ interval. VQ and
perceptual fine-tuning and phase correction are deliberately deferred until
the backbone decision is complete.

The inherited chunk API remains the whole-history correctness reference. This
programme tests enhancement quality and causal structure; it does not claim a
constant-cost streaming runtime. Cached STFT, overlap-add, convolution, and
Mamba states remain a separate deployment gate before any real-time claim.

## Prepare without training

```bash
bash scripts/run_v10_programme.sh --prepare-only --run-id v10_main
```

## Run after the GPU becomes idle

The foreground command waits for every existing `scripts/train.py` process:

```bash
bash scripts/run_v10_programme.sh --wait-for-idle --run-id v10_main
```

Queue the same behaviour as a detached user service:

```bash
bash scripts/start_v10_after_current.sh v10_main
```

Full three-seed training is not automatic. Enable it explicitly only when the
programme should continue beyond the successful architecture gate:

```bash
bash scripts/start_v10_after_current.sh v10_main --execute-full
```

Environment overrides for short-screen budgets are `V10_SCREEN_EPOCHS` and
`V10_SCREEN_BATCHES`. The resumable
decision record is `results/v10/<run-id>/decision_report.json`.
