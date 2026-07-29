# V5.1 guarded model tournament

The tournament searches only on the locked validation split. It never reads
the VoiceBank+DEMAND test metadata.

Stages:

1. Hard parameter, latency, future-independence, streaming-equivalence and
   BF16-gradient gates.
2. Twenty-batch integration runs.
3. A 250-batch architecture screen.
4. Successive halving at 1,000 and 2,000 batches.
5. A three-profile loss tournament with gradient-cosine rejection.
6. Five-epoch confirmation on the selected candidate.
7. Full-utterance paired evaluation and bootstrap comparison against V5.
8. Generation of three fresh final-training configs only when every quality
   and no-harm gate passes.

Prepare configurations without training:

```bash
bash scripts/run_v51_tournament.sh --run-id v51_trial --prepare-only
```

Run selection but do not launch final training:

```bash
bash scripts/run_v51_tournament.sh --run-id v51_trial
```

Allow the controller to launch the three final seeds after all gates pass:

```bash
bash scripts/run_v51_tournament.sh --run-id v51_trial --execute-final
```

Start the selection in a detached user service (replace the run ID as needed):

```bash
bash scripts/start_v51_tournament_detached.sh v51_trial
```

Append `--execute-final` only when the controller should automatically start
all three full training seeds after the evidence gates pass:

```bash
bash scripts/start_v51_tournament_detached.sh v51_trial --execute-final
```

The helper prints exact status, log-follow, and stop commands. Reusing a run ID
resumes completed stages from their latest checkpoints.

The resumable decision record is written to
`results/v51_tournament/<run-id>/decision_report.json`. A failed gate stops
promotion; it does not silently substitute test-set results.
