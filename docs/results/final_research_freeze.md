# Final research freeze — 27 July 2026

## Decision

Model development is frozen. V14.2 (`CN-VQG-GRU-T1-PD`) is the final
deployable dissertation model. Its confidence-weighted privileged
distillation recipe, three fixed epoch-3 checkpoints, evaluation protocol,
and model-selection policy may not be changed after this date.

V17 Recipe 7a is retained only as the best balanced preservation-controller
ablation. Recipe 8 is a negative result. Neither controller is promoted.

## Final model

- Architecture: lightweight causal GRU enhancement model
- Parameters: 808,095
- Algorithmic latency: 20 ms
- Training seeds: 1200, 1201, and 1202
- Canonical deployment checkpoint: V14.2 seed 1200, epoch 3
- Training-only privileged teacher: not required at inference

## Frozen evidence

On the 824-file VoiceBank+DEMAND standard comparability test, V14.2 obtains a
three-seed mean PESQ of 2.61755. Relative to V13, the paired hierarchical mean
changes are +0.07069 PESQ, +0.002614 STOI, and +0.002414 ESTOI. SI-SDR is
statistically neutral.

On the independent 150-file DNS1 synthetic no-reverberation condition, V14.2
obtains a three-seed mean PESQ of 1.84210 and improves over V13 by +0.05946
PESQ. The external limitation remains explicit: mean enhanced STOI is below
the unprocessed noisy input.

The seed-1200 native CPU streamer has a real-time factor of 0.372, with 20 ms
algorithmic latency and p95/p99 frame times below the 10 ms hop deadline.

## Claim boundaries

- VoiceBank+DEMAND is a standard comparability result, not a project-wide
  pristine holdout.
- DNS1 supports external generalisation of the frozen V14.2 recipe.
- No broad cross-domain harmlessness claim is permitted because DNS1 STOI
  remains below noisy speech.
- V15–V18 support an ablation and limitation analysis, not a promoted
  preservation controller.
- No later architecture, loss, checkpoint, threshold, or test result may
  replace these frozen claims.

## Integrity

Run:

```bash
python scripts/verify_final_research_freeze.py
```

The verifier checks every checkpoint, configuration, protocol, dataset
manifest, aggregate result, relevant source file, and controller conclusion
against the final SHA-256 manifest.
