# V13 frozen final-hybrid protocol

## Decision

The final dissertation candidate is **CN-VQG-GRU-T1**, the V12
`gru_matched_time1` student. The architecture and checkpoint policy were frozen
on 25 July 2026 before this candidate was evaluated on the standard
VoiceBank+DEMAND test metadata.

This is a hybrid by division of responsibility:

1. A causal STFT frontend provides a compact time-frequency representation.
2. Time-kernel-one convolutions learn local spectral structure without keeping
   temporal convolution history.
3. A four-frame rolling continuous state estimates the current noise context
   and modulates the bottleneck.
4. A projected GRU provides causal temporal memory.
5. A direct scalar magnitude mask performs bounded reconstruction while noisy
   phase is retained.
6. Causal overlap-add returns the waveform with 20 ms algorithmic latency and
   a 10 ms processing deadline.

The frozen model has 808,095 parameters. On the development CPU it had a
structural p95 frame time of 3.577 ms and 0.119 MiB of persistent FP32 tensor
state.

## Why the other explored modules are absent

The final model does not combine every technique that was implemented. Each
branch had to earn inclusion in a controlled comparison.

| Decision | Evidence | Final status |
|---|---|---|
| Projected GRU, time kernel 1 | Reserve PESQ delta -0.0300; hierarchical 95% CI [-0.0438, -0.0157], above the -0.05 non-inferiority boundary | Included |
| Global-frequency attention | No V10 screen promotion; full D40 check gave PESQ delta -0.0040 with CI crossing zero | Excluded |
| Explicit scale decoder | PESQ changes below +0.001 with intervals crossing zero | Excluded |
| Asymmetric magnitude loss | PESQ and SI-SDR regressed | Excluded |
| Auxiliary VQ | Disabled in the strongest V8/V12 enhancement path | Excluded from final inference |
| Phase correction | Earlier objectives did not establish a reliable benefit | Excluded |

The quality claim is therefore non-inferiority within a predeclared PESQ
margin in exchange for a material runtime and state reduction. It is not a
claim that the GRU has equal or superior enhancement quality.

## Frozen artifacts

The machine-readable record is
`configs/v13/frozen_final_protocol.yaml`. It locks:

- the three training seeds and exact checkpoint hashes;
- the exact training-config hashes;
- the V12 reserve decision and structural-runtime evidence;
- critical inference and metric source-file hashes;
- the standard-test metadata hash and row count;
- metrics, seed aggregation, bootstrap settings, runtime settings, and the
  post-test no-reselection policy.

If a locked artifact changes, the final runner stops before evaluation.

## Standard VoiceBank+DEMAND evaluation

All three preselected best checkpoints are evaluated. No seed is selected from
the standard-test result. Report the three-seed mean, seed standard deviation,
and hierarchical 95% confidence interval over both training seed and paired
file identity for:

- wideband PESQ;
- SI-SDR and SI-SDR improvement;
- STOI;
- ESTOI.

The one-thread native-streaming runtime is repeated for seed 1200 over a
30-second measurement after one second of warm-up. Both p95 and p99 must remain
below the 10 ms hop deadline.

Prepare and verify everything without evaluating audio:

```bash
LD_LIBRARY_PATH=/home/mohamedb/miniconda3/envs/cnvqg/lib \
PYTHONPATH=src \
/home/mohamedb/miniconda3/envs/cnvqg/bin/python \
  scripts/run_v13_final_protocol.py
```

Execute the frozen protocol later:

```bash
LD_LIBRARY_PATH=/home/mohamedb/miniconda3/envs/cnvqg/lib \
PYTHONPATH=src \
/home/mohamedb/miniconda3/envs/cnvqg/bin/python \
  scripts/run_v13_final_protocol.py \
  --device cuda \
  --execute \
  --acknowledge-standard-test
```

The explicit acknowledgement records that these scores cannot be used to alter
the architecture or select a preferred seed.

## Claim boundary and external validation

The repository contains historical VoiceBank test evaluations from earlier
model generations. Consequently, the standard test remains valid for
comparison with published VoiceBank results, but it is not a project-wide
pristine holdout. The dissertation should state this limitation.

A genuinely unseen external corpus is still required for the stronger
generalisation claim. Its speaker-disjoint holdout must be created and hashed
before model access, and must not be used for further selection. Until that is
done, describe external robustness as untested.

CSIG, CBAK, COVL, DNSMOS, and listening-test results should be reported only
after their implementations or protocols are independently validated. They
must not be retroactively introduced as selection criteria.
