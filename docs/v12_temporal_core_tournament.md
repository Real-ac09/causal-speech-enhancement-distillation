# V12 temporal-core tournament

## Research question

Does temporal Mamba improve enhancement quality enough to justify its CPU time
and recurrent state when compared with a capacity-matched stateful GRU?

The experiment then isolates two deployment changes:

1. replacing time-kernel-three TF convolutions with time-kernel-one
   convolutions;
2. reducing the GRU state from 232 to 128 channels.

All variants retain the V8 frontend, frequency resolution, continuous noise
condition, direct scalar magnitude decoder, losses, training data, seed, and
training schedule.

## Arms

| Arm | Temporal core | TF time kernel | Parameters |
|---|---|---:|---:|
| `mamba_control` | Mamba, 232 channels | 3 | 1,067,471 |
| `gru_matched` | GRU, 232 channels | 3 | 1,083,479 |
| `gru_matched_time1` | GRU, 232 channels | 1 | 808,095 |
| `gru128_time1` | GRU, 128 channels | 1 | 588,527 |

The matched GRU differs from the control by 1.5% in total parameter count.
Comparing adjacent rows isolates one design decision at a time.

## Structural result

These figures use random weights and measure computational structure only.
They are not enhancement-quality results.

| Arm | p95 frame time | RTF | Persistent state | 10 ms gate |
|---|---:|---:|---:|---|
| Mamba control | 10.38 ms | 0.908 | 5.68 MiB | Fail |
| Matched GRU | 6.65 ms | 0.660 | 1.26 MiB | Pass |
| Matched GRU, time kernel 1 | 3.58 ms | 0.355 | 0.12 MiB | Pass |
| GRU-128, time kernel 1 | 3.25 ms | 0.309 | 0.07 MiB | Pass |

The deployment hypothesis is therefore supported structurally. It still
requires a controlled quality comparison.

## Validation protocol

The previously reused locked-400 subset is excluded. The remaining 937
validation utterances are divided into:

- 100 utterances for epoch selection;
- 400 disjoint utterances for architecture selection;
- 437 disjoint utterances held as an internal reserve.

All three subsets contain speakers p250, p268, and p270. Their membership
hashes are stored in
`data/processed/voicebank_demand/metadata/v12_validation_manifest.json`.
The official test set is not used anywhere in this tournament.

The internal reserve is not a replacement for the final unseen external
holdout.

## Commands

Regenerate and verify the validation split:

```bash
/home/mohamedb/miniconda3/envs/cnvqg/bin/python \
  scripts/create_v12_validation_splits.py
```

Reproduce the structural benchmark:

```bash
LD_LIBRARY_PATH=/home/mohamedb/miniconda3/envs/cnvqg/lib \
PYTHONPATH=src \
/home/mohamedb/miniconda3/envs/cnvqg/bin/python \
  scripts/benchmark_v12_structures.py \
  --warmup-frames 50 \
  --measured-frames 300 \
  --output results/runtime/v12_structural_latency_7800x3d.json
```

Prepare the 12-epoch screen without starting training:

```bash
LD_LIBRARY_PATH=/home/mohamedb/miniconda3/envs/cnvqg/lib \
PYTHONPATH=src \
/home/mohamedb/miniconda3/envs/cnvqg/bin/python \
  scripts/run_v12_temporal_core_tournament.py --stage screen
```

Add `--execute` to train all four arms and evaluate their best checkpoints on
the disjoint architecture-selection set.

Only candidates that pass the real-time gate and remain within 0.05 PESQ of
the Mamba control should enter full training. A candidate should be preferred
over the control when its paired 95% confidence interval is non-inferior in
quality and it materially improves p95 latency or persistent state.

Full runs should use three seeds:

```bash
LD_LIBRARY_PATH=/home/mohamedb/miniconda3/envs/cnvqg/lib \
PYTHONPATH=src \
/home/mohamedb/miniconda3/envs/cnvqg/bin/python \
  scripts/run_v12_temporal_core_tournament.py \
  --stage full \
  --variants mamba_control PROMOTED_VARIANT \
  --seeds 1200 1201 1202 \
  --execute
```

## Untouched internal-reserve confirmation

After the temporal-core decision is frozen, verify the six best full-run
checkpoints on the disjoint 437-utterance internal reserve. The runner verifies
the reserve row count and membership digest before printing or executing any
evaluation command.

Prepare and inspect the commands without loading a model or using VRAM:

```bash
LD_LIBRARY_PATH=/home/mohamedb/miniconda3/envs/cnvqg/lib \
PYTHONPATH=src \
/home/mohamedb/miniconda3/envs/cnvqg/bin/python \
  scripts/run_v12_reserve_confirmation.py
```

The default execution device is CPU. To execute later on the GPU, VRAM use must
be requested explicitly:

```bash
LD_LIBRARY_PATH=/home/mohamedb/miniconda3/envs/cnvqg/lib \
PYTHONPATH=src \
/home/mohamedb/miniconda3/envs/cnvqg/bin/python \
  scripts/run_v12_reserve_confirmation.py \
  --device cuda \
  --execute
```

The workflow evaluates both architectures for seeds 1200, 1201, and 1202,
writes a paired comparison for each seed, and creates a seed-aware hierarchical
bootstrap report at
`results/v12/reserve/aggregate_three_seed/comparison.json`.

The official VoiceBank+DEMAND test set remains untouched until the final hybrid
architecture and its analysis protocol are frozen.

## Frozen decision

The three-seed internal-reserve confirmation promoted
`gru_matched_time1`. Its mean PESQ delta relative to the Mamba control was
-0.0300 with a hierarchical seed-and-file bootstrap 95% interval of
[-0.0438, -0.0157]. The lower bound remained above the predeclared -0.05
non-inferiority boundary. Together with its 3.577 ms structural p95 frame time,
the candidate passes the quality-efficiency gate.

The final architecture, artifact hashes, standard-test safeguards, aggregation
method, and claim limitations are frozen in
`docs/v13_final_hybrid_protocol.md` and
`configs/v13/frozen_final_protocol.yaml`.
