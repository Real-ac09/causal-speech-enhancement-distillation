# V14.2 frozen standard-test and runtime result

## Protocol

The V14.2 checkpoint was frozen before this evaluation:

- checkpoint: `checkpoints/v14/distillation/mag_005/epoch_003.pt`
- checkpoint SHA-256:
  `0b58d6def618cc67db7b14848c406015f76b25c247757d797120ff3a2801f891`
- protocol: `configs/v14/frozen_final_protocol.yaml`
- protocol SHA-256:
  `7fbd62c82ae96d369aadcdec6c7fac38f40dc71c4937c70d685b60b698bfa41c`
- standard test: 824 VoiceBank+DEMAND utterances
- comparison: paired against the frozen V13 seed-1200 result
- uncertainty: 20,000 paired file-bootstrap samples
- model or checkpoint reselection after observing this result: prohibited

The standard set has historical use in this project. This is therefore a
comparability result, not a claim of a project-wide pristine holdout. V14.2 is
a single-seed distillation result and must not be presented as a three-seed
estimate.

## Standard-test quality

| Metric | V13 seed 1200 | V14.2 seed 1200 | Paired change | Paired 95% CI | File win rate |
|---|---:|---:|---:|---:|---:|
| PESQ | 2.56477 | **2.64286** | **+0.07809** | **[+0.06810, +0.08802]** | 70.75% |
| SI-SDR (dB) | 18.16018 | **18.22858** | **+0.06841** | **[+0.05808, +0.07919]** | 74.39% |
| SI-SDRi (dB) | 9.71465 | **9.78306** | **+0.06841** | **[+0.05808, +0.07919]** | 74.39% |
| STOI | 0.936228 | **0.937126** | **+0.000898** | **[+0.000166, +0.001652]** | 50.24% |
| ESTOI | 0.840490 | **0.841084** | +0.000594 | [-0.000159, +0.001348] | 54.13% |

PESQ, SI-SDR, and STOI improved with paired confidence intervals wholly above
zero. ESTOI remained statistically compatible with no change and showed no
evidence of a material regression. The standard-test PESQ improvement is
smaller than the protected-development improvement (+0.078 versus +0.119),
but it remains clear and directionally consistent.

## Real-time deployment benchmark

The existing native causal streamer was measured for 30 seconds (3,000
10-millisecond frames) on one CPU thread on an AMD Ryzen 7 7800X3D.

| Deployment quantity | V13 seed 1200 | V14.2 seed 1200 |
|---|---:|---:|
| Parameters | 808,095 | 808,095 |
| Algorithmic latency | 20 ms | 20 ms |
| Median frame time | 3.593 ms | 3.652 ms |
| p95 frame time | 4.056 ms | 4.077 ms |
| p99 frame time | 4.160 ms | 4.184 ms |
| Mean frame time | 3.685 ms | 3.724 ms |
| Streaming RTF | 0.368 | 0.372 |
| Persistent state (FP32) | 0.119 MiB | 0.119 MiB |
| p95 and p99 below 10 ms deadline | pass | **pass** |

The small timing difference is normal run-to-run CPU variation because the
deployment graph and state dimensions are unchanged. The privileged teacher
is used only during training and adds no inference parameters, state, or
algorithmic latency.

## Verdict

V14.2 is promoted as the current single-seed final model. It delivers a
statistically clear PESQ improvement without trading away SI-SDR or STOI, and
it preserves the lightweight causal real-time deployment contract.

The defensible dissertation claim is:

> Confidence-weighted privileged distillation improved the frozen
> 808k-parameter causal model from 2.565 to 2.643 PESQ on the standard
> VoiceBank+DEMAND test set, while slightly improving SI-SDR and STOI and
> retaining 20 ms algorithmic latency and one-thread CPU real-time operation
> (RTF 0.372).

Two limitations remain: the V14.2 improvement has only one trained seed, and
the VoiceBank+DEMAND test set is not a project-wide pristine holdout.
Multi-seed replication and an external holdout are required for a stronger
generalisability claim.
