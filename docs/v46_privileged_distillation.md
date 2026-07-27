# V4.6 privileged causal distillation

V4.6 tests whether the completed noncausal V4.3 model can provide privileged
training targets to the causal V4.4 student without changing deployment.

## Contract

- Teacher: frozen V4.3 epoch 15, used only during training.
- Student: V4.4 epoch 17, 745,155 parameters, 20 ms causal frontend.
- Deployment graph: unchanged V4.4 student; no teacher operations or future
  context are present at inference.
- Phase: the deployed student retains noisy phase. Future-informed teacher
  phase and hidden states are deliberately not distilled.

Teacher and student use different frame rates and temporal contracts. The
distillation loss therefore computes both outputs on the student's common
causal STFT grid and copies teacher log-magnitude only where the teacher is
closer to clean than the noisy input. Confidence is detached and has a small
floor to avoid discontinuous target removal.

The teacher scores 2.5750 PESQ, 16.659 dB SI-SDR, 0.9075 STOI and 0.7927 ESTOI
on the fixed 100-file search set. The starting causal student scores 2.3610
PESQ, leaving a 0.214 PESQ teacher margin. The integration smoke found the
teacher better than noisy in roughly 85% of time-frequency bins.

## Search

Two four-epoch continuation candidates use log-magnitude distillation weights
of 0.02 and 0.05. Both start from the same V4.4 epoch-17 checkpoint and use
600 batches per epoch. Every epoch checkpoint is scored externally on complete
utterances.

The search promotion gate requires at least +0.01 PESQ relative to V4.4 while
limiting losses to 0.15 dB SI-SDR, 0.002 STOI and 0.003 ESTOI. The best safe
candidate is evaluated on locked-400 regardless of its search result. A locked
promotion additionally requires a positive paired-bootstrap PESQ interval.

Monitor `logs/v46/privileged_distillation.log`. Decisions are written to
`results/v46/search/decision.json` and locked comparison results to
`results/v46/locked400/v44_vs_v46/`.
## Final outcome

The search and locked-validation gates both passed. The selected
`mag_005/epoch_004.pt` checkpoint improved locked-400 WB-PESQ from 2.31422 to
2.34518 while also improving SI-SDR, STOI, and ESTOI. All four paired-bootstrap
95% confidence intervals excluded zero. Structural validation also passed
causality, streaming equivalence, 20 ms latency, parameter-cap, and BF16-gradient
checks. Full results are recorded in `results/v46/RUN_CONCLUDED.md`.
