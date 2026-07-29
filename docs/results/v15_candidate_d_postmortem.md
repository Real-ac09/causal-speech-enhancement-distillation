# V15 candidate D post-mortem

## Scope

This is a post-hoc development analysis of candidate D, the frozen
candidate-A backbone with a small causal residual-strength gate. It uses the
same 60-file cross-domain development set and 400-file VoiceBank development
set used by the frozen V15 promotion gate.

- Candidate A: quiet-level augmentation, fixed epoch 3.
- Candidate D: candidate A frozen, causal scalar strength gate, fixed epoch 3.
- Paired uncertainty: 20,000 file-bootstrap samples, seed 15041.
- Gate diagnostics and fixed-strength sweeps: CPU only.
- External DNS1 test: not used.
- Status: descriptive diagnosis, not a new candidate or promotion result.

The fixed-strength sweep is post-hoc and therefore cannot be reported as a
predeclared model result. Its purpose is to distinguish a control-capacity
failure from a gate-training failure.

## Verdict

Candidate D's residual-control mechanism has enough capacity to address the
cross-domain intelligibility failure, but the learned gate did not use that
capacity. Its mean strength is approximately 0.9983 on both development sets,
and no analysed frame has strength below 0.99. Candidate D is therefore an
almost constant, approximately 0.17% relaxation of candidate A rather than an
input-dependent preservation system.

The decisive counterfactual is a constant strength of 0.5 on the cross-domain
development set. Relative to the noisy input, it obtains:

- +0.0741 PESQ;
- +0.346 dB SI-SDR;
- +0.01176 STOI, with 20.0% harm;
- +0.01786 ESTOI.

At the quietest -35 dBFS level it obtains +0.00770 STOI with 40.0% harm. These
values clear all four absolute STOI-preservation thresholds that candidate D
failed, while mean cross-domain PESQ, SI-SDR, and ESTOI also improve.

That strength is not globally safe. On VoiceBank, strength 0.5 reduces PESQ
from 2.1494 at full candidate-A strength to 1.6561 and SI-SDR from 14.482 dB
to 10.215 dB. V16 therefore needs a genuinely input-dependent controller; a
global blend setting is not a solution.

## Candidate D versus candidate A

### Cross-domain development

| Metric | A | D | D minus A | Paired 95% CI | D win rate |
|---|---:|---:|---:|---:|---:|
| PESQ | 1.47171 | 1.47547 | +0.00376 | [+0.00262, +0.00502] | 90.0% |
| SI-SDR | 6.7689 dB | 6.7760 dB | +0.0071 dB | [+0.0052, +0.0091] | 86.7% |
| STOI | 0.796489 | 0.799677 | +0.003188 | [+0.001873, +0.004723] | 93.3% |
| ESTOI | 0.671157 | 0.674778 | +0.003621 | [+0.002207, +0.005211] | 93.3% |

The small relaxation improves all four metrics with paired confidence
intervals above zero. It repairs only one of candidate A's 42 STOI-harmed
files, however, leaving 41/60 files harmed relative to noisy.

PESQ and STOI move together on 51/60 files. Only eight files show a
PESQ/STOI disagreement and one regresses on both, so the D intervention is
not exposing an unavoidable aggregate PESQ-versus-STOI trade-off.

### VoiceBank development

| Metric | A | D | D minus A | Paired 95% CI | D win rate |
|---|---:|---:|---:|---:|---:|
| PESQ | 2.14946 | 2.15455 | +0.00509 | [+0.00308, +0.00710] | 62.0% |
| SI-SDR | 14.4824 dB | 14.4800 dB | -0.00235 dB | [-0.00308, -0.00164] | 33.3% |
| STOI | 0.888031 | 0.888244 | +0.000213 | [-0.000050, +0.000475] | 58.0% |
| ESTOI | 0.767441 | 0.767601 | +0.000160 | [-0.000045, +0.000362] | 55.3% |

The SI-SDR decrease is statistically clear but negligible in magnitude.
Candidate D improves VoiceBank PESQ and retains all frozen VoiceBank
safeguards.

## What the gate learned

| Diagnostic | Cross-domain | VoiceBank |
|---|---:|---:|
| Mean file-level strength | 0.998315 | 0.998321 |
| SD of mean strength across files | 0.0000238 | 0.0000281 |
| Mean within-file strength SD | 0.0000279 | 0.0000209 |
| Lowest observed frame strength | 0.997062 | 0.996998 |
| Fraction of frames below 0.99 | 0% | 0% |

The gate was initialised at 0.995, so optimisation moved it closer to full
enhancement. The spectral and waveform reconstruction objectives dominated
the high-SNR identity regulariser. The training set also remained VoiceBank
based; quiet-level scaling exposed absolute-level variation but did not expose
the controller to enough diverse acoustic domains or explicit optimal-strength
targets.

The gate's four mixture summaries do differ between the two development sets.
For example, mean normalised log RMS is -0.172 cross-domain versus -0.090 on
VoiceBank, and mean spectral flatness is 0.369 versus 0.301. The input contains
some useful separating information, but the indirect loss did not turn it into
meaningful control.

## Fixed-strength capacity audit

### Cross-domain development

| Strength | PESQ gain | SI-SDR gain | STOI gain | STOI harm | ESTOI gain |
|---:|---:|---:|---:|---:|---:|
| 0.00 | -0.00069 | -1.024 dB | ~0 | 0.0% | ~0 |
| 0.25 | +0.03129 | -0.337 dB | +0.00740 | 6.7% | +0.00981 |
| 0.50 | +0.07415 | +0.346 dB | +0.01176 | 20.0% | +0.01786 |
| 0.75 | +0.13483 | +0.779 dB | +0.00926 | 35.0% | +0.02141 |
| 1.00 | +0.08365 | +0.400 dB | -0.03037 | 70.0% | -0.00894 |

The non-monotonic result is important. Candidate A's full suppression is past
the useful operating point on this set. Moderate blending improves quality,
distortion, and intelligibility together.

An utterance-level STOI oracle over the five strengths selects 0, 0.25, 0.5,
0.75, and 1.0 for 4, 15, 18, 15, and 8 files respectively. It obtains +0.1106
PESQ, +1.004 dB SI-SDR, +0.01632 STOI, and +0.02855 ESTOI, with no STOI-harmed
files. This is a clean-reference upper bound, not a deployable score, but it
shows that useful strength choices are strongly item dependent.

### VoiceBank development

| Strength | PESQ | SI-SDR | STOI | ESTOI |
|---:|---:|---:|---:|---:|
| 0.00 | 1.43368 | 6.435 dB | 0.856543 | 0.663863 |
| 0.25 | 1.52186 | 8.175 dB | 0.867451 | 0.686004 |
| 0.50 | 1.65607 | 10.215 dB | 0.877372 | 0.710506 |
| 0.75 | 1.86959 | 12.542 dB | 0.885165 | 0.738340 |
| 1.00 | 2.14942 | 14.482 dB | 0.888026 | 0.767438 |

The VoiceBank STOI oracle selects strength 1.0 for 252/400 files and 0.75 for
another 111. This contrasts with the broad cross-domain selection and rules
out a single global strength.

## Additional architectural finding

Strength zero is not a true waveform bypass in candidate D. The implementation
sets the estimated spectral mask to one and still passes the mixture through
the model's analysis/synthesis path. On the cross-domain set this preserves
STOI numerically but loses 1.024 dB mean SI-SDR. A preservation controller
should instead mix a latency-aligned noisy waveform with the enhanced
waveform:

`output = delayed_noisy + strength * (enhanced - delayed_noisy)`

This makes strength zero an actual identity path while retaining constant
work. Alignment and native-streaming equivalence must be unit tested before
training.

## V16 recommendation

The next research hypothesis should be:

> A tiny causal controller trained from explicit, clean-reference
> residual-strength targets on acoustically diverse training mixtures can
> select moderate enhancement when candidate A would damage intelligibility,
> while retaining full enhancement when it improves VoiceBank quality.

The first V16 candidate should remain scalar rather than immediately becoming
bandwise. Strength 0.5 already passes the cross-domain STOI thresholds, so the
present evidence does not justify the extra complexity of a frequency-wise
gate.

The recommended sequence is:

1. Implement a true delay-aligned waveform residual path around the frozen
   candidate-A backbone.
2. Create oracle strength labels on training data only, using the fixed grid
   and clean references. Select a Pareto-safe strength that improves STOI or
   ESTOI while constraining PESQ and SI-SDR regression.
3. Include both VoiceBank-like and diverse DNS training mixtures, with the V15
   quiet-level and SNR range. Do not train on either frozen development set.
4. Train the gate using explicit strength supervision plus light temporal
   smoothness. Retain the enhancement reconstruction objective only as a
   secondary safeguard.
5. Use a separate training-domain calibration split for epoch and threshold
   selection. Evaluate the frozen V15 development gates once after the V16
   recipe is frozen.
6. Replicate only after the seed-1200 candidate clears every gate. Keep the
   external DNS1 test sealed until replication passes.

If supervised scalar control still cannot clear the gate, the next justified
ablation is a small bandwise controller. Input-level normalisation alone is
lower priority because the strength sweep fixes the quiet condition without
altering the backbone frontend.

## Reproducible artifacts

- `scripts/analyze_v15_candidate_d.py`
- `results/v15/preservation/causal_preservation_gate_seed1200/error_analysis_vs_quiet_level/summary.json`
- `results/v15/preservation/causal_preservation_gate_seed1200/error_analysis_vs_quiet_level/cross_per_file.csv`
- `results/v15/preservation/causal_preservation_gate_seed1200/error_analysis_vs_quiet_level/voice_per_file.csv`
- `results/v15/preservation/causal_preservation_gate_seed1200/error_analysis_vs_quiet_level/cross_gate_diagnostics.csv`
- `results/v15/preservation/causal_preservation_gate_seed1200/error_analysis_vs_quiet_level/voice_gate_diagnostics.csv`
- `results/v15/preservation/causal_preservation_gate_seed1200/error_analysis_vs_quiet_level/cross_constant_strength_sweep.csv`
- `results/v15/preservation/causal_preservation_gate_seed1200/error_analysis_vs_quiet_level/voice_constant_strength_sweep.csv`
- `results/v15/preservation/causal_preservation_gate_seed1200/error_analysis_vs_quiet_level/cross_condition_summary.csv`
- `results/v15/preservation/causal_preservation_gate_seed1200/error_analysis_vs_quiet_level/voice_speaker_summary.csv`
- `results/v15/preservation/causal_preservation_gate_seed1200/error_analysis_vs_quiet_level/top_regressions.csv`
- `results/v15/preservation/causal_preservation_gate_seed1200/error_analysis_vs_quiet_level/top_improvements.csv`
