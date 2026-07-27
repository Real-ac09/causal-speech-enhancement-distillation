# V15 candidate B paired error analysis

## Scope

This analysis compares candidate A (quiet-level augmentation) with candidate B
(the same recipe plus 10% clean-input identity examples). It uses the frozen
60-file cross-domain development set and 400-file VoiceBank development set.
No enhancement inference was rerun and the DNS1 external test was not used.

All confidence intervals use a 20,000-sample paired file bootstrap. Results
remain single-seed development evidence and are not final test claims.

## Verdict

Reject candidate B and retain candidate A as the best V15 checkpoint.
Random clean-input replacement did not deliver a reliable intelligibility
improvement. It caused statistically supported VoiceBank PESQ and SI-SDR
regressions, while its mean STOI change was indistinguishable from zero.

Candidate C should remain skipped because it was defined as a continuation of
B. The evidence instead activates the predeclared candidate D: a small causal
preservation gate around the frozen candidate-A backbone.

## Overall paired result

### Cross-domain development

| Metric | A | B | B minus A | Paired 95% CI | B win rate |
|---|---:|---:|---:|---:|---:|
| PESQ | 1.47171 | 1.45898 | -0.01274 | [-0.04215, +0.00633] | 53.3% |
| SI-SDR | 6.7689 dB | 6.7101 dB | -0.0588 dB | [-0.2060, +0.0867] | 41.7% |
| STOI | 0.796489 | 0.796364 | -0.000125 | [-0.001586, +0.001402] | 55.0% |
| ESTOI | 0.671157 | 0.670611 | -0.000546 | [-0.002390, +0.001373] | 46.7% |

None of the aggregate A-to-B cross-domain changes is statistically clear.
Candidate B repairs two of A's 42 STOI-harmed files and introduces none, but
the remaining negative changes are large enough that mean STOI becomes
slightly worse.

### VoiceBank development

| Metric | A | B | B minus A | Paired 95% CI | B win rate |
|---|---:|---:|---:|---:|---:|
| PESQ | 2.14946 | 2.12571 | **-0.02375** | **[-0.03548, -0.01406]** | 45.5% |
| SI-SDR | 14.4824 dB | 14.3552 dB | **-0.1271 dB** | **[-0.2551, -0.0332]** | 36.0% |
| STOI | 0.888031 | 0.888220 | +0.000189 | [-0.000163, +0.000543] | 50.2% |
| ESTOI | 0.767441 | 0.767030 | -0.000412 | [-0.000949, +0.000141] | 43.0% |

The VoiceBank PESQ and SI-SDR regressions are statistically supported. The
5%-trimmed changes remain negative at -0.01250 PESQ and -0.0289 dB SI-SDR, so
the result is not explained solely by the largest outliers.

## Condition dependence

The intervention behaves in opposite directions at the two clean-level
extremes:

| Target clean RMS | PESQ B-A | SI-SDR B-A | STOI B-A | ESTOI B-A |
|---:|---:|---:|---:|---:|
| -35 dBFS | -0.06055 | -0.2591 dB | -0.002294 | -0.002161 |
| -30 dBFS | -0.00050 | +0.1501 dB | -0.001345 | -0.003283 |
| -25 dBFS | -0.00233 | -0.2487 dB | -0.000728 | -0.001105 |
| -20 dBFS | +0.01244 | +0.1225 dB | +0.003866 | +0.004364 |

At -35 dBFS, B improves only 4/15 files in PESQ and 5/15 in STOI. At -20
dBFS, it improves 12/15 in PESQ and 13/15 in STOI. Random identity examples
therefore do not address the quietest condition that motivated V15; their
benefit is concentrated at the loudest level.

On VoiceBank, mean PESQ changes are -0.0400 for speaker `p250`, -0.0393 for
`p268`, and approximately neutral (+0.00015) for `p270`. This is evidence of
speaker/content sensitivity, not a demographic conclusion.

## Collapse-like cases

A diagnostic flags a case when candidate A gains at least 1 dB SI-SDR or 0.20
PESQ and candidate B retains at most 25% of that gain. It identifies 2/60
cross-domain files and 5/400 VoiceBank files.

The clearest cases are:

| File | Dataset | A SI-SDR gain | B SI-SDR gain | A PESQ gain | B PESQ gain |
|---|---|---:|---:|---:|---:|
| `p250_465` | VoiceBank | +18.064 dB | +0.030 dB | +0.029 | -0.000 |
| `p250_301` | VoiceBank | +14.403 dB | +2.871 dB | +1.790 | +0.501 |
| `p250_095` | VoiceBank | +2.616 dB | +2.258 dB | +1.396 | +0.191 |
| `dns_train_dev_041` | cross-domain | +2.017 dB | +0.167 dB | +0.111 | +0.014 |

CPU-only re-inference supports the near-identity interpretation. On
`p250_465`, B's mean magnitude mask is 0.990 versus A's 0.932, and B's
waveform change from the noisy input is only 0.75% of A's change. On
`dns_train_dev_041`, the corresponding values are 0.975 versus 0.934 and
16.4%. On `p250_301`, they are 0.969 versus 0.926 and 58.9%. Six of the seven
flagged cases move closer to identity by waveform-residual magnitude, although
the strength of that effect varies. The intervention has therefore created
intermittent bypass-like behaviour rather than a consistently calibrated
preservation response.

## Metric disagreement

The identity intervention does not create a single PESQ-versus-STOI trade-off:

| Outcome | Cross-domain files | VoiceBank files |
|---|---:|---:|
| PESQ and STOI improve | 27 | 107 |
| STOI improves, PESQ regresses | 6 | 94 |
| PESQ improves, STOI regresses | 5 | 75 |
| Both regress | 22 | 124 |

Because all four quadrants are well represented, a single global loss weight
or residual strength is unlikely to solve the failure. The control needs to be
input-dependent.

## Recommended next candidate

Proceed to candidate D from the frozen V15 ablation programme:

1. Use candidate A epoch 3 as a frozen backbone.
2. Add the existing causal residual-strength gate, staying below the
   predeclared 10,000-parameter gate budget.
3. Train only the gate; do not continue candidate B and do not use random
   clean-input replacement in the first D run.
4. Retain quiet-level augmentation during gate training so the gate sees the
   -35 to -20 dBFS range.
5. Initialise close to full candidate-A enhancement and penalise unnecessary
   bypass behaviour. Permit reduced strength only when causal mixture and
   backbone-state features predict over-processing.
6. Apply the same 17 gates on both development sets. Do not reuse the external
   test unless a candidate passes the frozen seed-1200 development gate and
   then the required multi-seed replication.

This is better targeted than candidate C. A magnitude-underestimation penalty
would globally change suppression, whereas the observed failure is strongly
condition- and utterance-dependent.

## Reproducible artifacts

- `scripts/analyze_v15_identity_errors.py`
- `results/v15/preservation/quiet_level_identity_seed1200/error_analysis_vs_quiet_level/summary.json`
- `results/v15/preservation/quiet_level_identity_seed1200/error_analysis_vs_quiet_level/cross_per_file.csv`
- `results/v15/preservation/quiet_level_identity_seed1200/error_analysis_vs_quiet_level/voice_per_file.csv`
- `results/v15/preservation/quiet_level_identity_seed1200/error_analysis_vs_quiet_level/cross_condition_summary.csv`
- `results/v15/preservation/quiet_level_identity_seed1200/error_analysis_vs_quiet_level/voice_speaker_summary.csv`
- `results/v15/preservation/quiet_level_identity_seed1200/error_analysis_vs_quiet_level/top_regressions.csv`
- `results/v15/preservation/quiet_level_identity_seed1200/error_analysis_vs_quiet_level/identity_collapse_cases.csv`
- `scripts/export_v15_identity_diagnostics.py`
- `results/v15/preservation/quiet_level_identity_seed1200/error_analysis_vs_quiet_level/listening_set/selection_manifest.csv`
- `results/v15/preservation/quiet_level_identity_seed1200/error_analysis_vs_quiet_level/listening_set/paired_diagnostics.csv`
