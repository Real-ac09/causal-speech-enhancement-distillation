# V13 final-model error analysis

## Scope and method

This analysis uses the completed three-seed outputs for all 824 standard
VoiceBank+DEMAND test utterances. It does not rerun enhancement inference.
Per-file results are averaged across seeds 1200, 1201, and 1202 before harm
rates are calculated.

The test metadata contains speaker and duration but not noise identity or SNR.
For diagnostic stratification only, input SNR and noise-spectrum descriptors
are calculated from the aligned clean/noisy pairs. These descriptors use clean
references and are not available to a deployed model. They must not be
described as inference-time features or official dataset labels.

## Overall failure rates

| Metric | Mean gain | 95% file-bootstrap CI | Files harmed | Seed sign disagreement |
|---|---:|---:|---:|---:|
| PESQ | +0.5784 | [0.5503, 0.6072] | 40 / 824 (4.9%) | 31 (3.8%) |
| SI-SDR | +9.768 dB | [9.473, 10.055] | 2 / 824 (0.2%) | 0 |
| STOI | +0.01324 | [0.01174, 0.01476] | 165 / 824 (20.0%) | 151 (18.3%) |
| ESTOI | +0.05191 | [0.04831, 0.05555] | 91 / 824 (11.0%) | 80 (9.7%) |

Enhancement improves every metric on 614 files (74.5%). At least one metric is
harmed on 210 files (25.5%), two or more are harmed on 73 files (8.9%), and no
file is harmed on all four metrics.

The most common failure patterns are:

| Harm pattern | Files |
|---|---:|
| STOI only | 97 |
| STOI and ESTOI | 46 |
| ESTOI only | 25 |
| PESQ only | 15 |
| PESQ, STOI, and ESTOI | 13 |
| PESQ and STOI | 7 |
| PESQ and ESTOI | 5 |
| SI-SDR, STOI, and ESTOI | 2 |

This is primarily an intelligibility-consistency problem, not a general
denoising collapse.

## Condition-level findings

### Already-clean and high-SNR inputs

The easiest noisy-PESQ quartile has the highest harm rates:

- 13.1% PESQ harm;
- 26.7% STOI harm;
- 29.6% ESTOI harm.

The signal-derived `>=15 dB` group similarly has 9.9% PESQ harm, 24.6% STOI
harm, and 22.0% ESTOI harm. Its mean SI-SDR gain is still positive at
+5.57 dB, but substantially smaller than the +13.39 dB obtained below 5 dB
input SNR.

This pattern is consistent with over-processing: the fixed-strength enhancer
has less useful noise to remove and a greater opportunity to alter speech that
was already intelligible. This interpretation is a hypothesis until the
selected files are listened to and their masks/spectrograms are inspected.

### Noise spectral character

The low-centroid group has:

- 10.2% PESQ harm;
- 25.5% STOI harm;
- 23.3% ESTOI harm.

The tonal/low-flatness group gives a similar 9.8%, 24.7%, and 21.8%. In
contrast, the high-centroid group has only 2.2% PESQ harm and 3.3% ESTOI harm.
The model obtains large SI-SDR gains on low-centroid/tonal interference but
does not preserve intelligibility as consistently.

The evidence supports a suppression-calibration hypothesis: low-frequency or
structured interference may overlap speech cues in a way that a scalar
magnitude mask can remove energetically while still damaging intelligibility.
The analysis does not identify the named noise source because that label is
absent from the metadata.

### Speaker/content sensitivity

Speaker `p232` has lower mean PESQ gain than `p257` (+0.417 versus +0.726) and
a higher PESQ harm rate (7.4% versus 2.6%). Speaker `p257` has the higher STOI
harm rate (23.4% versus 16.3%). With only two test speakers, this should be
reported as speaker/content sensitivity rather than a demographic
generalisation.

### Seed sensitivity

Mean seed-to-seed variation is modest for aggregate quality, but per-file
intelligibility decisions are unstable near zero gain:

- 18.3% of files change the sign of STOI gain between seeds;
- 9.7% change the sign of ESTOI gain;
- 3.8% change the sign of PESQ gain;
- no file changes the sign of SI-SDR gain.

The model is therefore stable as a denoiser but less stable in which fine
speech cues it preserves.

## Representative failures

The largest mean PESQ regressions are:

| File | Input SNR | Noisy PESQ | PESQ gain |
|---|---:|---:|---:|
| `p232_325` | 15.38 dB | 2.244 | -0.474 |
| `p257_304` | 15.87 dB | 4.167 | -0.430 |
| `p232_083` | 5.42 dB | 1.861 | -0.391 |
| `p232_045` | 14.37 dB | 3.758 | -0.365 |
| `p232_350` | 15.37 dB | 4.301 | -0.297 |

Only two files lose SI-SDR: `p257_001` (-0.625 dB) and `p257_376`
(-1.574 dB). Both are high-SNR, very low-centroid cases; both still improve
PESQ while harming STOI and ESTOI. They are useful examples of why a single
objective metric is insufficient.

Fifteen files lose PESQ while improving both intelligibility measures and
SI-SDR. Metric disagreement must therefore be retained in the dissertation
rather than collapsing the evaluation into a single pass/fail score.

## Diagnosis

The results do not point to the projected GRU as the main remaining
bottleneck. Temporal denoising is strong: SI-SDR improves on 822 of 824 files,
with the largest gains under difficult low-SNR conditions.

The evidence instead points to:

1. insufficient identity/bypass behaviour for already-clean speech;
2. suppression that is not calibrated to input difficulty;
3. inconsistent preservation of intelligibility cues, especially for
   low-centroid or tonal interference;
4. a magnitude-only reconstruction path whose perceptual trade-offs are not
   fully controlled;
5. seed-level variability in fine speech-cue preservation despite stable
   aggregate denoising.

Items 1--3 are supported by stratified associations. Item 4 is an architectural
hypothesis and cannot be isolated from this analysis alone.

## Next actions

For the completed V13 study:

1. Create a fixed listening set containing worst-harm, metric-disagreement,
   median, and best-gain cases.
2. Inspect enhanced/noisy/clean spectrograms and predicted mask behaviour for
   those files.
3. Report per-file harm rates and seed sensitivity alongside mean metrics.
4. Evaluate the frozen model on a genuinely unseen external holdout.

If a separate V14 programme is justified, the first controlled experiment
should test a causal input-quality/noise-confidence residual gate that can
approach identity on clean inputs. It should be trained with explicit
near-clean examples and a no-harm preservation objective. It must use new
development data and cannot be selected from the completed standard-test
results.

## Reproducible artifacts

- `scripts/analyze_v13_errors.py`
- `results/v13/error_analysis/summary.json`
- `results/v13/error_analysis/per_file_analysis.csv`
- `results/v13/error_analysis/condition_summary.csv`
- `results/v13/error_analysis/worst_cases.csv`
- `results/v13/error_analysis/gain_vs_input_quality.png`
- `results/v13/error_analysis/gain_by_input_snr.png`
- `results/v13/error_analysis/metric_harm_rates.png`
- `results/v13/error_analysis/seed_sensitivity.png`
