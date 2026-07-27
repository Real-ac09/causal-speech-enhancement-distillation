# V14.0 development residual-blend screen

## Scope

This is a diagnostic screen on the 400-file V12 architecture-selection
development partition using the frozen V13 seed-1200 checkpoint. It is not a
standard-test result and it is not eligible as a final dissertation claim.

The evaluated output was:

`blended = noisy + strength * (enhanced - noisy)`

The current V13 output is the `1.00` reference.

## Results

| Strength | PESQ delta | PESQ paired 95% CI | SI-SDR delta | STOI delta | ESTOI delta | Promoted |
|---:|---:|---:|---:|---:|---:|:---:|
| 0.80 | -0.1011 | [-0.1308, -0.0715] | -1.5127 dB | +0.00003 | -0.02134 | No |
| 0.90 | +0.0489 | [+0.0259, +0.0716] | -0.6068 dB | +0.00164 | -0.00886 | No |
| 0.95 | +0.1151 | [+0.0956, +0.1343] | -0.2279 dB | +0.00198 | -0.00291 | No |

The `0.95` blend provides a large, statistically supported PESQ increase and
also improves mean STOI, but it exceeds the predeclared maximum SI-SDR and
ESTOI drops. No constant blend passes the no-harm gate.

PESQ harm rate falls from 3.75% at `1.00` to 2.25% at `0.95`. STOI harm rate
falls from 11.75% to 7.50%, and ESTOI harm rate falls from 6.25% to 3.75%.
Thus, the failed promotion is caused by the mean SI-SDR and ESTOI safeguards,
not by an increase in per-file harm incidence.

## Heterogeneity diagnostic

At strength `0.95`:

- PESQ improves on 291 of 400 files;
- SI-SDR improves on 79 files;
- STOI improves on 212 files;
- ESTOI improves on 144 files;
- all three safeguard metrics are non-worse on 35 files;
- 32 of those 35 files also improve PESQ.

A non-deployable strict oracle was calculated only as a feasibility bound. For
each file, it selects the highest-PESQ strength among `0.80`, `0.90`, `0.95`,
and `1.00`, but permits a reduced strength only when SI-SDR, STOI, and ESTOI
are each no worse than the `1.00` result. It selects:

| Strength | Files |
|---:|---:|
| 0.80 | 6 |
| 0.90 | 5 |
| 0.95 | 22 |
| 1.00 | 367 |

Relative to always using `1.00`, this diagnostic oracle changes the means by:

| Metric | Delta |
|---|---:|
| PESQ | +0.02420 |
| SI-SDR | +0.02613 dB |
| STOI | +0.00096 |
| ESTOI | +0.00116 |

The oracle uses clean-reference metrics on the same development files and
therefore cannot be deployed or treated as an unbiased performance estimate.
It establishes only that beneficial and harmful blend regimes coexist.

## Decision

Reject a global residual blend. Proceed to V14.1: an input-dependent causal
confidence gate, initially trained with the V13 backbone frozen. The gate must
use noisy-input and existing causal-state features only. It must be evaluated
against the unchanged `1.00` V13 output using the predeclared no-harm gates.
