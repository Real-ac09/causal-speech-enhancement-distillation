# Lightweight speech-enhancement benchmark

This benchmark deliberately separates causal systems from offline systems. A
bidirectional or centred-STFT score is not evidence that the same model can meet
a 20 ms streaming budget. `NR` in the companion
[`lightweight_se_benchmark.csv`](lightweight_se_benchmark.csv) means that the
value still needs to be verified from the primary paper or released code; it is
not treated as zero.

## Causal and streaming ranking

| Rank | Model | Parameters | WB-PESQ | Important qualification |
|---:|---|---:|---:|---|
| 1 | SEMamba-Uni | 1.41M | 3.29 | Unidirectional; verify end-to-end streaming frontend and CPU timing |
| 2 | aTENNuate | ~0.84M | 3.27 | Raw waveform, but headline training uses additional data |
| 3 | FastEnhancer-M | 0.492M | 3.24 | Streaming comparison target |
| 4 | AdaptCRN | 0.135M | 2.98 | Streaming |
| 5 | GTCRN | 0.024M | 2.87 | Streaming; reported STOI 0.940, ESTOI 0.848, SI-SDR 18.8 dB |
| 6 | CN-VQG-GRU-T1-PD | 0.808M | 2.618 | 20 ms causal; measured one-thread CPU RTF 0.372; three training seeds |

CN-VQG-GRU-T1-PD is now a validated real-time research baseline rather than an
unscored proposal. It remains below the published causal comparison targets in
PESQ, but its runtime claim is backed by an end-to-end native-streaming CPU
measurement and its quality estimate by three training seeds. Its independent
DNS1 result is reported separately in
[`results/v14_2_three_seed_external.md`](results/v14_2_three_seed_external.md).

## Offline ranking

| Rank | Model | Parameters | WB-PESQ | Qualification |
|---:|---|---:|---:|---|
| 1 | SEMamba + PCS | ~2.25M | 3.69 | PESQ-oriented target transform; report separately |
| 2 | SEMamba + consistency | ~2.25M | 3.55 | Offline |
| 3 | SEMamba-Bi | 2.25M | 3.52 | Bidirectional |
| 4 | MP-SENet | 2.05M | 3.50 | Offline magnitude/phase model |
| 5 | TridentSE-L | 3.03M | 3.47 | Offline |
| 6 | TridentSE-M | 1.42M | 3.44 | Offline |
| 7 | CMGAN | 1.83M | 3.41 | Offline, metric-guided training |
| 8 | TridentSE-S | 1.00M | 3.36 | Offline |
| 9 | DPT-FSNet | 0.88M | 3.33 | Primarily offline |
| 10 | DB-AIAT | 2.81M | 3.31 | Offline |
| 11 | Current V4.3 | 1.147M | 2.871 | Centred-STFT project reference |

## Sources and reporting policy

Primary sources: [FastEnhancer](https://arxiv.org/abs/2509.21867),
[aTENNuate](https://arxiv.org/abs/2409.03377),
[DPT-FSNet](https://arxiv.org/abs/2104.13002),
[DB-AIAT](https://arxiv.org/abs/2110.06467),
[TridentSE](https://www.isca-archive.org/interspeech_2023/yin23_interspeech.html),
[MP-SENet](https://arxiv.org/abs/2305.13686),
[CMGAN](https://arxiv.org/abs/2209.11112), and
[SEMamba](https://arxiv.org/abs/2405.06573).

Every final proposed-model row must include PESQ, STOI, ESTOI, SI-SDR,
DNSMOS/SCOREQ (where
available), CPU/GPU RTF, p50/p95 frame time, memory, training data, and three-seed
uncertainty. PESQ-oriented training is accepted only when intelligibility and
SI-SDR safeguards pass and the fixed listening set does not reveal musical
noise, phase smearing, or speech distortion.
