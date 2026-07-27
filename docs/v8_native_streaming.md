# V8 native streaming baseline

## Purpose

The original V5/V8 streaming contract is a correctness reference: it stores
the full utterance and recomputes all previous frames for every input chunk.
That implementation is useful for causality tests but its work and memory grow
with stream duration.

`V8NativeStreamer` evaluates exactly one causal STFT frame at a time. It keeps:

- an analysis input ring;
- left-context caches for every causal convolution;
- the four-frame rolling noise context;
- convolution and SSM state for every temporal Mamba block;
- overlap-add numerator and denominator rings.

The state is independent of stream duration. The reference API remains
unchanged because V9--V11 have different frame graphs and require separately
validated native adapters.

## Correctness

`tests/test_v8_native_streaming.py` covers:

- arbitrary input chunk boundaries;
- utterances shorter than one analysis window;
- exact-window and multi-frame utterances;
- convolutional and Mamba temporal cores;
- bounded state over a long stream;
- flush and reset behaviour.

The native output matches whole-utterance inference to an absolute and relative
tolerance of `3e-5`. The full suite passes with 138 tests and 13
hardware-dependent skips.

## Reproducible benchmark

```bash
LD_LIBRARY_PATH=/home/mohamedb/miniconda3/envs/cnvqg/lib \
PYTHONPATH=src \
/home/mohamedb/miniconda3/envs/cnvqg/bin/python \
scripts/measure_latency.py \
  --checkpoint checkpoints/v8/v8_direct_scalar_full_scratch/best.pt \
  --warmup-seconds 1 \
  --seconds 10 \
  --output results/runtime/v8_native_streaming_7800x3d.json
```

The benchmark includes causal STFT analysis, the complete model, inverse STFT,
and overlap-add. It uses one PyTorch CPU thread.

| Measurement | Result |
|---|---:|
| Parameters | 1,067,471 |
| Algorithmic latency | 20 ms |
| Hop deadline | 10 ms |
| Frame time p50 | 8.93 ms |
| Frame time p95 | 10.52 ms |
| Frame time p99 | 10.80 ms |
| RTF | 0.917 |
| Persistent tensor state | 5.68 MiB |

The implementation is faster than real time on average, but it does not pass
the stricter per-hop p95 or p99 deadline. It must therefore be described as
near-real-time on this CPU rather than deadline-safe real-time.

## Profile and next experiment

On the same single-thread CPU, convolution accounts for roughly half of
self-CPU time. The temporal Mamba scan is the next largest component, including
matrix products, elementwise state updates, and exponentials.

The next controlled architecture experiment should compare:

1. the present V8 temporal Mamba;
2. a stateful GRU with matched parameter count;
3. time-kernel-one convolution and a reduced-width temporal core.

Each candidate should use this same native benchmark. Promotion requires a
quality gain or no statistically meaningful quality loss, p95 below the 10 ms
hop deadline, and enough headroom to remain below the deadline on a slower
target CPU.
