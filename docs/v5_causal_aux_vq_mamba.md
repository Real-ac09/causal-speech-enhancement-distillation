# V5 causal auxiliary-VQ Mamba

V5 is selected with:

```yaml
architecture: causal_aux_vq_mamba_v5
variant: student  # or teacher
```

The cap search selected 208 channels for the 1,041,297-parameter student and
336 channels for the 2,628,721-parameter teacher. The requested 128/208 initial
widths were below the useful parameter envelopes. Both models use a 320-sample
Hann analysis window, 160-sample hop, 512-point transform, one frequency
downsample, full-resolution skip, tied dual-axis refinement, continuous causal
noise conditioning, and separate magnitude/phase heads.

The `train_only` VQ path cannot affect enhanced audio. `bounded_adapter` is
zero-initialized, capped at 5%, and dropped on half of training branches. It is
promoted only by `gate_v5_vq_adapter.py` on the locked validation set.

## Streaming

```python
state = model.init_stream_state(batch_size, device, dtype)
audio, state = model.forward_chunk(chunk, state)
tail, state = model.flush(state)
```

The current implementation is a correctness reference: it releases only
overlap-complete samples and exactly matches whole-file causal inference for
arbitrary chunks. The runtime benchmark is a hard gate and is expected to
expose whether a cached/native deployment kernel is still required. The pure
PyTorch Mamba scan makes CPU correctness available even when the installed
`mamba-ssm` package is CUDA-only; it does not claim real-time performance.

## Programme

Run the architecture-selection sequence with:

```bash
bash scripts/run_v5_programme.sh
```

This creates the locked 400-file subset, runs the 1,000-example smoke test,
foundation and perceptual stages, calibrates the PESQ regressor, evaluates the
VQ adapter gate, distils the student, writes listening/SQUIM reports, and runs
one-thread runtime gates. It deliberately does not touch the 824-file test set.
Set `V5_RUN_PCS=1` to run the separately reported five-epoch PCS400 target
stage. Final test reporting is performed only after architecture selection and
must aggregate exactly three seeds with `aggregate_v5_seeds.py`.

Install the deployment extras in `requirements-optional.txt` before ONNX
export. `export_v5_onnx.py` uses a real-valued DFT implementation (avoiding
unsupported complex FFT operators), exports explicit audio-history state, and
rejects exports whose ONNX Runtime output differs by more than `1e-4`.
