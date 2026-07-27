# A Sub-Million-Parameter Causal Speech Enhancement Model with GRU Temporal Modelling and Privileged Distillation


## Main Goals

- Real-time speech enhancement
- CPU-friendly inference
- VoiceBank + DEMAND training and testing
- Evaluation using PESQ, STOI, ESTOI, SI-SDR, CSIG, CBAK and COVL
- Latency testing on CPU
- Final model release with simple inference support

## Project Structure

```text
configs/        Training and model configuration files
data/           Dataset folders, ignored by Git
src/cnvqg/      Main Python package
scripts/        Training, preprocessing, evaluation and inference scripts
experiments/    Experiment notes and logs
results/        Evaluation outputs, ignored by Git
checkpoints/    Saved models, ignored by Git
docs/           Dissertation/project documentation
hf_release/     Hugging Face release files

```
## Dataset

The main dataset is VoiceBank + DEMAND.

- Training: official training split
- Validation: speaker-level validation split from training data
- Testing: official test split only for final evaluation

No self-recorded audio is used.

## Planned Metrics

- PESQ
- STOI
- ESTOI
- SI-SDR
- CSIG
- CBAK
- COVL
- Real-time factor
- Milliseconds per frame
- CPU memory usage
- Model parameter count
