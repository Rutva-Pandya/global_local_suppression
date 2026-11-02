# Simulating Ignorance

This repository contains code and saved data for *Simulating Ignorance:
Measuring Global and Local Knowledge Suppression in Language Models*,
by Rutva Pandya and Jennifer Hu.

The repository is structured as follows:

- `analysis`: per-scenario observations and statistical results
- `data`: stimuli, model responses, judge ratings and log-probability scores
- `figures`: rendered paper figures
- `scripts`: pipeline launchers and topic-pairing utilities
- `src`: response generation, scoring, analysis and plotting code
- `tests`: unit tests and paper reproduction checks

## 1. Setup

Use a Python 3.11 environment and R. Run commands from the repository root.

```bash
python -m pip install -r requirements.txt
Rscript -e 'install.packages(c("nlme", "digest"), repos="https://cloud.r-project.org")'
```

GPU workers also require CUDA and [vLLM](https://docs.vllm.ai/en/v0.23.0/getting_started/installation/gpu/):

```bash
python -m pip install vllm==0.23.0
```

The GPU smoke test used PyTorch 2.11.0, Transformers 5.12.1, minicons 0.3.38
and vLLM 0.23.0. Use compatible CUDA/PyTorch builds and authorize Hugging Face
access to the gated Llama models before running them.

## 2. Running new experiments

On a Slurm cluster, activate the worker environment and configure your account
and partitions once. Omit these settings to use scheduler defaults.

```bash
export SLURM_ACCOUNT=your_account
export SLURM_PARTITION=your_gpu_partition
export PIPELINE_CPU_PARTITION=your_cpu_partition
```

Submit a model and dataset:

```bash
bash scripts/run_pipeline.sh <MODEL> <DATASET>
```

`<MODEL>` is one of:

- `llama8`: Llama 3.1 8B Instruct
- `llama70`: Llama 3.1 70B Instruct
- `qwen7`: Qwen 2.5 7B Instruct
- `qwen72`: Qwen 2.5 72B Instruct

`<DATASET>` is `critical` or `control`. For example:

```bash
bash scripts/run_pipeline.sh qwen7 critical
```

The launcher generates responses, judges expertise and fact use with
GPT-OSS-120B, and computes local and global log probabilities. Run once per
model; analysis and figures run automatically after all four models finish
scoring. No separate analysis command is needed. The paper settings are 20
responses per stimulus and five judge ratings per response, with fixed
generator revisions.

Defaults assume 80 GB GPUs: one for small-model generation/scoring, two for
large models, and two for judging. Configure GPU types, memory and wall times
through the `PIPELINE_*` variables in `scripts/run_pipeline.sh`.
`PYTHON` and `RSCRIPT` can select interpreters.

Outputs use the existing folders; released paper files are preserved:

- Responses: `data/<dataset>/responses/<model>/rerun.json`
- Judgments: `data/<dataset>/judgments/<model>/rerun.json`
- Scores: `data/<dataset>/delta_logprob/<model>/rerun_{local,global}.{csv,json}`
- Observations: `analysis/rerun/<dataset>/inputs/`
- Fits and tables: `analysis/rerun/<dataset>/results/`
- Figures: `figures/rerun/<dataset>/`
- Job logs and tracking: `logs/pipeline/<dataset>/`

Critical batches export Tables 5–6. Between-dataset comparisons require both datasets.

Repeating the command reuses active jobs and completed stages. Before starting
a new batch, let its jobs finish and preserve and move aside its `rerun` data
files, analysis, figures and job-tracking folder listed above.

## 3. Data analysis

To reproduce the paper's analyses and figures, extract the saved data and run:

```bash
unzip data/data.zip -d data

bash scripts/run_paper_analysis.sh --from-raw --output-root analysis/reproduction \
  --figure-output-dir figures/reproduction
```

This rebuilds the observations, fits the statistical models, renders five
figures, exports Tables 5–6 and runs the supplemental gap analysis. Outputs
are saved in `analysis/reproduction/{inputs,fits,tables,gap}/`, with figures
in `figures/reproduction/`.

Use fresh output directories. `--from-raw` uses the saved judgments and logprob
scores for both datasets; it does not generate responses or rerun scoring.
Both scripts provide `--help` for options.

## 4. Tests

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
```

Paper reproduction checks require the saved data. Optional tokenizer checks
skip when their tokenizers are unavailable locally.
