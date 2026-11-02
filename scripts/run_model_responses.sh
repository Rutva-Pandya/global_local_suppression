#!/usr/bin/env bash
# Generate model responses from one stimulus CSV into one explicit JSON file.
#
# Needs a GPU: one 80 GB GPU for the 7B/8B models; the 70B/72B models need at
# least two (about 140 GB of bfloat16 weights).
# Run inside the project's Python environment (requirements.txt); set PYTHON to
# use another interpreter. The Llama models are gated: log in with
# `huggingface-cli login` or set HF_TOKEN.
#
# Paper settings (model revisions are fixed in the generator and saved in every record):
#   bash scripts/run_model_responses.sh \
#     --model meta-llama/Llama-3.1-8B-Instruct \
#     --stimuli data/control_stimuli.csv \
#     --samples 20 --temperature 1 --top-p 1 --top-k 0 \
#     --seed 20260719 --output <responses.json>
# The output file must not exist yet; pass --resume to continue an interrupted
# run. --limit 1 runs only the first scenario's prompts; --conditions selects a subset.
# On a Slurm cluster, submit the same command from the repository root, e.g.
#   sbatch --gres=gpu:1 --mem=64G --time=4:00:00 scripts/run_model_responses.sh ...

set -euo pipefail
[[ -d src/generation ]] || cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"

echo "Started: $(date) | Python: $(command -v "$PYTHON") | Arguments: $*"
"$PYTHON" -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
"$PYTHON" -m src.generation.generate "$@"
echo "Finished: $(date)"
