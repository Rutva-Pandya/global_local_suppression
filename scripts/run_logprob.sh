#!/usr/bin/env bash
# Self-score local or global saved responses. Requires CUDA and model access.
# Example: bash scripts/run_logprob.sh --scope local \
#   --input data/critical/responses/llama8/local.json --output /path/to/new/local.csv
# Global scoring takes the merged global.json. The per-token JSON is written next
# to the CSV; neither output file may exist yet.
set -euo pipefail
[[ -d src/scoring ]] || cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"

echo "Started: $(date) | Python: $(command -v "$PYTHON") | Arguments: $*"
"$PYTHON" - <<'PY'
import importlib.metadata
import torch
import transformers

print(f"PyTorch {torch.__version__}, Transformers {transformers.__version__}, "
      f"minicons {importlib.metadata.version('minicons')}, GPUs: {torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
PY
"$PYTHON" -m src.scoring.logprob "$@"
echo "Finished: $(date)"
