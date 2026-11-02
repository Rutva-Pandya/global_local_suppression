#!/usr/bin/env bash
# Judge saved responses with GPT-OSS-120B (expertise or fact use, on a 1-7 scale).
# Starts a local vLLM server for the judge model, waits until it is ready, runs
# the judge against it and stops the server on exit.
#
# Needs two 80 GB GPUs (tensor parallel size 2) and vLLM in the project's Python
# environment; set PYTHON to use another interpreter. Optional settings:
#   JUDGE_MODEL (openai/gpt-oss-120b), VLLM_MODEL_PATH (a local copy of the
#   weights), VLLM_REVISION, VLLM_HOST (127.0.0.1), VLLM_PORT (8000),
#   TENSOR_PARALLEL_SIZE (2), MAX_MODEL_LEN (4096), GPU_MEMORY_UTILIZATION (0.90),
#   VLLM_STARTUP_TIMEOUT_SECONDS (900).
# vLLM server logs are written to logs/.
#
# Paper settings (default stimuli: data/critical_stimuli.csv):
#   bash scripts/run_expertise_judge.sh \
#     --model-outputs <responses.json> --output <judgments.json> \
#     --run-name <run-name> --samples 5 --temperature 1 --top-p 1 \
#     --reasoning-effort low
# For control responses add --stimuli data/control_stimuli.csv.
# The output file must not exist yet; pass --resume to continue an interrupted run.
# On a Slurm cluster, submit the same command from the repository root, e.g.
#   sbatch --gres=gpu:2 --mem=128G --time=12:00:00 scripts/run_expertise_judge.sh ...

set -euo pipefail
[[ -d src/scoring ]] || cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"

ARGS=("$@")
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-oss-120b}"
HAS_JUDGE_MODEL=0
HAS_API_BASE=0
for ((i = 0; i < ${#ARGS[@]}; i++)); do
    case "${ARGS[$i]}" in
        --judge-model)
            HAS_JUDGE_MODEL=1
            if (( i + 1 >= ${#ARGS[@]} )); then
                echo "Missing value for --judge-model" >&2
                exit 2
            fi
            JUDGE_MODEL="${ARGS[$((i + 1))]}"
            ;;
        --judge-model=*)
            HAS_JUDGE_MODEL=1
            JUDGE_MODEL="${ARGS[$i]#--judge-model=}"
            ;;
        --api-base|--api-base=*)
            HAS_API_BASE=1
            ;;
    esac
done

VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-$JUDGE_MODEL}"
VLLM_REVISION="${VLLM_REVISION:-}"
if [[ "$VLLM_MODEL_PATH" == /* && ! -d "$VLLM_MODEL_PATH" ]]; then
    echo "Local vLLM model path does not exist: $VLLM_MODEL_PATH" >&2
    exit 2
fi
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_STARTUP_TIMEOUT_SECONDS="${VLLM_STARTUP_TIMEOUT_SECONDS:-900}"
VLLM_API_BASE="http://${VLLM_HOST}:${VLLM_PORT}/v1"
mkdir -p logs
VLLM_LOG="logs/vllm_judge_$(date +%Y%m%d_%H%M%S)_$$"

echo "Started: $(date) | Python: $(command -v "$PYTHON")"
echo "Judge model: $JUDGE_MODEL (weights: $VLLM_MODEL_PATH, revision: ${VLLM_REVISION:-default})"
echo "vLLM API base: $VLLM_API_BASE | tensor parallel size: $TENSOR_PARALLEL_SIZE"
"$PYTHON" - <<'PY'
import torch
import vllm

print(f"PyTorch {torch.__version__}, vLLM {vllm.__version__}, GPUs: {torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; refusing to start vLLM")
PY

VLLM_ARGS=(
    --model "$VLLM_MODEL_PATH"
    --served-model-name "$JUDGE_MODEL"
    --host "$VLLM_HOST"
    --port "$VLLM_PORT"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
)
if [[ -n "$VLLM_REVISION" ]]; then
    VLLM_ARGS+=(--revision "$VLLM_REVISION")
fi

"$PYTHON" -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" \
    > "$VLLM_LOG.out" 2> "$VLLM_LOG.err" &
VLLM_PID=$!

cleanup() {
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM server (pid $VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Waiting for vLLM to become ready..."
deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT_SECONDS))
until "$PYTHON" - "$VLLM_API_BASE/models" <<'PY'
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=5) as response:
        raise SystemExit(0 if 200 <= response.status < 300 else 1)
except Exception:
    raise SystemExit(1)
PY
do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited before becoming ready. See $VLLM_LOG.out and $VLLM_LOG.err" >&2
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "Timed out waiting for vLLM. See $VLLM_LOG.out and $VLLM_LOG.err" >&2
        exit 1
    fi
    sleep 10
done
echo "vLLM is ready."

PYTHON_ARGS=("${ARGS[@]}")
if (( HAS_JUDGE_MODEL == 0 )); then
    PYTHON_ARGS+=(--judge-model "$JUDGE_MODEL")
fi
if (( HAS_API_BASE == 0 )); then
    PYTHON_ARGS+=(--api-base "$VLLM_API_BASE")
fi
"$PYTHON" -m src.scoring.judge "${PYTHON_ARGS[@]}"
echo "Finished: $(date)"
