#!/usr/bin/env bash
# Submit one model's generation and scoring, then analyze the four-model batch.
# Usage: bash scripts/run_pipeline.sh {llama8|llama70|qwen7|qwen72} {critical|control}
# Cluster account, partitions and resource overrides come from the environment.
set -euo pipefail

if [[ "${1:-}" == --help ]]; then
    echo "Usage: bash scripts/run_pipeline.sh {llama8|llama70|qwen7|qwen72} {critical|control}"
    exit 0
fi
stage=submit
if [[ "${1:-}" == --stage && $# == 4 ]]; then
    # Slurm copies this script; --chdir supplies the repository working directory.
    stage="$2"; alias="$3"; dataset="$4"
elif [[ $# == 2 ]]; then
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    alias="$1"; dataset="$2"
else
    echo "Expected a model alias and critical or control; see --help" >&2
    exit 2
fi
case "$alias" in llama8|llama70|qwen7|qwen72) ;; *) echo "Unknown model: $alias" >&2; exit 2 ;; esac
case "$dataset" in critical|control) ;; *) echo "Unknown dataset: $dataset" >&2; exit 2 ;; esac
PYTHON="${PYTHON:-python}"
batch="$PWD/logs/pipeline/$dataset"
folder="$batch/$alias"
responses="$PWD/data/$dataset/responses/$alias/rerun.json"
judgments="$PWD/data/$dataset/judgments/$alias/rerun.json"
scores="$PWD/data/$dataset/delta_logprob/$alias"
analysis="$PWD/analysis/rerun/$dataset"
figure_dir="$PWD/figures/rerun/$dataset"
stimuli="data/${dataset}_stimuli.csv"
model=$("$PYTHON" -c 'from src.analysis.prepare import MODEL_IDS; import sys; print(MODEL_IDS[sys.argv[1]])' "$alias")

check_batch() {
    "$PYTHON" - "$dataset" "$stage" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

dataset, stage = sys.argv[1:]
path = Path('logs/pipeline') / dataset / 'batch.json'
files = sorted(p for p in Path('src').rglob('*') if p.suffix in {'.py', '.R'})
files += sorted(Path('scripts').glob('run_*.sh'))
files += [Path('requirements.txt'), Path('data') / f'{dataset}_stimuli.csv']
identity = {
    'dataset': dataset,
    'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    'judge_weights': os.environ.get('VLLM_MODEL_PATH') or 'openai/gpt-oss-120b',
    'judge_revision': os.environ.get('VLLM_REVISION', ''),
}
if path.exists():
    if json.loads(path.read_text()) != identity:
        raise SystemExit(f'{path}: inputs, code or judge weights changed. Preserve this batch and its rerun outputs before starting a new batch; see README.md.')
elif stage == 'submit':
    data = Path('data') / dataset
    outputs = [Path('analysis/rerun') / dataset, Path('figures/rerun') / dataset]
    for model in ('llama8', 'llama70', 'qwen7', 'qwen72'):
        outputs += [data / kind / model / 'rerun.json' for kind in ('responses', 'judgments')]
        outputs += [data / 'delta_logprob' / model / f'rerun_{scope}.{suffix}'
                    for scope in ('local', 'global') for suffix in ('csv', 'json')]
    if any(p.exists() for p in outputs):
        raise SystemExit('Rerun outputs exist without a batch identity. Preserve and move them aside before starting a new batch; see README.md.')
    path.write_text(json.dumps(identity, indent=2) + '\n')
else:
    raise SystemExit(f'Missing batch identity: {path}')
PY
}

if [[ "$stage" != submit ]]; then
    check_batch
    case "$stage" in
        generate)
            bash scripts/run_model_responses.sh --model "$model" --stimuli "$stimuli" \
                --samples 20 --temperature 1 --top-p 1 --top-k 0 --seed 20260719 \
                --output "$responses" --resume
            ;;
        judge)
            # Each local judge server gets an available port on its own node.
            VLLM_PORT=$("$PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
            export VLLM_PORT
            TENSOR_PARALLEL_SIZE=2 VLLM_HOST=127.0.0.1 bash scripts/run_expertise_judge.sh \
                --stimuli "$stimuli" --model-outputs "$responses" \
                --output "$judgments" --run-name "${dataset}_${alias}_rerun" \
                --judge-model openai/gpt-oss-120b --samples 5 --temperature 1 --top-p 1 \
                --max-tokens 1024 --reasoning-effort low --resume
            "$PYTHON" - "$judgments" <<'PY'
import json
from pathlib import Path
import sys
if any(record['n_invalid'] for record in json.loads(Path(sys.argv[1]).read_text())):
    raise SystemExit('Invalid judge ratings remain; repeat the pipeline command to resume judging.')
PY
            ;;
        logprob)
            for scope in local global; do
                csv="$scores/rerun_$scope.csv"; json="$scores/rerun_$scope.json"
                if [[ -f "$csv" && -f "$json" ]]; then
                    "$PYTHON" - "$json" "$responses" <<'PY'
import hashlib
from pathlib import Path
import sys
from src.analysis.prepare import read_leading_metadata
metadata = read_leading_metadata(Path(sys.argv[1]))
if metadata['input_sha256'] != hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest():
    raise SystemExit('Existing logprob scores belong to different responses.')
PY
                elif [[ -e "$csv" || -e "$json" ]]; then
                    echo "Incomplete $scope scoring outputs in $scores; move them aside before retrying." >&2
                    exit 1
                else
                    bash scripts/run_logprob.sh --scope "$scope" \
                        --input "$responses" --output "$csv"
                fi
            done
            ;;
        analysis)
            judges=(); local_scores=(); global_scores=()
            for other in llama8 llama70 qwen7 qwen72; do
                [[ -f "$batch/$other/judge.done" && -f "$batch/$other/logprob.done" ]] || {
                    echo "Scoring is not complete for $other" >&2; exit 1;
                }
                judges+=("$PWD/data/$dataset/judgments/$other/rerun.json")
                local_scores+=("$PWD/data/$dataset/delta_logprob/$other/rerun_local.csv")
                global_scores+=("$PWD/data/$dataset/delta_logprob/$other/rerun_global.csv")
            done
            profile=actual; global_options=(--scope global)
            if [[ "$dataset" == control ]]; then profile=control; global_options+=(--control); fi
            global_root="$analysis/inputs/figure1/global_knowledge/expert_novice/$profile"
            local_root="$analysis/inputs/figure2/local_knowledge/use_ignore/$profile"
            anchors="$analysis/inputs/figure5/theoretical_baseline_anchors/expert_novice/$profile"
            if [[ -e "$analysis" || -e "$figure_dir" ]]; then
                echo "Existing analysis or figures for $dataset; preserve and move $analysis and $figure_dir aside before retrying analysis." >&2
                exit 1
            fi
            "$PYTHON" -m src.analysis.prepare observations --measurement expertise "${global_options[@]}" \
                --inputs "${judges[@]}" --output "$global_root/judge/observations.csv"
            "$PYTHON" -m src.analysis.prepare observations --measurement fact_usage --scope local \
                --inputs "${judges[@]}" --output "$local_root/judge/observations.csv"
            "$PYTHON" -m src.analysis.prepare observations --measurement logprob "${global_options[@]}" \
                --inputs "${global_scores[@]}" --output "$global_root/delta_logprob/observations.csv"
            "$PYTHON" -m src.analysis.prepare observations --measurement logprob --scope local \
                --inputs "${local_scores[@]}" --output "$local_root/delta_logprob/observations.csv"
            "$PYTHON" -m src.analysis.prepare observations --measurement anchors "${global_options[@]}" \
                --inputs "${global_scores[@]}" --output "$anchors/observations.csv"
            bash scripts/run_paper_analysis.sh --input-root "$analysis/inputs" \
                --output-root "$analysis/results" --figure-output-dir "$figure_dir" --datasets "$dataset"
            ;;
        *) echo "Unknown worker stage: $stage" >&2; exit 2 ;;
    esac
    if [[ "$stage" == analysis ]]; then touch "$batch/analysis.done"; else touch "$folder/$stage.done"; fi
    exit 0
fi

for command in sbatch squeue scancel; do
    command -v "$command" >/dev/null || { echo "Slurm command not found: $command" >&2; exit 1; }
done
mkdir -p "$folder" "$(dirname "$responses")" "$(dirname "$judgments")" "$scores"
if ! mkdir "$batch/.submission-lock" 2>/dev/null; then
    echo "Another submission is in progress in $batch; retry when it finishes." >&2
    exit 1
fi
trap 'rmdir "$batch/.submission-lock"' EXIT
check_batch

# A job file stores the ID and dependencies. Finished stages need no new job.
submit() {
    local task="$1" name="$2" dependencies="$3" gres="$4" memory="$5" limit="$6"
    local prefix="$batch/$name/$task" job_id old_dependencies state partition dependency changed
    if [[ "$task" == analysis ]]; then prefix="$batch/analysis"; fi
    if [[ -f "$prefix.done" ]]; then return; fi
    if [[ -f "$prefix.job" ]]; then
        read -r job_id old_dependencies < "$prefix.job"
        state=$(squeue --noheader --me --format='%i %T' | awk -v id="$job_id" '$1 == id {print $2}') || return
        if [[ -n "$state" ]]; then
            changed=0
            for dependency in ${dependencies//:/ }; do
                if [[ ":$old_dependencies:" != *":$dependency:"* ]]; then changed=1; fi
            done
            if (( changed == 0 )); then echo "$job_id"; return; fi
            if [[ "$state" != PENDING ]]; then
                echo "Job $job_id is $state with different dependencies; wait before retrying." >&2
                return 1
            fi
            scancel "$job_id" || return
        fi
    fi
    local options=(--parsable --chdir="$PWD" --nodes=1 --ntasks=1
        --cpus-per-task="${PIPELINE_CPUS:-4}" --mem="$memory" --time="$limit"
        --job-name="suppression-$dataset-$name-$task"
        --output="$batch/%x-%j.out" --error="$batch/%x-%j.err"
        --kill-on-invalid-dep=yes --export=ALL)
    partition="${SLURM_PARTITION:-}"
    if [[ "$task" == analysis ]]; then partition="${PIPELINE_CPU_PARTITION:-$partition}"; fi
    if [[ -n "${SLURM_ACCOUNT:-}" ]]; then options+=(--account="$SLURM_ACCOUNT"); fi
    if [[ -n "$partition" ]]; then options+=(--partition="$partition"); fi
    if [[ -n "$gres" ]]; then options+=(--gres="$gres"); fi
    if [[ -n "$dependencies" ]]; then options+=(--dependency="afterok:$dependencies"); fi
    job_id=$(sbatch "${options[@]}" "$PWD/scripts/run_pipeline.sh" --stage "$task" "$name" "$dataset") || return
    job_id="${job_id%%;*}"
    [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch job ID: $job_id" >&2; return 1; }
    printf '%s %s\n' "$job_id" "${dependencies:-none}" > "$prefix.job"
    echo "Submitted $task for $name: $job_id" >&2
    echo "$job_id"
}

gres="${PIPELINE_SMALL_GRES:-gpu:1}"; memory="${PIPELINE_SMALL_MEM:-64G}"
if [[ "$alias" == llama70 || "$alias" == qwen72 ]]; then
    gres="${PIPELINE_LARGE_GRES:-gpu:2}"; memory="${PIPELINE_LARGE_MEM:-128G}"
fi
generation=$(submit generate "$alias" "" "$gres" "$memory" "${PIPELINE_GENERATION_TIME:-24:00:00}")
submit judge "$alias" "$generation" "${PIPELINE_JUDGE_GRES:-gpu:2}" \
    "${PIPELINE_JUDGE_MEM:-128G}" "${PIPELINE_JUDGE_TIME:-24:00:00}" >/dev/null
submit logprob "$alias" "$generation" "$gres" "$memory" "${PIPELINE_LOGPROB_TIME:-24:00:00}" >/dev/null

dependencies=""; registered=0
for other in llama8 llama70 qwen7 qwen72; do
    if [[ ! -f "$batch/$other/judge.job" || ! -f "$batch/$other/logprob.job" ]]; then continue; fi
    registered=$((registered + 1))
    for task in judge logprob; do
        if [[ ! -f "$batch/$other/$task.done" ]]; then
            read -r job_id _ < "$batch/$other/$task.job"
            dependencies="${dependencies:+$dependencies:}$job_id"
        fi
    done
done
if (( registered == 4 )); then
    submit analysis "$alias" "$dependencies" "" "${PIPELINE_ANALYSIS_MEM:-16G}" \
        "${PIPELINE_ANALYSIS_TIME:-01:00:00}" >/dev/null
else
    echo "Registered $registered/4 models; analysis will be queued after all four are submitted."
fi
echo "Batch: $batch"
