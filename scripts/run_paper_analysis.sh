#!/usr/bin/env bash
# Reproduce saved observations, fits, tables, and figures into a fresh directory.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
RSCRIPT="${RSCRIPT:-Rscript}"
input_root="analysis/paper"
output_root=""
figure_output=""
raw=0
figures="global_expertise,global_delta_logprob,fact_usage,local_delta_logprob,context_anchor_diagnostics"
datasets="critical,control"
while (( $# )); do
    case "$1" in
        --output-root|--input-root|--figure-output-dir|--figures|--datasets)
            (( $# >= 2 )) || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --output-root) output_root="$2" ;;
                --input-root) input_root="$2" ;;
                --figure-output-dir) figure_output="$2" ;;
                --figures) figures="$2" ;;
                --datasets) datasets="$2" ;;
            esac
            shift 2 ;;
        --from-raw) raw=1; shift ;;
        --help)
            echo "bash scripts/run_paper_analysis.sh --output-root EMPTY_DIR [--from-raw | --input-root analysis/paper] [--figure-output-dir EMPTY_DIR] [--figures comma,separated] [--datasets critical,control]"
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$output_root" ]] || { echo "--output-root is required" >&2; exit 2; }
figure_output="${figure_output:-$output_root/figures}"
# Validate before starting any producer; existing results are never replaced.
"$PYTHON" - "$output_root" "$figure_output" <<'PY'
from pathlib import Path
import sys
output = Path(sys.argv[1]).resolve()
figures = Path(sys.argv[2]).resolve()
protected = [Path(p).resolve() for p in ("data", "analysis/paper")]
for path in (output, figures):
    if any(path.is_relative_to(p) or p.is_relative_to(path) for p in protected):
        raise SystemExit("Output cannot overlap data or canonical artifacts")
canonical_figures = Path("figures").resolve()
if output.is_relative_to(canonical_figures) or any(canonical_figures.is_relative_to(p) for p in (output, figures)):
    raise SystemExit("Output cannot overlap canonical artifacts")
if output.is_relative_to(figures):
    raise SystemExit("Figure output cannot contain the analysis output")
for path in (output, figures):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise SystemExit(f"Output must be a new or empty directory: {path}")
output.mkdir(parents=True, exist_ok=True)
PY
IFS=',' read -r -a selected_datasets <<< "$datasets"
if (( raw )); then
    input_root="$output_root/inputs"
    PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m src.analysis.prepare paper \
        --output-root "$input_root" --datasets "${selected_datasets[@]}"
fi
"$RSCRIPT" src/analysis/analyze.R --input-root "$input_root" \
    --output-root "$output_root/fits" --figures "$figures" --datasets "$datasets"
IFS=',' read -r -a selected_figures <<< "$figures"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m src.figures.plot \
    --analysis-root "$output_root/fits" --output-dir "$figure_output" \
    --figures "${selected_figures[@]}" --datasets "${selected_datasets[@]}"
if [[ ",$datasets," == *,critical,* ]]; then
    PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m src.analysis.prepare tables \
        --input-root "$input_root" --output-dir "$output_root/tables"
fi
if [[ ",$datasets," == *,critical,* && ",$datasets," == *,control,* ]]; then
    "$RSCRIPT" src/analysis/analyze.R gap --input-root "$input_root" --output-root "$output_root/gap"
fi
