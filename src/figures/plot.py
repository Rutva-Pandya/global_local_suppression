"""Render the paper's five result figures for critical, control, or both datasets.

Each figure plots every model's estimate and 95% CI for a few contrasts, read
from a fits run of src/analysis/analyze.R (--analysis-root) or, by default,
from the contrasts published under analysis/paper.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "global_local_suppression_cache"),
)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "global_local_suppression_matplotlib"),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MODEL_ORDER = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
)
MODEL_SHORT_LABELS = {
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "meta-llama/Llama-3.1-70B-Instruct": "Llama 3.1 70B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5 7B",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen 2.5 72B",
}
MODEL_MARKERS = {
    "meta-llama/Llama-3.1-8B-Instruct": "o",
    "meta-llama/Llama-3.1-70B-Instruct": "s",
    "Qwen/Qwen2.5-7B-Instruct": "^",
    "Qwen/Qwen2.5-72B-Instruct": "D",
}
SOURCE_COLORS = {"actual": "#2F80ED", "control": "#D62728"}
INK = "#202020"
MUTED_INK = "#777777"
GRID = "#E5E5E5"
JUDGE_MODEL = "openai/gpt-oss-120b"
SIGNIFICANCE_LABELS = {"", "*", "**", "***"}
REFERENCE_LINE_WIDTH = 1.2
# Every figure is a half-width panel of 2400 x 1350 pixels.
DPI = 600
WIDTH_PX = 2400
HEIGHT_PX = 1350
TEXT_SCALE = 1.5
ARTIST_SCALE = 1.3
LEGEND_SCALE = 1.1
X_AXIS_LABEL_PAD = 2
Y_AXIS_LABEL_PAD = 6
COMMON_COLUMNS = {
    "contrast_id",
    "evaluated_model",
    "model_label",
    "estimate",
    "ci95_low",
    "ci95_high",
    "significance",
    "n_scenarios",
    "n_models",
}
DELTA_LOGPROB_COLUMNS = {"scoring_model", "is_self_scored", "n_scoring_models"}
JUDGE_COLUMNS = {"judge_model", "judge_run", "n_judges"}
LOG_PROBABILITY_Y_AXIS_LABEL = "$\\Delta$ LP / # response tokens"
PAPER_METRICS = (
    "global_expertise",
    "global_delta_logprob",
    "fact_usage",
    "local_delta_logprob",
    "context_anchor_diagnostics",
)
# Where analysis/paper keeps each figure's published contrasts, per profile.
PUBLISHED_FOLDERS = {
    "global_expertise": "figure1/global_knowledge/expert_novice/{profile}/judge",
    "global_delta_logprob": "figure1/global_knowledge/expert_novice/{profile}/delta_logprob",
    "fact_usage": "figure2/local_knowledge/use_ignore/{profile}/judge",
    "local_delta_logprob": "figure2/local_knowledge/use_ignore/{profile}/delta_logprob",
    "context_anchor_diagnostics": "figure5/theoretical_baseline_anchors/expert_novice/{profile}",
}
FIGURE_SPECS = {
    "local_delta_logprob": {
        "delta_logprob": True,
        "point_contrasts": ("high_knowledge", "suppression"),
        "x_labels": ("High\nKnowledge", "Suppression"),
        "x_tick_fontsize": 5.6,
        "x_axis_label": "Knowledge Condition",
        "y_axis_label": LOG_PROBABILITY_Y_AXIS_LABEL,
        "y_axis_label_fontsize": 6,
        "y_axis_label_pad": 3,
        "y_limits": (-1.5, 2.75),
        "y_ticks": tuple((value / 2 for value in range(-3, 6))),
        "filename": "local_knowledge_results_logprob.png",
    },
    "context_anchor_diagnostics": {
        "delta_logprob": True,
        "point_contrasts": ("b_e", "b_n", "p_s"),
        "x_labels": ("High\nKnowledge", "Low\nKnowledge", "Suppression"),
        "x_tick_fontsize": 5.6,
        "x_axis_label": "Knowledge Condition",
        "y_axis_label": LOG_PROBABILITY_Y_AXIS_LABEL,
        "y_axis_label_fontsize": 6,
        "y_axis_label_pad": 3,
        "y_limits": (-3.2, 1.9),
        "y_ticks": (-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5),
        "filename": "global_knowledge_results_logprob_theoretical.png",
        "axes_left": 0.2,
        "axes_right": 0.73,
        "legend_scale": 0.95,
    },
    "global_delta_logprob": {
        "delta_logprob": True,
        "point_contrasts": ("high_knowledge", "low_knowledge", "suppression"),
        "x_labels": ("High\nKnowledge", "Low\nKnowledge", "Suppression"),
        "x_tick_fontsize": 5.6,
        "x_axis_label": "Knowledge Condition",
        "y_axis_label": LOG_PROBABILITY_Y_AXIS_LABEL,
        "y_axis_label_fontsize": 6,
        "y_axis_label_pad": 3,
        "y_limits": (-3.5, 2.0),
        "y_ticks": tuple((value / 2 for value in range(-7, 4))),
        "filename": "global_knowledge_results_logprob.png",
        "axes_left": 0.2,
        "axes_right": 0.73,
        "legend_scale": 0.95,
    },
    "fact_usage": {
        "delta_logprob": False,
        "point_contrasts": (
            "high_knowledge_minus_baseline",
            "suppression_minus_baseline",
        ),
        "x_labels": ("High\nKnowledge", "Suppression"),
        "x_tick_fontsize": 5.6,
        "x_axis_label": "Fact Condition",
        "y_axis_label": "$\\Delta$ Fact-use score\n(relative to Baseline)",
        "y_limits": (-1.5, 6.0),
        "y_ticks": (-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        "filename": "local_knowledge_results_judge.png",
    },
    "global_expertise": {
        "delta_logprob": False,
        "point_contrasts": (
            "high_knowledge_minus_baseline",
            "low_knowledge_minus_baseline",
            "suppression_minus_baseline",
        ),
        "x_labels": ("High\nKnowledge", "Low\nKnowledge", "Suppression"),
        "x_tick_fontsize": 6,
        "x_axis_label": "Persona Condition",
        "y_axis_label": "$\\Delta$ Expertise score\n(relative to Baseline)",
        "y_limits": (-2.0, 3.0),
        "y_ticks": (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
        "filename": "global_knowledge_results_judge.png",
        "axes_left": 0.2,
        "axes_right": 0.73,
        "legend_scale": 0.95,
    },
}


def read_contrasts(path: Path, extra_columns: Iterable[str]) -> list[dict[str, Any]]:
    """Read a contrasts.csv from src/analysis/analyze.R and check every row."""
    if not path.is_file():
        raise ValueError(f"Contrast file does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = COMMON_COLUMNS | set(extra_columns)
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Contrast file is empty: {path}")
    for row_number, row in enumerate(rows, start=2):
        status = row.get("status", "ok")
        if status not in {"ok", "descriptive_floor"}:
            raise ValueError(
                f"{path}:{row_number} has unsuccessful inference: {status}"
            )
        # A descriptive floor (no variance to fit) has an estimate but no inference.
        descriptive = status == "descriptive_floor"
        if descriptive and any(
            row.get(field, "") != ""
            for field in (
                "ci95_low",
                "ci95_high",
                "standard_error",
                "p_value",
                "significance",
            )
        ):
            raise ValueError(
                f"{path}:{row_number} gives inference for a descriptive floor"
            )
        numeric_fields = (
            ["estimate"] if descriptive else ["estimate", "ci95_low", "ci95_high"]
        )
        if descriptive:
            row["ci95_low"] = row["ci95_high"] = None
        for field in numeric_fields:
            try:
                row[field] = float(row[field])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}:{row_number} has invalid {field}: {row[field]!r}"
                ) from error
            if not math.isfinite(row[field]):
                raise ValueError(f"{path}:{row_number} has non-finite {field}")
        if (
            not descriptive
            and not row["ci95_low"] <= row["estimate"] <= row["ci95_high"]
        ):
            raise ValueError(f"{path}:{row_number} has an invalid confidence interval")
        if row["significance"] not in SIGNIFICANCE_LABELS:
            raise ValueError(
                f"{path}:{row_number} has invalid significance: {row['significance']!r}"
            )
    if {row["n_scenarios"] for row in rows} != {"25"}:
        raise ValueError(f"{path} must contain 25-scenario contrasts")
    if {row["n_models"] for row in rows} != {"4"}:
        raise ValueError(f"{path} must contain four-model contrasts")
    return rows


def validate_model_grid(
    rows: Sequence[Mapping[str, Any]], contrasts: Sequence[str]
) -> dict[tuple[str, str], Mapping[str, Any]]:
    """Index rows by (contrast, model), requiring every model for each drawn contrast."""
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["contrast_id"]), str(row["evaluated_model"]))
        if key in index:
            raise ValueError(f"Duplicate contrast row: {key}")
        index[key] = row
    expected = {(contrast, model) for contrast in contrasts for model in MODEL_ORDER}
    missing = sorted(expected - set(index))
    if missing:
        raise ValueError(
            f"Missing required contrast rows; first missing row: {missing[0]}"
        )
    return index


def load_delta_logprob_index(path: Path, contrasts: Sequence[str]) -> dict:
    """ΔLP contrasts, each model scored by itself."""
    rows = read_contrasts(path, DELTA_LOGPROB_COLUMNS)
    if {row["n_scoring_models"] for row in rows} != {"4"}:
        raise ValueError("Paper ΔLP figures require four self-scoring models")
    for row in rows:
        if (
            row["scoring_model"] != row["evaluated_model"]
            or str(row["is_self_scored"]).upper() != "TRUE"
        ):
            raise ValueError("Paper ΔLP figures require self-scored contrasts")
    return validate_model_grid(rows, contrasts)


def load_judge_index(path: Path, contrasts: Sequence[str]) -> dict:
    """Judge contrasts, rated by GPT-OSS-120B as the one judge."""
    rows = read_contrasts(path, JUDGE_COLUMNS)
    if {row["judge_model"] for row in rows} != {JUDGE_MODEL}:
        raise ValueError("Paper judge figures require GPT-OSS 120B contrasts")
    if {row["n_judges"] for row in rows} != {"1"}:
        raise ValueError("Paper judge figures require one judge model")
    return validate_model_grid(rows, contrasts)


def series_rows(
    index: Mapping[tuple[str, str], Mapping[str, Any]],
    contrasts: Sequence[str],
    model: str,
) -> list[Mapping[str, Any]]:
    return [index[contrast, model] for contrast in contrasts]


def configure_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(INK)
    axis.spines["bottom"].set_linewidth(0.9)
    axis.tick_params(axis="both", length=0)
    axis.tick_params(axis="y", colors=MUTED_INK, labelsize=6.5, pad=4)
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)


def draw_series(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    x_positions: Sequence[float],
    *,
    model: str,
    source: str,
) -> None:
    """One model's connected estimates, with 95% CIs where they were inferred."""
    color = SOURCE_COLORS[source]
    zorder = 2 if source == "control" else 3
    line = axis.plot(
        x_positions,
        [float(row["estimate"]) for row in rows],
        color=color,
        linestyle="-",
        linewidth=1.25 * ARTIST_SCALE,
        marker=MODEL_MARKERS[model],
        markersize=5.2 * ARTIST_SCALE,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.6 * ARTIST_SCALE,
        zorder=zorder,
    )[0]
    line.set_gid(f"{source}_series:{model}")
    inferred = [
        (position, row)
        for position, row in zip(x_positions, rows)
        if row.get("status", "ok") == "ok"
    ]
    if not inferred:
        return
    artists = axis.errorbar(
        [position for position, _ in inferred],
        [float(row["estimate"]) for _, row in inferred],
        yerr=[
            [float(row["estimate"]) - float(row["ci95_low"]) for _, row in inferred],
            [float(row["ci95_high"]) - float(row["estimate"]) for _, row in inferred],
        ],
        fmt="none",
        ecolor=color,
        elinewidth=0.9 * ARTIST_SCALE,
        capsize=2.2 * ARTIST_SCALE,
        capthick=0.9 * ARTIST_SCALE,
        zorder=zorder,
    )
    for cap in artists.lines[1]:
        cap.set_gid(f"{source}_confidence_interval_cap:{model}")
    for interval in artists.lines[2]:
        interval.set_gid(f"{source}_confidence_interval:{model}")


def render_comparison_figure(
    actual_index: Mapping[tuple[str, str], Mapping[str, Any]],
    control_index: Mapping[tuple[str, str], Mapping[str, Any]],
    spec: Mapping[str, Any],
    sources: Sequence[str] = ("control", "actual"),
) -> plt.Figure:
    point_contrasts = tuple(spec["point_contrasts"])
    figure = plt.figure()
    left = float(spec.get("axes_left", 0.22))
    right = float(spec.get("axes_right", 0.69))
    figure.subplots_adjust(left=left, right=right, bottom=0.25, top=0.95)
    axis = figure.add_subplot(1, 1, 1)
    base_positions = list(range(len(point_contrasts)))
    for model in MODEL_ORDER:
        for source in sources:
            index = control_index if source == "control" else actual_index
            # Nudge critical and control apart so their intervals do not overlap.
            offset = (
                (0.025 if source == "control" else -0.025) if len(sources) == 2 else 0
            )
            draw_series(
                axis,
                series_rows(index, point_contrasts, model),
                [position + offset for position in base_positions],
                model=model,
                source=source,
            )
    axis.axhline(0, color=INK, linewidth=REFERENCE_LINE_WIDTH * ARTIST_SCALE, zorder=1)
    axis.set_xlim(-0.25, len(point_contrasts) - 1 + 0.25)
    axis.set_ylim(*spec["y_limits"])
    axis.set_yticks(spec["y_ticks"])
    axis.set_xticks(
        base_positions,
        spec["x_labels"],
        fontsize=float(spec["x_tick_fontsize"]) * TEXT_SCALE,
        fontweight="normal",
    )
    axis.set_xlabel(
        spec["x_axis_label"],
        fontsize=7 * TEXT_SCALE,
        fontweight="normal",
        labelpad=X_AXIS_LABEL_PAD * ARTIST_SCALE,
    )
    axis.set_ylabel(
        spec["y_axis_label"],
        fontsize=float(spec.get("y_axis_label_fontsize", 7)) * TEXT_SCALE,
        fontweight="normal",
        labelpad=float(spec.get("y_axis_label_pad", Y_AXIS_LABEL_PAD)) * ARTIST_SCALE,
    )
    configure_axis(axis)
    axis.tick_params(axis="y", labelsize=6.5 * TEXT_SCALE, pad=4 * ARTIST_SCALE)
    axis.grid(axis="y", color=GRID, linewidth=0.6 * ARTIST_SCALE)
    axis.spines["bottom"].set_visible(False)
    model_handles = [
        Line2D(
            [0],
            [0],
            color=INK,
            marker=MODEL_MARKERS[model],
            linestyle="none",
            markerfacecolor=INK,
            markeredgecolor=INK,
            fillstyle="full",
            markersize=6 * ARTIST_SCALE,
            label=MODEL_SHORT_LABELS[model],
        )
        for model in MODEL_ORDER
    ]
    source_handles = [
        Line2D(
            [0],
            [0],
            color=SOURCE_COLORS[source],
            linewidth=1.8 * ARTIST_SCALE,
            marker=None,
            markersize=5.2 * ARTIST_SCALE,
            label=label,
        )
        for source, label in (("actual", "Critical"), ("control", "Unrelated control"))
        if source in sources
    ]
    figure.legend(
        handles=[*model_handles, *source_handles],
        loc="center left",
        bbox_to_anchor=(right + 0.003, 0.58),
        ncol=1,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
        labelspacing=0.9,
        fontsize=6.5 * float(spec.get("legend_scale", LEGEND_SCALE)),
    )
    return figure


def save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.set_size_inches(WIDTH_PX / DPI, HEIGHT_PX / DPI)
    figure.savefig(path, dpi=DPI, facecolor="white")
    plt.close(figure)


def contrast_paths(
    analysis_root: Path | None, figures: Sequence[str], profiles: Sequence[str]
) -> dict[str, Path]:
    """The contrasts.csv of each selected figure and profile, keyed "<figure>_<profile>"."""
    paths = {}
    for metric in PAPER_METRICS:
        if metric not in figures:
            continue
        for profile in profiles:
            if analysis_root is None:
                folder = Path("analysis/paper") / PUBLISHED_FOLDERS[metric].format(
                    profile=profile
                )
            else:
                folder = analysis_root / metric / profile
            paths[f"{metric}_{profile}"] = folder / "contrasts.csv"
    return paths


def generate_compact_figures(
    input_paths: Mapping[str, Path],
    output_dir: Path,
    figures: Sequence[str] = PAPER_METRICS,
    datasets: Sequence[str] = ("critical", "control"),
) -> list[Path]:
    profiles = tuple("actual" if name == "critical" else name for name in datasets)
    outputs = []
    for metric in figures:
        spec = FIGURE_SPECS[metric]
        loader = load_delta_logprob_index if spec["delta_logprob"] else load_judge_index
        indices = {
            profile: loader(input_paths[f"{metric}_{profile}"], spec["point_contrasts"])
            for profile in profiles
        }
        output_path = output_dir / spec["filename"]
        figure = render_comparison_figure(
            indices.get("actual", {}),
            indices.get("control", {}),
            spec,
            sources=profiles,
        )
        save_figure(figure, output_path)
        outputs.append(output_path)
    return outputs


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--analysis-root",
        type=Path,
        help="A fits run of src/analysis/analyze.R (default: the contrasts in analysis/paper).",
    )
    parser.add_argument(
        "--figures", nargs="+", choices=PAPER_METRICS, default=list(PAPER_METRICS)
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("critical", "control"),
        default=["critical", "control"],
    )
    args = parser.parse_args(arguments)
    if len(set(args.figures)) != len(args.figures) or len(set(args.datasets)) != len(
        args.datasets
    ):
        parser.error("--figures and --datasets must not repeat a value")
    return args


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_arguments(arguments)
    if (
        args.analysis_root is not None
        and not (args.analysis_root / "COMPLETE").is_file()
    ):
        raise ValueError("Unified analysis is incomplete")
    profiles = tuple("actual" if name == "critical" else name for name in args.datasets)
    input_paths = contrast_paths(args.analysis_root, args.figures, profiles)
    outputs = generate_compact_figures(
        input_paths, args.output_dir, args.figures, args.datasets
    )
    for path in outputs:
        print(path)
    metadata = {
        "figures": args.figures,
        "inputs": {
            name: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in input_paths.items()
        },
        "outputs": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs
        },
        "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "notes": [
            "Descriptive floor estimates have no CI or significance.",
            "Inspect interpretation_status in the result tables for any unresolved control wording.",
            "Critical-versus-control inference is saved in separate tables, not inferred from interval overlap.",
        ],
    }
    (args.output_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
