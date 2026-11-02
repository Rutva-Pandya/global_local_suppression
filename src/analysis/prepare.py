"""Prepare per-scenario observations and Tables 5-6 from saved judge and ΔLP scores.

* `paper` rebuilds the paper's ten observation files from data/critical and
  data/control, checking every input against the runs the paper used.
* `observations` prepares one measurement from explicitly named files, for
  example from a rerun of the generation, judge and logprob scripts.
* `tables` exports Tables 5 and 6 from observation files.

Judge ratings are averaged over the 5 ratings of each response, then over the
20 responses of each scenario; ΔLP is averaged over the 20 responses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

BASELINE = "baseline"
HIGH_KNOWLEDGE = "high_knowledge"
SUPPRESSION = "suppression"
LOW_KNOWLEDGE = "low_knowledge"
GLOBAL_CONDITION_ORDER = (BASELINE, HIGH_KNOWLEDGE, SUPPRESSION, LOW_KNOWLEDGE)
FACT_USAGE_CONDITIONS = (BASELINE, HIGH_KNOWLEDGE, SUPPRESSION)
# Local conditions keep their stimulus-file names in the judge records.
LOCAL_CONDITION_NAMES = {
    "fact_use_explicit": HIGH_KNOWLEDGE,
    "fact_suppression_explicit": SUPPRESSION,
}
EXPECTED_SCENARIOS = 25
RESPONSE_SAMPLES = 20
JUDGE_SAMPLES = 5
JUDGE_MODEL = "openai/gpt-oss-120b"
JUDGE_LABEL = "GPT-OSS 120B"
MODELS = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
)
MODEL_LABELS = {
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B Instruct",
    "meta-llama/Llama-3.1-70B-Instruct": "Llama 3.1 70B Instruct",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5 7B Instruct",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5 72B Instruct",
}
JUDGE_ID_COLUMNS = (
    "source_file",
    "stimulus_id",
    "scenario_id",
    "topic",
    "suppression_type",
    "evaluated_model",
    "condition",
)
JUDGE_SAMPLING_COLUMNS = (
    "response_regime",
    "response_sample_count",
    "judge_sample_count",
    "n_valid_ratings",
    "response_score_sd",
    "response_score_se",
    "judge_model",
    "judge_label",
    "judge_run",
    "sample_source",
    "response_base_seed",
    "model_output_source",
)
EXPERTISE_COLUMNS = (*JUDGE_ID_COLUMNS, "expertise_score", *JUDGE_SAMPLING_COLUMNS)
FACT_USAGE_COLUMNS = (*JUDGE_ID_COLUMNS, "fact_usage_score", *JUDGE_SAMPLING_COLUMNS)
# The ΔLP column of each response condition in the logprob CSVs.
LOCAL_LOGPROB_SCORES = {
    "fact_provided_response": "delta_logprob_high_mean",
    "fact_suppression_response": "delta_logprob_suppression_mean",
}
GLOBAL_LOGPROB_SCORES = {
    "expert_response": "delta_logprob_high_mean",
    "expert_to_novice_response": "delta_logprob_suppression_mean",
    "novice_response": "delta_logprob_low_mean",
}
LOGPROB_REQUIRED_COLUMNS = (
    "scenario_id",
    "sample_index",
    "model",
    "model_revision",
    "scoring_model",
    "scoring_model_revision",
)
SCORE_SOURCE_COLUMNS = (
    "source_file",
    "scenario_id",
    "evaluated_model",
    "model_label",
    "model_revision",
    "scoring_model",
    "scoring_model_label",
    "scoring_model_revision",
    "is_self_scored",
)
LOGPROB_COLUMNS = (
    *SCORE_SOURCE_COLUMNS,
    "response_condition",
    "source_metric",
    "source_log_base",
    "value_unit",
    "delta_logprob",
    "n_samples",
)
# Figure 5. B_E and B_N: how much more likely a response is after its own
# (high- or low-knowledge) prompt than after the baseline prompt, i.e. surprisal
# after the baseline prompt minus surprisal after its own. P_S: the suppression
# response's ΔLP.
ANCHOR_QUANTITIES = ("b_e", "p_s", "b_n")
ANCHOR_LABELS = {
    "b_e": "High-knowledge boost B_E(R_E)",
    "p_s": "Suppression comparison P_S",
    "b_n": "Low-knowledge boost B_N(R_N)",
}
ANCHOR_SURPRISAL_COLUMNS = {
    "b_e": ("surprisal_r_expert_c_base_mean", "surprisal_r_expert_c_expert_mean"),
    "b_n": ("surprisal_r_novice_c_base_mean", "surprisal_r_novice_c_novice_mean"),
}
# Written to the source_metric column of the paper's observation files.
ANCHOR_SOURCE_METRICS = {
    "b_e": "logP_r_expert_c_expert_mean - logP_r_expert_c_base_mean",
    "p_s": "persona_alignment_r_pretend_mean",
    "b_n": "logP_r_novice_c_novice_mean - logP_r_novice_c_base_mean",
}
ANCHOR_COLUMNS = (
    *SCORE_SOURCE_COLUMNS,
    "quantity",
    "quantity_label",
    "source_metric",
    "source_log_base",
    "value_unit",
    "quantity_value",
    "n_samples",
)
BITS_PER_TOKEN = "bits_per_scored_token"
ALIASES = ("llama8", "llama70", "qwen7", "qwen72")
MODEL_IDS = dict(zip(ALIASES, MODELS))
GLOBAL_ROOT = Path("figure1/global_knowledge/expert_novice")
LOCAL_ROOT = Path("figure2/local_knowledge/use_ignore")
ANCHOR_ROOT = Path("figure5/theoretical_baseline_anchors/expert_novice")
CRITICAL = Path("data/critical")
CONTROL = Path("data/control")
# Stimulus file of the paper's control low-knowledge responses.
CONTROL_STIMULI_SHA256 = (
    "aaf71fbb9594f05645bcf98feb68079bda7fc977caa3c005ae258e64fe12a8da"
)
# A control rerun's low-knowledge responses come from the released control stimuli.
CONTROL_STIMULI = Path("data/control_stimuli.csv")
TABLE_COLUMNS = ("evaluated_model", "condition", "n_scenarios", "mean", "sd", "mean_sd")
TABLES = (
    (
        "table5_global_expertise.csv",
        "figure1/global_knowledge/expert_novice/actual/judge/observations.csv",
        "expertise_score",
        GLOBAL_CONDITION_ORDER,
    ),
    (
        "table6_local_fact_usage.csv",
        "figure2/local_knowledge/use_ignore/actual/judge/observations.csv",
        "fact_usage_score",
        FACT_USAGE_CONDITIONS,
    ),
)


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def normalize_condition(suppression_type: str, condition: str) -> str:
    if suppression_type == "local":
        return LOCAL_CONDITION_NAMES.get(condition, condition)
    return condition


def read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"Input file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    if not all((isinstance(record, dict) for record in data)):
        raise ValueError(f"Expected every record in {path} to be an object")
    return data


def read_sources(paths: Iterable[Path]) -> list[tuple[Path, list[dict[str, Any]]]]:
    return [(path, read_json_list(path)) for path in paths]


def write_observations(
    path: Path, observations: list[dict], columns: Iterable[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(observations)


def sampled_judge_scores(
    record: dict[str, Any],
    path: Path,
    task: str,
    score_key: str,
) -> list[float]:
    if record.get("judge_task") != task:
        raise ValueError(f"Expected sampled {task} records in {path}")
    sampling = record.get("sampling", {})
    if (
        sampling.get("n") != JUDGE_SAMPLES
        or sampling.get("temperature") != 1.0
        or sampling.get("top_p") != 1.0
    ):
        raise ValueError(
            f"Unexpected sampled judge parameters for {record.get('stimulus_id')}"
        )
    if record.get("n_valid") != JUDGE_SAMPLES or record.get("n_invalid") != 0:
        raise ValueError(
            f"Expected {JUDGE_SAMPLES} valid judge ratings for {record.get('stimulus_id')} response sample {record.get('response_sample_index')}"
        )
    samples = record.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"Missing judge samples in {path}")
    indexed: dict[int, float] = {}
    for sample in samples:
        sample_index = sample.get("sample_index")
        if not isinstance(sample_index, int) or isinstance(sample_index, bool):
            raise ValueError(f"Invalid judge sample index in {path}: {sample_index}")
        if sample_index in indexed:
            raise ValueError(
                f"Duplicate judge sample index for {record.get('stimulus_id')}: {sample_index}"
            )
        value = sample.get(score_key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or (not math.isfinite(float(value)))
            or (not 1 <= float(value) <= 7)
        ):
            raise ValueError(
                f"Invalid {score_key} for {record.get('stimulus_id')}: {value}"
            )
        indexed[sample_index] = float(value)
    expected_indices = set(range(JUDGE_SAMPLES))
    if set(indexed) != expected_indices:
        raise ValueError(
            f"Unexpected judge sample indexes for {record.get('stimulus_id')}: {sorted(indexed)}"
        )
    scores = [indexed[index] for index in range(JUDGE_SAMPLES)]
    stored_mean = record.get(score_key + "_mean")
    calculated_mean = statistics.fmean(scores)
    if (
        not isinstance(stored_mean, (int, float))
        or isinstance(stored_mean, bool)
        or (not math.isclose(float(stored_mean), calculated_mean, abs_tol=1e-12))
    ):
        raise ValueError(
            f"Stored {score_key}_mean does not match the judge samples for {record.get('stimulus_id')} response sample {record.get('response_sample_index')}"
        )
    return scores


def aggregate_judgments(
    sources: Iterable[tuple[Path, list[dict[str, Any]]]],
    *,
    task: str,
    conditions: tuple[str, ...],
    suppression_type: str,
    grid_name: str,
    require_matching_condition_seeds: bool = True,
) -> list[dict[str, Any]]:
    """Average the ratings of each response, then the responses of each scenario.

    Uses the `task` records of the shared baseline and of `conditions` for
    `suppression_type`; everything else in `sources` is skipped. Returns one row
    per scenario, paper model and condition, and requires that complete grid.
    """
    score_key = "fact_usage" if task == "fact_usage" else "expertise_score"
    response_scores: defaultdict[tuple[str, str, str], dict[int, float]] = defaultdict(
        dict
    )
    response_seeds: defaultdict[tuple[str, str, int], set[int]] = defaultdict(set)
    metadata: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path, records in sources:
        for record in records:
            record_type = str(record.get("suppression_type", ""))
            condition = normalize_condition(
                record_type, str(record.get("condition", ""))
            )
            in_grid = (condition == BASELINE and record_type == "default_baseline") or (
                condition != BASELINE
                and condition in conditions
                and record_type == suppression_type
            )
            if record.get("judge_task") != task or not in_grid:
                continue
            if record.get("score_scale") != {"min": 1, "max": 7}:
                raise ValueError(f"Expected the 1-7 judge scale in {path}")
            judge_model = str(record.get("judge_model", ""))
            if judge_model != JUDGE_MODEL:
                raise ValueError(f"Unexpected sampled judge model in {path}")
            model = str(record.get("evaluated_model", ""))
            if model not in MODELS:
                raise ValueError(f"Unknown evaluated model in {path}: {model}")
            scenario = str(record.get("scenario_id", ""))
            generation = record.get("response_generation_params", {})
            if (
                generation.get("do_sample") is not True
                or generation.get("samples") != RESPONSE_SAMPLES
                or generation.get("temperature") != 1.0
                or generation.get("top_p") != 1.0
                or generation.get("top_k") != 0
                or not isinstance(generation.get("base_seed"), int)
                or isinstance(generation.get("base_seed"), bool)
            ):
                raise ValueError(
                    f"Unexpected response generation parameters for {record.get('stimulus_id')}"
                )
            response_index = record.get("response_sample_index")
            response_seed = record.get("response_sample_seed")
            if (
                not isinstance(response_index, int)
                or isinstance(response_index, bool)
                or response_index < 0
                or not isinstance(response_seed, int)
                or isinstance(response_seed, bool)
            ):
                raise ValueError(f"Invalid sampled response identity in {path}")
            cell = (scenario, model, condition)
            if response_index in response_scores[cell]:
                raise ValueError(
                    f"Duplicate sampled response for {cell}: {response_index}"
                )
            scores = sampled_judge_scores(record, path, task, score_key)
            response_scores[cell][response_index] = statistics.fmean(scores)
            response_seeds[scenario, model, response_index].add(response_seed)
            cell_metadata = {
                "source_file": str(path),
                "stimulus_id": str(record.get("stimulus_id", "")),
                "scenario_id": scenario,
                "topic": str(record.get("topic", "")),
                "suppression_type": record_type,
                "evaluated_model": model,
                "condition": condition,
                "judge_model": judge_model,
                "judge_label": JUDGE_LABEL,
                "judge_run": str(record.get("run_name", "")),
                "response_base_seed": generation["base_seed"],
                "model_output_source": str(record.get("model_output_source", "")),
            }
            if any(
                not str(value).strip()
                for name, value in cell_metadata.items()
                if name != "response_base_seed"
            ):
                raise ValueError(f"Blank sampled metadata for {cell}")
            if cell in metadata and metadata[cell] != cell_metadata:
                raise ValueError(f"Inconsistent sampled metadata for {cell}")
            metadata[cell] = cell_metadata
    scenarios = {cell[0] for cell in response_scores}
    if len(scenarios) != EXPECTED_SCENARIOS:
        raise ValueError(
            f"Expected {EXPECTED_SCENARIOS} scenarios, found {len(scenarios)}"
        )
    expected_cells = {
        (scenario, model, condition)
        for scenario in scenarios
        for model in MODELS
        for condition in conditions
    }
    if set(response_scores) != expected_cells:
        missing = sorted(expected_cells - set(response_scores))
        extra = sorted(set(response_scores) - expected_cells)
        raise ValueError(
            f"Incomplete sampled {grid_name} grid; missing={missing[:3]}, extra={extra[:3]}"
        )
    for cell, indexed in response_scores.items():
        if set(indexed) != set(range(RESPONSE_SAMPLES)):
            raise ValueError(
                f"Unexpected response sample indexes for {cell}: {sorted(indexed)}"
            )
    for key, seeds in response_seeds.items():
        if require_matching_condition_seeds and len(seeds) != 1:
            raise ValueError(f"Conditions use different response seeds for {key}")
    base_seeds = {row["response_base_seed"] for row in metadata.values()}
    if len(base_seeds) != 1:
        raise ValueError(f"Expected one sampled response base seed: {base_seeds}")
    observations = []
    for cell in expected_cells:
        response_means = [
            response_scores[cell][index] for index in range(RESPONSE_SAMPLES)
        ]
        response_score_sd = statistics.stdev(response_means)
        observations.append(
            {
                **metadata[cell],
                (
                    "fact_usage_score" if task == "fact_usage" else "expertise_score"
                ): statistics.fmean(response_means),
                "response_regime": "sampled",
                "response_sample_count": RESPONSE_SAMPLES,
                "judge_sample_count": JUDGE_SAMPLES,
                "n_valid_ratings": RESPONSE_SAMPLES * JUDGE_SAMPLES,
                "response_score_sd": response_score_sd,
                "response_score_se": response_score_sd / math.sqrt(RESPONSE_SAMPLES),
                "sample_source": f"sampled_responses_n{RESPONSE_SAMPLES}_judge_n{JUDGE_SAMPLES}",
            }
        )
    return sorted(
        observations,
        key=lambda row: (
            row["scenario_id"],
            MODELS.index(row["evaluated_model"]),
            conditions.index(row["condition"]),
        ),
    )


def extract_expertise(sources) -> list[dict[str, Any]]:
    """Global expertise: the shared baseline and the three global conditions."""
    return aggregate_judgments(
        sources,
        task="expertise",
        conditions=GLOBAL_CONDITION_ORDER,
        suppression_type="global",
        grid_name="global_expertise",
    )


def extract_fact_usage(
    sources, require_matching_condition_seeds: bool = True
) -> list[dict[str, Any]]:
    """Local fact use: the shared baseline and the fact-use and fact-ignore conditions."""
    return aggregate_judgments(
        sources,
        task="fact_usage",
        conditions=FACT_USAGE_CONDITIONS,
        suppression_type="local",
        grid_name="local_fact_usage",
        require_matching_condition_seeds=require_matching_condition_seeds,
    )


def read_leading_metadata(path: Path) -> dict[str, Any]:
    """Read a top-level metadata object without loading a large results array."""
    decoder = json.JSONDecoder()
    text = ""
    metadata_start: int | None = None
    with path.open(encoding="utf-8") as handle:
        while chunk := handle.read(65536):
            text += chunk
            if metadata_start is None:
                marker = '"metadata"'
                marker_index = text.find(marker)
                if marker_index < 0:
                    continue
                colon_index = text.find(":", marker_index + len(marker))
                if colon_index < 0:
                    continue
                metadata_start = colon_index + 1
            candidate = text[metadata_start:].lstrip()
            try:
                metadata, _ = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                if len(text) > 1024 * 1024:
                    break
                continue
            if not isinstance(metadata, dict):
                raise ValueError(f"Expected metadata object in {path}")
            return metadata
    raise ValueError(f"Could not read metadata object from {path}")


def check_logprob_units(path: Path) -> None:
    """A ΔLP CSV must hold mean log2 probabilities per scored token (see its JSON)."""
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        raise ValueError(f"Missing ΔLP metadata file: {metadata_path}")
    metadata = read_leading_metadata(metadata_path)
    reduction = metadata.get("primary_reduction")
    if reduction != "mean_per_scored_token":
        raise ValueError(
            f"Unexpected primary_reduction in {metadata_path}: {reduction!r}"
        )
    if metadata.get("log_base") not in (2, "2"):
        raise ValueError(
            f"Expected log_base 2 in {metadata_path}: {metadata.get('log_base')!r}"
        )


def finite_float(row: dict[str, str], column: str, path: Path) -> float:
    try:
        value = float(row[column])
    except ValueError as error:
        raise ValueError(f"Non-numeric {column} in {path}") from error
    require(math.isfinite(value), f"Non-finite {column} in {path}")
    return value


def only_value(rows: list[dict[str, str]], column: str, path: Path) -> str:
    values = {row[column] for row in rows}
    require(len(values) == 1, f"Expected one {column} value in {path}")
    return next(iter(values))


def read_logprob_scores(
    path: Path, columns: Iterable[str]
) -> tuple[dict[str, str], dict[str, list[dict[str, str]]]]:
    """The model identity and the rows, by scenario, of one ΔLP CSV from src.scoring.logprob.

    The file must hold one model and scorer, all scenarios, response samples
    0-19 of each, and every column in `columns`.
    """
    require(path.exists(), f"Input file does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(rows, f"Expected at least one row in {path}")
    missing = (set(LOGPROB_REQUIRED_COLUMNS) | set(columns)) - set(rows[0])
    require(not missing, f"Input file {path} is missing columns: {sorted(missing)}")
    check_logprob_units(path)
    identity = {
        column: only_value(rows, column, path)
        for column in (
            "model",
            "model_revision",
            "scoring_model",
            "scoring_model_revision",
        )
    }
    require(
        identity["model"] in MODELS,
        f"Unexpected response model in {path}: {identity['model']}",
    )
    require(
        identity["scoring_model"] in MODELS,
        f"Unexpected scoring model in {path}: {identity['scoring_model']}",
    )
    by_scenario: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_scenario[row["scenario_id"]].append(row)
    require(
        len(by_scenario) == EXPECTED_SCENARIOS,
        f"Expected {EXPECTED_SCENARIOS} scenarios in {path}, found {len(by_scenario)}",
    )
    for scenario, samples in by_scenario.items():
        try:
            indices = {int(row["sample_index"]) for row in samples}
        except ValueError as error:
            raise ValueError(f"Non-integer sample_index in {path}") from error
        require(
            len(indices) == len(samples), f"Duplicate scenario/sample row in {path}"
        )
        require(
            indices == set(range(RESPONSE_SAMPLES)),
            f"Expected sample indices 0-{RESPONSE_SAMPLES - 1} for {scenario} in {path}",
        )
    return identity, dict(sorted(by_scenario.items()))


def score_source(path: Path, scenario: str, identity: dict[str, str]) -> dict[str, Any]:
    """The source and model columns of ΔLP and Figure 5 observations."""
    return {
        "source_file": str(path),
        "scenario_id": scenario,
        "evaluated_model": identity["model"],
        "model_label": MODEL_LABELS[identity["model"]],
        "model_revision": identity["model_revision"],
        "scoring_model": identity["scoring_model"],
        "scoring_model_label": MODEL_LABELS[identity["scoring_model"]],
        "scoring_model_revision": identity["scoring_model_revision"],
        "is_self_scored": identity["model"] == identity["scoring_model"],
        "source_log_base": "2",
        "value_unit": BITS_PER_TOKEN,
    }


def aggregate_logprob(path: Path, scope: str) -> list[dict[str, Any]]:
    """Mean ΔLP over the samples of each scenario and response condition in one file."""
    scores = LOCAL_LOGPROB_SCORES if scope == "local" else GLOBAL_LOGPROB_SCORES
    identity, by_scenario = read_logprob_scores(path, scores.values())
    observations = []
    for scenario, samples in by_scenario.items():
        for condition, column in scores.items():
            values = [finite_float(row, column, path) for row in samples]
            observations.append(
                {
                    **score_source(path, scenario, identity),
                    "response_condition": condition,
                    "source_metric": column,
                    "delta_logprob": sum(values) / len(values),
                    "n_samples": len(values),
                }
            )
    return observations


def aggregate_anchors(path: Path) -> list[dict[str, Any]]:
    """Mean B_E, P_S and B_N (Figure 5) over the samples of each scenario in one global file."""
    suppression_column = GLOBAL_LOGPROB_SCORES["expert_to_novice_response"]
    surprisal_columns = [c for pair in ANCHOR_SURPRISAL_COLUMNS.values() for c in pair]
    identity, by_scenario = read_logprob_scores(
        path, [suppression_column, *surprisal_columns]
    )
    observations = []
    for scenario, samples in by_scenario.items():
        values = {
            "p_s": [finite_float(row, suppression_column, path) for row in samples]
        }
        for quantity, (base_column, own_column) in ANCHOR_SURPRISAL_COLUMNS.items():
            values[quantity] = [
                finite_float(row, base_column, path)
                - finite_float(row, own_column, path)
                for row in samples
            ]
        for quantity in ANCHOR_QUANTITIES:
            observations.append(
                {
                    **score_source(path, scenario, identity),
                    "quantity": quantity,
                    "quantity_label": ANCHOR_LABELS[quantity],
                    "source_metric": ANCHOR_SOURCE_METRICS[quantity],
                    "quantity_value": sum(values[quantity]) / len(values[quantity]),
                    "n_samples": len(samples),
                }
            )
    return observations


def collect_self_scored(
    paths: Iterable[Path], aggregate, rows_per_scenario: int
) -> list[dict]:
    """Aggregate one ΔLP file per paper model, each scored by the model itself."""
    observations = []
    seen = set()
    for path in paths:
        rows = aggregate(path)
        model, scorer = rows[0]["evaluated_model"], rows[0]["scoring_model"]
        require(
            (model, scorer) not in seen,
            f"Duplicate ΔLP file for {model} scored by {scorer}",
        )
        require(
            all(row["model_revision"] == row["scoring_model_revision"] for row in rows),
            f"Self-scoring revisions do not match in {path}",
        )
        seen.add((model, scorer))
        observations.extend(rows)
    require(
        seen == {(model, model) for model in MODELS},
        "Missing sampled model/scorer file",
    )
    expected = EXPECTED_SCENARIOS * len(MODELS) * rows_per_scenario
    require(
        len(observations) == expected,
        f"Expected {expected} observations, found {len(observations)}",
    )
    return observations


def extract_logprob(paths: Iterable[Path], scope: str) -> list[dict[str, Any]]:
    conditions = list(
        LOCAL_LOGPROB_SCORES if scope == "local" else GLOBAL_LOGPROB_SCORES
    )
    observations = collect_self_scored(
        paths, lambda path: aggregate_logprob(path, scope), len(conditions)
    )
    return sorted(
        observations,
        key=lambda row: (
            row["scenario_id"],
            MODELS.index(row["evaluated_model"]),
            conditions.index(row["response_condition"]),
        ),
    )


def extract_anchors(paths: Iterable[Path]) -> list[dict[str, Any]]:
    observations = collect_self_scored(paths, aggregate_anchors, len(ANCHOR_QUANTITIES))
    return sorted(
        observations,
        key=lambda row: (
            row["scenario_id"],
            MODELS.index(row["evaluated_model"]),
            ANCHOR_QUANTITIES.index(row["quantity"]),
        ),
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def judgment_path(profile: str, kind: str, alias: str) -> Path:
    root = CRITICAL if profile == "actual" else CONTROL
    return root / "judgments" / alias / f"{kind}.json"


def judge_rules(profile: str, alias: str) -> dict[str, dict]:
    """Expected (run, wording, stimuli hash) per (judge task, condition) in each file."""
    baseline_run = ("matched_small-run_gpt_oss_120b_n5", None, None)
    if profile == "actual":
        small = alias in {"llama8", "qwen7"}
        high = (
            (
                f"global_one_token_expert_novice_{alias}_response_n20_judge_n5",
                "global_one_token_expert_novice",
                None,
            )
            if small
            else (
                "actual_global_expert_novice_llama70_qwen72_n5",
                "expert_novice",
                None,
            )
        )
        size = "small" if small else "large"
        low = (
            f"actual_global_expert_novice_low_knowledge_{size}_models_n5",
            "expert_novice",
            None,
        )
        local = (
            "wording_all_models_all_versions_gpt_oss_120b_n5",
            "v1_use_ignore",
            None,
        )
        baseline_fact = baseline_run
    else:
        high = (
            "control_global_expert_novice_llama8_qwen7_llama70_qwen72_n5",
            "expert_novice",
            None,
        )
        low = (
            "control_global_expert_novice_low_refresh_20261028_n5",
            None,
            CONTROL_STIMULI_SHA256,
        )
        local = (
            "control_local_use_ignore_llama8_qwen7_llama70_qwen72_n5",
            "use_ignore",
            None,
        )
        baseline_fact = ("control_local_donor_baseline_all_models_n5", None, None)
    return {
        "baseline": {
            ("expertise", "baseline"): baseline_run,
            ("fact_usage", "baseline"): baseline_fact,
        },
        "global": {
            ("expertise", "high_knowledge"): high,
            ("expertise", "suppression"): high,
            ("expertise", "low_knowledge"): low,
        },
        "local": {
            ("fact_usage", "fact_use_explicit"): local,
            ("fact_usage", "fact_suppression_explicit"): local,
        },
    }


def validate_judge_source(path: Path, rules: dict, model: str | None = None) -> None:
    for record in read_json_list(path):
        key = (record.get("judge_task"), record.get("condition"))
        if key not in rules:
            raise ValueError(f"Unexpected judge record {key} in {path}")
        if model is not None and record.get("evaluated_model") != model:
            raise ValueError(f"Unexpected evaluated model in {path}")
        run, wording, stimuli_sha256 = rules[key]
        if record.get("run_name") != run:
            raise ValueError(f"Unexpected judge run in {path}")
        if wording is not None and record.get("wording_id") != wording:
            raise ValueError(f"Unexpected wording in {path}")
        if record.get("sampling", {}).get("reasoning_effort") != "low":
            raise ValueError(f"Expected low-effort judging in {path}")
        if (
            stimuli_sha256 is not None
            and record.get("stimuli_sha256") != stimuli_sha256
        ):
            raise ValueError(f"Unverified control novice stimuli in {path}")


def validate_judge_inputs(profiles=("actual", "control")) -> None:
    for profile in profiles:
        for alias in ALIASES:
            for kind, rules in judge_rules(profile, alias).items():
                validate_judge_source(
                    judgment_path(profile, kind, alias), rules, MODEL_IDS[alias]
                )


def select_fact_records(records: list[dict], selection: str) -> list[dict]:
    """The paper's fact-use records: the shared baseline, or one dataset's local wording."""
    wordings = {"actual": "v1_use_ignore", "control": "use_ignore"}
    if selection != "baseline" and selection not in wordings:
        raise ValueError(f"Unknown fact-use selection: {selection}")
    selected = []
    for record in records:
        if record.get("judge_task") != "fact_usage":
            continue
        record_type = record["suppression_type"]
        condition = normalize_condition(record_type, record["condition"])
        if selection == "baseline":
            keep = record_type == "default_baseline" and condition == BASELINE
        else:
            keep = (
                record_type == "local"
                and condition in (HIGH_KNOWLEDGE, SUPPRESSION)
                and record.get("wording_id") == wordings[selection]
            )
        if keep:
            selected.append(record)
    return selected


def responses_sha256(path: Path, conditions: set[str] | None = None) -> str:
    """Hash of the original response file whose records this file holds.

    Merged files keep each original file's records unchanged and in order, so
    re-serializing the selected conditions reproduces that file's bytes.
    """
    if conditions is None:
        return sha256(path)
    records = [r for r in read_json_list(path) if r["condition"] in conditions]
    encoded = (json.dumps(records, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_lp_source(
    path: Path,
    response: Path,
    conditions: set[str] | None = None,
    low_conditions: set[str] | None = None,
) -> None:
    metadata = read_leading_metadata(path.with_suffix(".json"))
    if metadata.get("matched_low_input_sha256") is None:
        # Scored from the whole response file, as src.scoring.logprob does.
        inputs = [("input_sha256", None)]
    else:
        # The paper's global scores came from separate high/suppression and
        # low-knowledge files; their hashes are re-derived from the merged file.
        inputs = [("input_sha256", conditions)]
        if low_conditions is not None:
            inputs.append(("matched_low_input_sha256", low_conditions))
    for digest, selected in inputs:
        if metadata.get(digest) != responses_sha256(response, selected):
            raise ValueError(f"LP response source/hash mismatch in {path}: {digest}")


def lp_paths(profile: str, scope: str) -> list[Path]:
    root = CRITICAL if profile == "actual" else CONTROL
    paths = []
    for alias in ALIASES:
        path = root / "delta_logprob" / alias / f"{scope}.csv"
        response = root / "responses" / alias / f"{scope}.json"
        if scope == "global":
            if profile == "control" and any(
                (
                    r.get("stimuli_sha256") != CONTROL_STIMULI_SHA256
                    for r in read_json_list(response)
                    if r["condition"] == "low_knowledge"
                )
            ):
                raise ValueError(f"Unverified control novice responses in {response}")
            validate_lp_source(
                path, response, {"high_knowledge", "suppression"}, {"low_knowledge"}
            )
        else:
            validate_lp_source(path, response)
        paths.append(path)
    return paths


def prepare_paper(output_root: Path, datasets=("critical", "control")) -> list[Path]:
    output_root = output_root.resolve()
    protected = (
        Path("analysis/paper").resolve(),
        Path("figures").resolve(),
    )
    if any(
        (
            output_root.is_relative_to(path) or path.is_relative_to(output_root)
            for path in protected
        )
    ):
        raise ValueError("Output cannot overlap canonical paper artifacts")
    if output_root.exists() and (
        not output_root.is_dir() or any(output_root.iterdir())
    ):
        raise ValueError("Output must be a new or empty directory")
    if (
        not datasets
        or len(set(datasets)) != len(datasets)
        or set(datasets) - {"critical", "control"}
    ):
        raise ValueError("Select critical, control, or both datasets")
    profiles = tuple("actual" if name == "critical" else name for name in datasets)
    validate_judge_inputs(profiles)
    tables = []
    for profile in profiles:
        baselines = [judgment_path(profile, "baseline", alias) for alias in ALIASES]
        global_judge = extract_expertise(
            read_sources(
                baselines
                + [judgment_path(profile, "global", alias) for alias in ALIASES]
            )
        )
        sources = [
            (path, select_fact_records(read_json_list(path), "baseline"))
            for path in baselines
        ]
        for alias in ALIASES:
            path = judgment_path(profile, "local", alias)
            sources.append((path, select_fact_records(read_json_list(path), profile)))
        # The critical local responses were not sampled with the baseline's seeds.
        local_judge = extract_fact_usage(
            sources, require_matching_condition_seeds=False
        )
        global_paths = lp_paths(profile, "global")
        quantities = extract_anchors(global_paths)
        global_scores = extract_logprob(global_paths, "global")
        local_scores = extract_logprob(lp_paths(profile, "local"), "local")
        groups = (
            (GLOBAL_ROOT / profile / "judge", global_judge, EXPERTISE_COLUMNS),
            (LOCAL_ROOT / profile / "judge", local_judge, FACT_USAGE_COLUMNS),
            (GLOBAL_ROOT / profile / "delta_logprob", global_scores, LOGPROB_COLUMNS),
            (LOCAL_ROOT / profile / "delta_logprob", local_scores, LOGPROB_COLUMNS),
            (ANCHOR_ROOT / profile, quantities, ANCHOR_COLUMNS),
        )
        for folder, rows, columns in groups:
            if profile == "control" and (
                folder.is_relative_to(GLOBAL_ROOT) or folder.is_relative_to(ANCHOR_ROOT)
            ):
                rows, columns = mark_novice_verified(rows, columns)
            tables.append((output_root / folder / "observations.csv", rows, columns))
    for path, rows, columns in tables:
        write_observations(path, rows, columns)
    return [path for path, _, _ in tables]


def mark_novice_verified(rows: list[dict], columns: tuple[str, ...]):
    """Mark control global tables as using the verified true-novice low-knowledge wording.

    The R fitter reads this column; without it, low-knowledge control contrasts
    are labelled unresolved and the critical-control gap analysis refuses to run.
    """
    status = "true_novice_verified"
    return [{**row, "novice_wording_status": status} for row in rows], (
        *columns,
        "novice_wording_status",
    )


def low_knowledge_stimuli_hashes(
    measurement: str, inputs: list[Path], sources: list | None
) -> set[str | None]:
    """The stimulus-file hashes recorded for the low-knowledge responses behind the inputs."""
    if measurement == "expertise":
        return {
            record.get("stimuli_sha256")
            for _path, records in sources
            for record in records
            if record.get("judge_task") == "expertise"
            and record.get("suppression_type") == "global"
            and record.get("condition") == LOW_KNOWLEDGE
        }
    hashes = set()
    for path in inputs:
        with path.open(newline="", encoding="utf-8") as handle:
            # low_knowledge_* provenance is filled only where it differs from the row's own.
            hashes |= {
                row.get("low_knowledge_stimuli_sha256") or row.get("stimuli_sha256")
                for row in csv.DictReader(handle)
            }
    return hashes


def prepare_observations(args: argparse.Namespace) -> Path:
    """One measurement from explicitly named judge or ΔLP files, e.g. from a rerun."""
    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}")
    if args.control:
        # Only these tables have a low-knowledge condition.
        require(
            args.scope == "global" and args.measurement != "fact_usage",
            "--control applies to global expertise, global logprob and anchors",
        )
    sources = None
    if args.measurement == "expertise":
        require(args.scope == "global", "Expertise observations require --scope global")
        sources = read_sources(args.inputs)
        rows, columns = extract_expertise(sources), EXPERTISE_COLUMNS
    elif args.measurement == "fact_usage":
        require(args.scope == "local", "Fact usage requires --scope local")
        rows, columns = (
            extract_fact_usage(read_sources(args.inputs)),
            FACT_USAGE_COLUMNS,
        )
    elif args.measurement == "anchors":
        require(args.scope == "global", "Context anchors require --scope global")
        rows, columns = extract_anchors(args.inputs), ANCHOR_COLUMNS
    else:
        rows, columns = extract_logprob(args.inputs, args.scope), LOGPROB_COLUMNS
    if args.control:
        verified = {sha256(CONTROL_STIMULI), CONTROL_STIMULI_SHA256}
        hashes = low_knowledge_stimuli_hashes(args.measurement, args.inputs, sources)
        require(
            hashes and hashes <= verified,
            f"--control needs low-knowledge responses generated from {CONTROL_STIMULI}",
        )
        rows, columns = mark_novice_verified(rows, columns)
    write_observations(args.output, rows, columns)
    return args.output


def summary_row(model: str, condition: str, values: list[float]) -> dict:
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    return {
        "evaluated_model": model,
        "condition": condition,
        "n_scenarios": len(values),
        "mean": mean,
        "sd": sd,
        "mean_sd": f"{mean:.2f} ± {sd:.2f}",
    }


def summarize_table(
    path: Path, score_column: str, conditions: tuple[str, ...]
) -> list[dict]:
    values = defaultdict(dict)
    scenarios = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"scenario_id", "evaluated_model", "condition", score_column}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Missing observation columns in {path}")
        for row in reader:
            scenario = row["scenario_id"]
            model, condition = (row["evaluated_model"], row["condition"])
            if (
                not scenario.strip()
                or model not in MODELS
                or condition not in conditions
            ):
                raise ValueError(f"Unexpected scenario/model/condition in {path}")
            value = float(row[score_column])
            if not math.isfinite(value) or not 1 <= value <= 7:
                raise ValueError(
                    f"Judge scores must be finite and within 1-7 in {path}"
                )
            cell = values[model, condition]
            if scenario in cell:
                raise ValueError(f"Duplicate scenario/model/condition in {path}")
            cell[scenario] = value
            scenarios.add(scenario)
    expected = {(model, condition) for model in MODELS for condition in conditions}
    if (
        len(scenarios) != EXPECTED_SCENARIOS
        or set(values) != expected
        or any((set(cell) != scenarios for cell in values.values()))
    ):
        raise ValueError(
            f"Expected a complete 25-scenario/model/condition grid in {path}"
        )
    ordered_scenarios = sorted(scenarios)
    rows = [
        summary_row(
            model, condition, [values[model, condition][s] for s in ordered_scenarios]
        )
        for model in MODELS
        for condition in conditions
    ]
    for condition in conditions:
        scenario_means = [
            statistics.fmean((values[model, condition][s] for model in MODELS))
            for s in ordered_scenarios
        ]
        rows.append(summary_row("mean_across_models", condition, scenario_means))
    return rows


def export_tables(input_root: Path, output_dir: Path) -> list[Path]:
    input_root, output_dir = (input_root.resolve(), output_dir.resolve())
    protected = (
        input_root,
        Path("analysis/paper").resolve(),
        Path("figures").resolve(),
    )
    if any(
        (
            output_dir.is_relative_to(p) or p.is_relative_to(output_dir)
            for p in protected
        )
    ):
        raise ValueError("Output cannot overlap input or canonical paper artifacts")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("Output must be a new or empty directory")
    tables = [
        (
            output_dir / filename,
            summarize_table(input_root / relative, score, conditions),
        )
        for filename, relative, score, conditions in TABLES
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, rows in tables:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=TABLE_COLUMNS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    return [path for path, _ in tables]


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    paper = commands.add_parser(
        "paper", help="Rebuild paper observations from saved raw inputs"
    )
    paper.add_argument("--output-root", type=Path, required=True)
    paper.add_argument(
        "--datasets",
        nargs="+",
        choices=("critical", "control"),
        default=["critical", "control"],
    )
    tables = commands.add_parser(
        "tables", help="Export Tables 5 and 6 from observations"
    )
    tables.add_argument("--input-root", type=Path, default=Path("analysis/paper"))
    tables.add_argument("--output-dir", type=Path, required=True)
    observations = commands.add_parser(
        "observations", help="Prepare one measurement from explicit inputs"
    )
    observations.add_argument("--inputs", type=Path, nargs="+", required=True)
    observations.add_argument(
        "--measurement",
        choices=("expertise", "fact_usage", "logprob", "anchors"),
        required=True,
    )
    observations.add_argument("--scope", choices=("local", "global"), required=True)
    observations.add_argument("--output", type=Path, required=True)
    observations.add_argument(
        "--control",
        action="store_true",
        help="The inputs are a control run: check that its low-knowledge responses "
        f"came from {CONTROL_STIMULI} and mark the table as verified true-novice wording.",
    )
    return parser.parse_args(arguments)


def main():
    args = parse_args()
    if args.command == "paper":
        paths = prepare_paper(args.output_root, args.datasets)
    elif args.command == "tables":
        paths = export_tables(args.input_root, args.output_dir)
    else:
        paths = [prepare_observations(args)]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
