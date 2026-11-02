#!/usr/bin/env python
"""Validate relevance ratings and select a one-to-one irrelevant-control pairing.

Gemini rated every directed pair of the 25 scenarios (600 pairs, 5 samples each)
for persona, fact and general relevance. `validate-results` checks the downloaded
batch; `select` keeps donors whose every mean relevance is at most the threshold
and finds a one-to-one pairing with the least total relevance.

Many pairings tie at that minimum: most pairs were rated 1 on every scale in
every sample. Ties go to the paper's pairs (Table 3, read from
data/control_stimuli.csv), so `select` reproduces Table 3 and shows that it is a
minimum-relevance pairing; without that preference it returns another tied pairing.

The rating requests in data/cross-scenario_mappings/ and data/control_stimuli.csv
(the target prompts with their donor's material) are released inputs; no script
in this repository regenerates them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_SAMPLES = 5
DEFAULT_JUDGE_OUTPUT_DIR = Path("data/cross-scenario_mappings")
DEFAULT_ANALYSIS_DIR = Path("analysis/matched/control_pairing/gemini_3_8_flash_v1")
# The paper's Table 3 pairs; they win ties among equally irrelevant donors.
DEFAULT_PREFERRED_CONTROLS = Path("data/control_stimuli.csv")
DEFAULT_THRESHOLD = 3.0
SCORE_KEYS = (
    "persona_relevance",
    "fact_relevance",
    "general_relevance",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} line {line_number}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"{path} line {line_number} is not a JSON object")
            rows.append(value)
    return rows


def rows_by_custom_id(
    rows: list[dict[str, Any]], source: str, id_key: str = "custom_id"
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        custom_id = row.get(id_key)
        if not isinstance(custom_id, str) or not custom_id:
            raise ValueError(f"{source} contains a row without a valid custom_id")
        if custom_id in indexed:
            raise ValueError(f"Duplicate custom_id in {source}: {custom_id}")
        indexed[custom_id] = row
    return indexed


def check_score(score: Any, score_key: str, custom_id: str) -> int:
    """Relevance scores are integers from 1 (unrelated) to 7."""
    if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 7:
        raise ValueError(f"Invalid {score_key} for {custom_id}: {score!r}")
    return score


def parse_rating_payload(text: str, custom_id: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed rating JSON for {custom_id}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Rating for {custom_id} is not a JSON object")

    expected_keys = set(SCORE_KEYS) | {"rationale"}
    if set(payload) != expected_keys:
        raise ValueError(f"Unexpected rating fields for {custom_id}: {sorted(payload)}")
    for score_key in SCORE_KEYS:
        check_score(payload[score_key], score_key, custom_id)
    if not isinstance(payload["rationale"], str) or not payload["rationale"].strip():
        raise ValueError(f"Missing rationale for {custom_id}")
    return payload


def parse_batch_results(
    metadata_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    *,
    expected_model: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    """Join Gemini batch output lines ({key, response} or {key, error}) to metadata."""
    metadata_by_id = rows_by_custom_id(metadata_rows, "request metadata")
    results_by_id = rows_by_custom_id(result_rows, "batch results", id_key="key")
    missing = sorted(set(metadata_by_id) - set(results_by_id))
    unexpected = sorted(set(results_by_id) - set(metadata_by_id))
    if missing or unexpected:
        raise ValueError(
            f"Batch-result coverage mismatch; missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )

    parsed: list[dict[str, Any]] = []
    failures: list[str] = []
    for custom_id, metadata in metadata_by_id.items():
        result = results_by_id[custom_id]
        response = result.get("response")
        if "error" in result or not isinstance(response, dict):
            error = result.get("error")
            status = error.get("status") if isinstance(error, dict) else None
            failures.append(f"{custom_id}:{status or 'invalid'}")
            continue

        returned_model = response.get("modelVersion")
        if returned_model != expected_model:
            raise ValueError(
                f"Unexpected model for {custom_id}: {returned_model!r}; "
                f"expected {expected_model!r}"
            )
        candidates = response.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ValueError(f"Expected one candidate for {custom_id}")
        candidate = candidates[0]
        if candidate.get("finishReason") != "STOP":
            raise ValueError(
                f"Unexpected finishReason for {custom_id}: "
                f"{candidate.get('finishReason')!r}"
            )
        parts = (candidate.get("content") or {}).get("parts")
        if not isinstance(parts, list):
            raise ValueError(f"Missing response content for {custom_id}")
        text_parts = [
            part["text"]
            for part in parts
            if isinstance(part, dict)
            and not part.get("thought")
            and isinstance(part.get("text"), str)
        ]
        if len(text_parts) != 1:
            raise ValueError(
                f"Expected one structured-output text part for {custom_id}, "
                f"found {len(text_parts)}"
            )
        rating = parse_rating_payload(text_parts[0], custom_id)
        parsed.append(
            {
                **metadata,
                **rating,
                "returned_model": returned_model,
                "response_id": response.get("responseId"),
                "usage": response.get("usageMetadata"),
            }
        )

    if failures:
        preview = ", ".join(failures[:5])
        raise ValueError(
            f"Batch contains {len(failures)} unsuccessful results: {preview}"
        )
    return parsed


def aggregate_ratings(
    ratings: list[dict[str, Any]],
    *,
    samples: int = DEFAULT_SAMPLES,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict[str, Any]]:
    if samples < 1:
        raise ValueError("samples must be positive")
    if not 1 <= threshold <= 7:
        raise ValueError("threshold must be between 1 and 7")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    scenario_orders: dict[str, int] = {}
    for row in ratings:
        custom_id = str(row.get("custom_id", "<missing>"))
        target_id = row.get("target_scenario_id")
        donor_id = row.get("donor_scenario_id")
        if not isinstance(target_id, str) or not isinstance(donor_id, str):
            raise ValueError(f"Missing target/donor scenario id for {custom_id}")
        if target_id == donor_id:
            raise ValueError(f"Self-pair found in ratings: {target_id}")
        for id_key, order_key in (
            ("target_scenario_id", "target_order"),
            ("donor_scenario_id", "donor_order"),
        ):
            scenario_id = row[id_key]
            order = row.get(order_key)
            if isinstance(order, bool) or not isinstance(order, int) or order < 0:
                raise ValueError(f"Invalid {order_key} for {custom_id}")
            previous = scenario_orders.setdefault(scenario_id, order)
            if previous != order:
                raise ValueError(f"Inconsistent scenario order for {scenario_id}")
        grouped.setdefault((target_id, donor_id), []).append(row)

    scenario_ids = [
        scenario_id
        for scenario_id, _ in sorted(scenario_orders.items(), key=lambda item: item[1])
    ]
    if len(set(scenario_orders.values())) != len(scenario_orders):
        raise ValueError("Scenario order values are not unique")
    expected_pairs = {
        (target_id, donor_id)
        for target_id in scenario_ids
        for donor_id in scenario_ids
        if target_id != donor_id
    }
    actual_pairs = set(grouped)
    if actual_pairs != expected_pairs:
        raise ValueError(
            f"Incomplete directed pair grid; expected={len(expected_pairs)}, "
            f"found={len(actual_pairs)}"
        )

    aggregates: list[dict[str, Any]] = []
    expected_sample_indices = set(range(samples))
    for target_id, donor_id in sorted(
        grouped,
        key=lambda pair: (scenario_orders[pair[0]], scenario_orders[pair[1]]),
    ):
        pair_rows = grouped[(target_id, donor_id)]
        sample_indices = [row.get("sample_index") for row in pair_rows]
        if len(sample_indices) != len(set(sample_indices)):
            raise ValueError(f"Duplicate sample index for {target_id}/{donor_id}")
        if set(sample_indices) != expected_sample_indices:
            raise ValueError(
                f"Invalid sample coverage for {target_id}/{donor_id}: "
                f"{sorted(sample_indices)}"
            )

        ordered_rows = sorted(pair_rows, key=lambda row: row["sample_index"])
        invariant_fields = (
            "prompt_version",
            "input_sha256",
            "target_order",
            "target_domain",
            "target_topic",
            "donor_order",
            "donor_domain",
            "donor_topic",
        )
        for field in invariant_fields:
            if len({row.get(field) for row in ordered_rows}) != 1:
                raise ValueError(
                    f"Inconsistent {field} across ratings for {target_id}/{donor_id}"
                )
        score_values = {
            score_key: [
                check_score(row.get(score_key), score_key, row["custom_id"])
                for row in ordered_rows
            ]
            for score_key in SCORE_KEYS
        }
        means = {
            score_key: statistics.mean(values)
            for score_key, values in score_values.items()
        }
        per_sample_composites = [
            statistics.mean([row[score_key] for score_key in SCORE_KEYS])
            for row in ordered_rows
        ]
        first = ordered_rows[0]
        total_score_sum = sum(sum(values) for values in score_values.values())
        aggregate = {
            "target_order": scenario_orders[target_id],
            "target_scenario_id": target_id,
            "target_domain": first.get("target_domain", ""),
            "target_topic": first.get("target_topic", ""),
            "donor_order": scenario_orders[donor_id],
            "donor_scenario_id": donor_id,
            "donor_domain": first.get("donor_domain", ""),
            "donor_topic": first.get("donor_topic", ""),
            "n_ratings": samples,
            "persona_mean": means["persona_relevance"],
            "fact_mean": means["fact_relevance"],
            "general_mean": means["general_relevance"],
            "composite_mean": total_score_sum / (len(SCORE_KEYS) * samples),
            "persona_sample_sd": (
                statistics.stdev(score_values["persona_relevance"])
                if samples > 1
                else 0.0
            ),
            "fact_sample_sd": (
                statistics.stdev(score_values["fact_relevance"]) if samples > 1 else 0.0
            ),
            "general_sample_sd": (
                statistics.stdev(score_values["general_relevance"])
                if samples > 1
                else 0.0
            ),
            "composite_sample_sd": (
                statistics.stdev(per_sample_composites) if samples > 1 else 0.0
            ),
            "persona_range": max(score_values["persona_relevance"])
            - min(score_values["persona_relevance"]),
            "fact_range": max(score_values["fact_relevance"])
            - min(score_values["fact_relevance"]),
            "general_range": max(score_values["general_relevance"])
            - min(score_values["general_relevance"]),
            "total_score_sum": total_score_sum,
            "eligible": all(mean <= threshold for mean in means.values()),
        }
        aggregates.append(aggregate)

    return aggregates


def edge_cost(edge: dict[str, Any], preferred: frozenset[tuple[str, str]]) -> int:
    """Relevance first; among equally irrelevant donors, prefer an accepted pair.

    The preference adds at most 1 per target (25 targets), so scaling relevance
    by 100 means it can never outweigh even a one-point relevance difference.
    """
    pair = (edge["target_scenario_id"], edge["donor_scenario_id"])
    return 100 * edge["total_score_sum"] + (0 if pair in preferred else 1)


def minimum_assignment_cost(
    target_ids: list[str],
    donor_ids: list[str],
    edge_by_pair: dict[tuple[str, str], dict[str, Any]],
    preferred: frozenset[tuple[str, str]] = frozenset(),
) -> int:
    if not target_ids:
        return 0
    unavailable = 10**9
    costs = np.full((len(target_ids), len(donor_ids)), unavailable, dtype=np.int64)
    for target_index, target_id in enumerate(target_ids):
        for donor_index, donor_id in enumerate(donor_ids):
            edge = edge_by_pair.get((target_id, donor_id))
            if edge is not None and edge["eligible"]:
                costs[target_index, donor_index] = edge_cost(edge, preferred)

    row_indices, column_indices = linear_sum_assignment(costs)
    selected_costs = costs[row_indices, column_indices]
    if len(selected_costs) != len(target_ids) or np.any(selected_costs == unavailable):
        raise ValueError("No complete eligible one-to-one matching exists")
    return int(selected_costs.sum())


def select_matching(
    aggregates: list[dict[str, Any]],
    preferred: frozenset[tuple[str, str]] = frozenset(),
) -> list[dict[str, Any]]:
    """Least total relevance; ties go to preferred pairs, then stimulus order."""
    edge_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    scenario_orders: dict[str, int] = {}
    for edge in aggregates:
        pair = (edge["target_scenario_id"], edge["donor_scenario_id"])
        if pair in edge_by_pair:
            raise ValueError(f"Duplicate aggregate pair: {pair}")
        edge_by_pair[pair] = edge
        for id_key, order_key in (
            ("target_scenario_id", "target_order"),
            ("donor_scenario_id", "donor_order"),
        ):
            scenario_id = edge[id_key]
            order = edge[order_key]
            previous = scenario_orders.setdefault(scenario_id, order)
            if previous != order:
                raise ValueError(f"Inconsistent scenario order for {scenario_id}")

    scenario_ids = [
        scenario_id
        for scenario_id, _ in sorted(scenario_orders.items(), key=lambda item: item[1])
    ]
    expected_pairs = {
        (target_id, donor_id)
        for target_id in scenario_ids
        for donor_id in scenario_ids
        if target_id != donor_id
    }
    if set(edge_by_pair) != expected_pairs:
        raise ValueError("Aggregates do not form a complete directed pair grid")

    remaining_donors = list(scenario_ids)
    remaining_cost = minimum_assignment_cost(
        scenario_ids, remaining_donors, edge_by_pair, preferred
    )
    selected: list[dict[str, Any]] = []
    for target_index, target_id in enumerate(scenario_ids):
        later_targets = scenario_ids[target_index + 1 :]
        chosen_donor: str | None = None
        for donor_id in remaining_donors:
            edge = edge_by_pair.get((target_id, donor_id))
            if edge is None or not edge["eligible"]:
                continue
            later_donors = [item for item in remaining_donors if item != donor_id]
            try:
                later_cost = minimum_assignment_cost(
                    later_targets, later_donors, edge_by_pair, preferred
                )
            except ValueError:
                continue
            if edge_cost(edge, preferred) + later_cost == remaining_cost:
                chosen_donor = donor_id
                remaining_cost = later_cost
                break
        if chosen_donor is None:
            raise ValueError("Could not recover the lexicographically first optimum")

        selected_edge = dict(edge_by_pair[(target_id, chosen_donor)])
        eligible_for_target = sorted(
            (
                candidate
                for (candidate_target, _), candidate in edge_by_pair.items()
                if candidate_target == target_id and candidate["eligible"]
            ),
            key=lambda candidate: (
                candidate["total_score_sum"],
                candidate["donor_order"],
            ),
        )
        selected_edge["within_target_rank"] = (
            eligible_for_target.index(edge_by_pair[(target_id, chosen_donor)]) + 1
        )
        selected_edge["kept_preferred_pair"] = (target_id, chosen_donor) in preferred
        selected_edge["human_review_status"] = "pending"
        selected.append(selected_edge)
        remaining_donors.remove(chosen_donor)

    return selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pairing_summary(
    aggregates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    samples: int,
    threshold: float,
) -> dict[str, Any]:
    return {
        "status": "requires_human_review",
        "aggregation": "arithmetic mean over repeated ratings",
        "composite": "equal-weight mean of persona, fact, and general relevance",
        "component_eligibility_rule": f"each component mean <= {threshold}",
        "matching_objective": "minimum total composite relevance",
        "tie_break": (
            "accepted-paper (Table 3) pair first, then lexicographically first "
            "donor sequence in stimulus order"
        ),
        "kept_preferred_pair_count": sum(
            bool(row["kept_preferred_pair"]) for row in selected
        ),
        "samples_per_pair": samples,
        "candidate_pair_count": len(aggregates),
        "eligible_pair_count": sum(bool(row["eligible"]) for row in aggregates),
        "selected_pair_count": len(selected),
        "selected_total_composite": sum(
            float(row["composite_mean"]) for row in selected
        ),
        "eligible_donors_per_target": {
            target_id: sum(
                row["target_scenario_id"] == target_id and bool(row["eligible"])
                for row in aggregates
            )
            for target_id in [row["target_scenario_id"] for row in selected]
        },
    }


def read_control_mapping(path: Path) -> dict[str, str]:
    """Read the target -> donor pairs used in an existing control stimuli CSV."""
    mapping: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            donor_id = (row.get("donor_scenario_id") or "").strip()
            if not donor_id:
                continue
            target_id = row["scenario_id"].strip()
            if mapping.setdefault(target_id, donor_id) != donor_id:
                raise ValueError(f"{path} uses several donors for {target_id}")
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate, aggregate, and select irrelevant-control pairings."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate-results", help="Validate downloaded Gemini batch results."
    )
    validate_parser.add_argument(
        "--request-metadata",
        type=Path,
        default=DEFAULT_JUDGE_OUTPUT_DIR / "request_metadata_full.jsonl",
    )
    validate_parser.add_argument("--batch-results", type=Path, required=True)
    validate_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_JUDGE_OUTPUT_DIR / "parsed_ratings_full.jsonl",
    )
    validate_parser.add_argument("--expected-model", default=DEFAULT_MODEL)

    select_parser = subparsers.add_parser(
        "select", help="Aggregate ratings and solve the constrained matching."
    )
    select_parser.add_argument(
        "--ratings",
        type=Path,
        default=DEFAULT_JUDGE_OUTPUT_DIR / "parsed_ratings_full.jsonl",
    )
    select_parser.add_argument(
        "--aggregated-output",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR / "aggregated_scores.csv",
    )
    select_parser.add_argument(
        "--matching-output",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR / "selected_matching.csv",
    )
    select_parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR / "validation_summary.json",
    )
    select_parser.add_argument(
        "--prefer-mapping",
        type=Path,
        default=DEFAULT_PREFERRED_CONTROLS,
        help="Control stimuli whose pairs win ties among equally irrelevant donors "
        "(default: the paper's Table 3).",
    )
    select_parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    select_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Largest allowed mean relevance on each scale (1-7).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "validate-results":
        metadata_rows = read_jsonl(args.request_metadata)
        result_rows = read_jsonl(args.batch_results)
        parsed = parse_batch_results(
            metadata_rows,
            result_rows,
            expected_model=args.expected_model,
        )
        write_jsonl(args.output, parsed)
        write_json(
            args.output.with_suffix(".metadata.json"),
            {
                "expected_model": args.expected_model,
                "request_metadata_source": str(args.request_metadata),
                "request_metadata_sha256": sha256_file(args.request_metadata),
                "batch_results_source": str(args.batch_results),
                "batch_results_sha256": sha256_file(args.batch_results),
                "parsed_ratings_sha256": sha256_file(args.output),
                "rating_count": len(parsed),
            },
        )
        print(f"Validated and wrote {len(parsed)} ratings to {args.output}")
        return

    if args.command == "select":
        ratings = read_jsonl(args.ratings)
        aggregates = aggregate_ratings(
            ratings,
            samples=args.samples,
            threshold=args.threshold,
        )
        preferred = frozenset(read_control_mapping(args.prefer_mapping).items())
        selected = select_matching(aggregates, preferred)
        write_csv(args.aggregated_output, aggregates)
        write_csv(args.matching_output, selected)
        summary = pairing_summary(
            aggregates,
            selected,
            samples=args.samples,
            threshold=args.threshold,
        )
        summary.update(
            {
                "ratings_source": str(args.ratings),
                "ratings_sha256": sha256_file(args.ratings),
                "preferred_mapping_source": str(args.prefer_mapping),
                "preferred_mapping_sha256": sha256_file(args.prefer_mapping),
                "aggregated_scores_sha256": sha256_file(args.aggregated_output),
                "selected_matching_sha256": sha256_file(args.matching_output),
            }
        )
        write_json(
            args.summary_output,
            summary,
        )
        print(f"Wrote {len(aggregates)} aggregate scores")
        print(f"Wrote {len(selected)} selected pairs requiring human review")


if __name__ == "__main__":
    main()
