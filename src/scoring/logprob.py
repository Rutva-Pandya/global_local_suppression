"""Self-score saved responses: ΔLP, the change in a response's log probability between prompts.

The model that wrote each response reads it again under several prompts, and we
record the log2 probability of every response token. ΔLP is the per-token
difference between two prompts, averaged over the response (bits per scored token):

* local:  log p(response | fact-use prompt) - log p(response | baseline prompt),
  for the fact-use and fact-ignore responses of each scenario and sample;
* global: log p(response | high-knowledge prompt) - log p(response | low-knowledge prompt),
  for the high-knowledge, low-knowledge and suppression responses.

Writes a CSV with one row per response pair (local) or triple (global), and a JSON
file next to it with the per-token scores and the run's metadata.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import platform
import unicodedata
from pathlib import Path
from typing import Any

NATS_TO_BITS = 1.0 / math.log(2.0)
# Stimulus provenance stored with every row; paired responses must agree on it.
PROVENANCE_FIELDS = (
    "stimuli_source",
    "stimuli_sha256",
    "donor_scenario_id",
    "donor_topic",
)
# The role each scored condition plays. The names match the output columns.
LOCAL_ROLES = {
    ("local", "fact_use_explicit"): "fact",
    ("local", "fact_suppression_explicit"): "supp",
}
GLOBAL_ROLES = {
    ("global", "high_knowledge"): "expert",
    ("global", "low_knowledge"): "novice",
    ("global", "suppression"): "pretend",
}
# (score name, prompt, response) for every continuation scored in one bundle.
LOCAL_SCORES = (
    ("fact_base", "base_context", "fact_response"),
    ("fact_fact", "fact_context", "fact_response"),
    ("supp_base", "base_context", "supp_response"),
    ("supp_fact", "fact_context", "supp_response"),
)
GLOBAL_SCORES = tuple(
    (f"{response}_{context}", f"{context}_context", f"{response}_response")
    for response in ("expert", "novice", "pretend")
    for context in ("base", "expert", "novice")
)


def check_input(records: list[dict[str, Any]]) -> tuple[str, str, dict[str, Any]]:
    """Return the one model, revision and generation settings of a response file."""
    models = {record["model"] for record in records}
    revisions = {record["model_revision"] for record in records}
    settings = {
        json.dumps(record["generation_params"], sort_keys=True) for record in records
    }
    if len(models) != 1 or len(revisions) != 1 or len(settings) != 1:
        raise ValueError("Input must contain one model, revision, and generation setup")
    return models.pop(), revisions.pop(), json.loads(settings.pop())


def shared_provenance(
    records: list[dict[str, Any]],
    label: str,
    fields: tuple[str, ...] = PROVENANCE_FIELDS,
) -> dict[str, Any]:
    """The provenance values that `records` share; a field must be set on all of them or none."""
    provenance = {}
    for field in fields:
        present = [record.get(field) not in (None, "") for record in records]
        if not any(present):
            continue
        if not all(present):
            raise ValueError(f"Incomplete {field} provenance across {label} records")
        values = {record[field] for record in records}
        if len(values) != 1:
            raise ValueError(f"Mismatched {field} provenance across {label} records")
        provenance[field] = values.pop()
    return provenance


def group_responses(
    records: list[dict[str, Any]], roles: dict[tuple[str, str], str], sample_count: int
) -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    """Group responses as {scenario_id: {role: {sample_index: record}}}, scenarios in file order.

    Every scenario must have every role, each with sample indices 0 to sample_count - 1.
    """
    grouped: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for record in records:
        role = roles.get((record["suppression_type"], record["condition"]))
        if role is None:
            continue
        samples = grouped.setdefault(record["scenario_id"], {}).setdefault(role, {})
        if record["sample_index"] in samples:
            raise ValueError(
                f"Duplicate sample {record['sample_index']} for {record['scenario_id']!r} "
                f"condition {record['condition']!r}"
            )
        samples[record["sample_index"]] = record
    if not grouped:
        raise ValueError(
            f"Input contains none of the conditions {[c for _, c in roles]}"
        )
    expected = set(range(sample_count))
    for scenario_id, by_role in grouped.items():
        missing = [
            condition for (_, condition), role in roles.items() if role not in by_role
        ]
        if missing:
            raise ValueError(f"Missing conditions for {scenario_id!r}: {missing}")
        for role, samples in by_role.items():
            if set(samples) != expected:
                raise ValueError(
                    f"Expected sample indices 0-{sample_count - 1} for scenario {scenario_id!r} "
                    f"role {role!r}; missing={sorted(expected - set(samples))}, "
                    f"extra={sorted(set(samples) - expected)}"
                )
    return grouped


def check_shared_seed(
    records: list[dict[str, Any]], scenario_id: str, sample_index: int
) -> int:
    """Responses compared with each other were sampled with the same seed."""
    seeds = {record["sample_seed"] for record in records}
    if len(seeds) != 1:
        raise ValueError(
            f"Sample seeds do not match for {scenario_id!r} sample {sample_index}"
        )
    return seeds.pop()


def pair_local(
    records: list[dict[str, Any]], sample_count: int
) -> list[dict[str, Any]]:
    """One bundle per scenario and sample: the fact-use and fact-ignore responses."""
    grouped = group_responses(records, LOCAL_ROLES, sample_count)
    bundles = []
    for scenario_id in sorted(grouped):
        for sample_index in range(sample_count):
            fact = grouped[scenario_id]["fact"][sample_index]
            supp = grouped[scenario_id]["supp"][sample_index]
            label = f"{scenario_id} sample {sample_index}"
            seed = check_shared_seed([fact, supp], scenario_id, sample_index)
            bundles.append(
                {
                    "scenario_id": scenario_id,
                    "sample_index": sample_index,
                    "sample_seed": seed,
                    "base_context": fact["baseline_model_input"],
                    "fact_context": fact["model_input"],
                    "fact_response": fact["response"],
                    "supp_response": supp["response"],
                    "baseline_stimulus_id": fact["baseline_stimulus_id"],
                    "fact_stimulus_id": fact["stimulus_id"],
                    "suppression_stimulus_id": supp["stimulus_id"],
                    **shared_provenance(
                        [fact, supp], label, ("wording_id", *PROVENANCE_FIELDS)
                    ),
                }
            )
    return bundles


def pair_global(
    records: list[dict[str, Any]], sample_count: int
) -> list[dict[str, Any]]:
    """One bundle per scenario and sample: the high, low and suppression responses."""
    grouped = group_responses(records, GLOBAL_ROLES, sample_count)
    bundles = []
    for scenario_id, by_role in grouped.items():
        for sample_index in range(sample_count):
            expert, novice, pretend = (
                by_role[role][sample_index] for role in ("expert", "novice", "pretend")
            )
            label = f"{scenario_id} sample {sample_index}"
            seed = check_shared_seed(
                [expert, novice, pretend], scenario_id, sample_index
            )
            # All three responses are scored against the same baseline prompt.
            for field in (
                "baseline_stimulus_id",
                "baseline_model_input",
                "baseline_prompt_messages",
            ):
                if not expert.get(field) == novice.get(field) == pretend.get(field):
                    raise ValueError(
                        f"Baseline field {field!r} does not match for {scenario_id!r} sample {sample_index}"
                    )
            provenance = shared_provenance([expert, pretend], label)
            # The paper's low-knowledge responses came from another stimulus file;
            # where their provenance differs it is stored under low_knowledge_*.
            low = shared_provenance([novice], f"{label} low knowledge")
            bundles.append(
                {
                    "scenario_id": scenario_id,
                    "sample_index": sample_index,
                    "sample_seed": seed,
                    "base_context": expert["baseline_model_input"],
                    "expert_context": expert["model_input"],
                    "novice_context": novice["model_input"],
                    "expert_response": expert["response"],
                    "novice_response": novice["response"],
                    "pretend_response": pretend["response"],
                    "baseline_stimulus_id": expert["baseline_stimulus_id"],
                    "expert_stimulus_id": expert["stimulus_id"],
                    "novice_stimulus_id": novice["stimulus_id"],
                    "pretend_stimulus_id": pretend["stimulus_id"],
                    **provenance,
                    **{
                        f"low_knowledge_{field}": value
                        for field, value in low.items()
                        if provenance.get(field) != value
                    },
                }
            )
    return bundles


# minicons, torch and transformers are imported only inside load_scorer, so this
# file and its tests can be used without them.
def load_scorer(model_name: str, revision: str) -> Any:
    import torch
    from minicons import scorer as minicons_scorer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token_id is None:
        # Llama has no padding token; reuse end-of-sequence instead.
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, device_map="auto", dtype=torch.bfloat16
    )
    model.eval()
    scorer = minicons_scorer.IncrementalLMScorer(
        model, device="auto", tokenizer=tokenizer
    )
    # With device_map="auto", inputs go to the device that holds the embeddings.
    scorer.device = str(model.get_input_embeddings().weight.device)
    return scorer


def target_token_ids(encoded: Any, offsets: list[int]) -> list[list[int]]:
    """The response tokens of each padded prompt + response sequence."""
    targets = []
    for input_ids, attention_mask, offset in zip(
        encoded["input_ids"], encoded["attention_mask"], offsets
    ):
        unpadded = [
            int(token_id)
            for token_id, attended in zip(input_ids, attention_mask)
            if int(attended) != 0
        ]
        targets.append(unpadded[1:][offset:])
    return targets


def score_continuations(
    scorer: Any, prefixes: list[str], responses: list[str]
) -> list[dict[str, Any]]:
    """Log2 probability of every token of each response, given its prompt.

    Returns {"token_ids", "tokens", "logprobs"} per response. Empty responses
    have no tokens, so they get empty lists and their averages are undefined.
    """
    scored = [{"token_ids": [], "tokens": [], "logprobs": []} for _ in responses]
    nonempty = [index for index, response in enumerate(responses) if response != ""]
    if not nonempty:
        return scored
    prefixes = [prefixes[index] for index in nonempty]
    responses = [responses[index] for index in nonempty]
    prefix_ids = scorer.tokenizer(prefixes, add_special_tokens=True)["input_ids"]
    encoded, _ = scorer.prime_text(prefixes, responses, separator="")
    offsets = [len(token_ids) - 1 for token_ids in prefix_ids]
    token_ids = target_token_ids(encoded, offsets)
    tensors = scorer.compute_stats(
        (encoded, offsets), rank=False, prob=False, base_two=False, return_tensors=True
    )
    logprobs_nats = [tensor.detach().float().cpu().tolist() for tensor in tensors]
    # The scored tokens must be exactly the stored response, not part of the prompt.
    for response, ids, scores in zip(responses, token_ids, logprobs_nats):
        if len(ids) != len(scores):
            raise ValueError(
                "Minicons returned a different number of scores and target tokens"
            )
        decoded = scorer.tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if decoded != response and unicodedata.normalize(
            "NFC", decoded
        ) != unicodedata.normalize("NFC", response):
            raise ValueError(
                "Minicons target-token boundary does not reconstruct the stored response: "
                f"expected={response!r}, decoded={decoded!r}"
            )
    for index, ids, scores in zip(nonempty, token_ids, logprobs_nats):
        scored[index] = {
            "token_ids": ids,
            "tokens": scorer.tokenizer.convert_ids_to_tokens(ids),
            "logprobs": [score * NATS_TO_BITS for score in scores],
        }
    return scored


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"sum": None, "mean": None}
    total = sum(values)
    return {"sum": total, "mean": total / len(values)}


def check_same_tokens(scores: list[dict[str, Any]]) -> None:
    """ΔLP compares the same tokens, so a response must tokenize the same after every prompt."""
    if len({tuple(score["token_ids"]) for score in scores}) != 1:
        raise ValueError(
            "The fixed response produced different target tokens across contexts"
        )


def add_context_scores(
    summary: dict[str, Any],
    name: str,
    contexts: tuple[str, ...],
    summaries: dict[str, dict],
) -> None:
    """Add logp_r_<name>_c_<context>_<sum|mean> and the matching surprisal (-logp) columns."""
    for kind in ("logp", "surprisal"):
        for reduction in ("sum", "mean"):
            for context in contexts:
                value = summaries[context][reduction]
                if kind == "surprisal" and value is not None:
                    value = -value
                summary[f"{kind}_r_{name}_c_{context}_{reduction}"] = value


def local_response(
    text: str, base: dict[str, Any], fact: dict[str, Any]
) -> dict[str, Any]:
    """Per-token scores of one response after the baseline and fact-use prompts."""
    check_same_tokens([base, fact])
    delta = [
        fact_value - base_value
        for base_value, fact_value in zip(base["logprobs"], fact["logprobs"])
    ]
    return {
        "text": text,
        "token_ids": base["token_ids"],
        "tokens": base["tokens"],
        "base_logprobs": base["logprobs"],
        "fact_logprobs": fact["logprobs"],
        "base": summarize(base["logprobs"]),
        "fact": summarize(fact["logprobs"]),
        "token_delta_logprob": delta,
        "delta_logprob": summarize(delta),
    }


def local_result(
    bundle: dict[str, Any], scores: dict[str, dict], model: str, revision: str
) -> dict[str, Any]:
    """The CSV row (summary) and per-token details of one local bundle."""
    responses = {
        "fact": local_response(
            bundle["fact_response"], scores["fact_base"], scores["fact_fact"]
        ),
        "supp": local_response(
            bundle["supp_response"], scores["supp_base"], scores["supp_fact"]
        ),
    }
    summary = {
        "scenario_id": bundle["scenario_id"],
        "sample_index": bundle["sample_index"],
        "sample_seed": bundle["sample_seed"],
        "model": model,
        "model_revision": revision,
        "scoring_model": model,
        "scoring_model_revision": revision,
        **{field: bundle.get(field, "") for field in PROVENANCE_FIELDS},
        "wording_id": bundle.get("wording_id"),
        "baseline_stimulus_id": bundle["baseline_stimulus_id"],
        "fact_context_stimulus_id": bundle["fact_stimulus_id"],
        "fact_stimulus_id": bundle["fact_stimulus_id"],
        "suppression_stimulus_id": bundle["suppression_stimulus_id"],
        "r_fact_token_count": len(responses["fact"]["token_ids"]),
        "r_supp_token_count": len(responses["supp"]["token_ids"]),
    }
    for name, label in (("fact", "high"), ("supp", "suppression")):
        response = responses[name]
        add_context_scores(
            summary,
            name,
            ("base", "fact"),
            {"base": response["base"], "fact": response["fact"]},
        )
        summary[f"delta_logprob_{label}_sum"] = response["delta_logprob"]["sum"]
        summary[f"delta_logprob_{label}_mean"] = response["delta_logprob"]["mean"]
    return {
        "summary": summary,
        "fact_response": responses["fact"],
        "supp_response": responses["supp"],
    }


def global_response(
    text: str, base: dict, expert: dict, novice: dict, anchor_context: str | None = None
) -> dict[str, Any]:
    """Per-token scores of one response after the baseline, high- and low-knowledge prompts.

    baseline_support is each knowledge prompt minus the baseline prompt; the context
    anchor (Figure 5) is that difference for the prompt the response was written under.
    """
    check_same_tokens([expert, novice, base])
    contexts = {"expert": expert, "novice": novice, "base": base}
    delta = [
        expert_value - novice_value
        for expert_value, novice_value in zip(expert["logprobs"], novice["logprobs"])
    ]
    support = {
        context: [
            value - base_value
            for value, base_value in zip(
                contexts[context]["logprobs"], base["logprobs"]
            )
        ]
        for context in ("expert", "novice")
    }
    result = {
        "text": text,
        "token_ids": expert["token_ids"],
        "tokens": expert["tokens"],
        "context_logprobs": {
            context: score["logprobs"] for context, score in contexts.items()
        },
        "context_summaries": {
            context: summarize(score["logprobs"]) for context, score in contexts.items()
        },
        "token_baseline_support": support,
        "baseline_support": {
            context: summarize(values) for context, values in support.items()
        },
        "token_delta_logprob": delta,
        "delta_logprob": summarize(delta),
    }
    if anchor_context is not None:
        anchor = support[anchor_context]
        result["token_context_anchor"] = anchor
        result["context_anchor"] = summarize(anchor)
        result["context_anchor_context"] = anchor_context
    return result


def global_result(
    bundle: dict[str, Any], scores: dict[str, dict], model: str, revision: str
) -> dict[str, Any]:
    """The CSV row (summary) and per-token details of one global bundle."""
    responses = {
        name: global_response(
            bundle[f"{name}_response"],
            scores[f"{name}_base"],
            scores[f"{name}_expert"],
            scores[f"{name}_novice"],
            anchor_context=None if name == "pretend" else name,
        )
        for name in ("expert", "novice", "pretend")
    }
    summary = {
        "scenario_id": bundle["scenario_id"],
        "sample_index": bundle["sample_index"],
        "sample_seed": bundle["sample_seed"],
        "model": model,
        "model_revision": revision,
        "scoring_model": model,
        "scoring_model_revision": revision,
        **{field: bundle.get(field, "") for field in PROVENANCE_FIELDS},
        "baseline_stimulus_id": bundle["baseline_stimulus_id"],
        "expert_stimulus_id": bundle["expert_stimulus_id"],
        "novice_stimulus_id": bundle["novice_stimulus_id"],
        "pretend_stimulus_id": bundle["pretend_stimulus_id"],
        **{
            f"r_{name}_token_count": len(response["token_ids"])
            for name, response in responses.items()
        },
    }
    for name, label in (
        ("expert", "high"),
        ("novice", "low"),
        ("pretend", "suppression"),
    ):
        response = responses[name]
        add_context_scores(
            summary, name, ("base", "expert", "novice"), response["context_summaries"]
        )
        summary[f"delta_logprob_{label}_sum"] = response["delta_logprob"]["sum"]
        summary[f"delta_logprob_{label}_mean"] = response["delta_logprob"]["mean"]
        for context in ("expert", "novice"):
            for reduction in ("sum", "mean"):
                summary[f"baseline_support_r_{name}_c_{context}_{reduction}"] = (
                    response["baseline_support"][context][reduction]
                )
        if "context_anchor" in response:
            summary[f"{label}_context_anchor_sum"] = response["context_anchor"]["sum"]
            summary[f"{label}_context_anchor_mean"] = response["context_anchor"]["mean"]
    summary.update(
        {
            f"low_knowledge_{field}": bundle.get(f"low_knowledge_{field}", "")
            for field in PROVENANCE_FIELDS
        }
    )
    return {
        "summary": summary,
        **{f"{name}_response": response for name, response in responses.items()},
    }


def run_metadata(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    model: str,
    revision: str,
    params: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "metric": f"{args.scope}_delta_logprob",
        "schema_version": "delta_logprob_v2",
        "scoring_type": "self_scoring",
        "positive_direction": "high_knowledge_context_support",
        "response_source": "stored_sampled_response",
        "empty_response_count": sum(record["response"] == "" for record in records),
        "empty_response_scoring_policy": "undefined_null_tokenwise_metrics",
        "context_source": "saved_model_input",
        "scoring_model_loader": "auto_model_for_causal_lm",
        "generation_params": params,
        "log_base": 2,
        "primary_unit": "bits_per_scored_token",
        "secondary_unit": "bits",
        "primary_reduction": "mean_per_scored_token",
        "secondary_reduction": "sum",
        "input_file": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "scenario_count": len({result["summary"]["scenario_id"] for result in results}),
        "model": model,
        "model_revision": revision,
        "response_model": model,
        "response_model_revision": revision,
        "scoring_model": model,
        "scoring_model_revision": revision,
        "device": "auto",
        "dtype": "bfloat16",
        "batch_size_scenarios": 1,
        "python_version": platform.python_version(),
        "torch_version": importlib.metadata.version("torch"),
        "transformers_version": importlib.metadata.version("transformers"),
        "minicons_version": importlib.metadata.version("minicons"),
    }
    if args.scope == "local":
        metadata.update(
            {
                "delta_logprob_definition": "mean_log2_probability_fact_context_minus_baseline_context",
                "response_sample_count": params["samples"],
                "wording_ids": sorted(
                    {
                        str(record["wording_id"])
                        for record in records
                        if record.get("wording_id")
                    }
                ),
                "response_pair_count": len(results),
            }
        )
    else:
        metadata.update(
            {
                "score_set": "matched",
                "conditions": ["high_knowledge", "suppression", "low_knowledge"],
                "primary_score": "delta_logprob",
                "delta_logprob_definition": "mean_log2_probability_high_knowledge_context_minus_low_knowledge_context",
                "positive_context": "high_knowledge_aligned",
                "negative_context": "low_knowledge_aligned",
                "baseline_support_definition": "mean_log2_probability_knowledge_context_minus_base_context",
                "positive_baseline_support": "knowledge_context_more_supportive_than_baseline",
                "response_bundle_count": len(results),
                "stored_response_count": len(results) * 3,
                "scored_continuation_count": len(results) * len(GLOBAL_SCORES),
            }
        )
    return metadata


def write_outputs(
    csv_path: Path, metadata: dict[str, Any], results: list[dict[str, Any]]
) -> None:
    """The CSV holds each result's summary (its keys are the columns); the JSON holds everything."""
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]["summary"]))
        writer.writeheader()
        writer.writerows(result["summary"] for result in results)
    csv_path.with_suffix(".json").write_text(
        json.dumps(
            {"metadata": metadata, "results": results}, ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    json_path = args.output.with_suffix(".json")
    for path in (args.output, json_path):
        if path.exists():
            raise FileExistsError(f"{path} already exists; choose a new --output")
    records = json.loads(args.input.read_text(encoding="utf-8"))
    model, revision, params = check_input(records)
    if args.scope == "local":
        bundles = pair_local(records, params["samples"])
        score_pairs, build_result = LOCAL_SCORES, local_result
    else:
        bundles = pair_global(records, params["samples"])
        score_pairs, build_result = GLOBAL_SCORES, global_result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Scoring {len(bundles)} {args.scope} response bundles from {args.input} with {model}"
    )

    scorer = load_scorer(model, revision)
    results = []
    for bundle in bundles:
        scored = score_continuations(
            scorer,
            [bundle[context] for _, context, _ in score_pairs],
            [bundle[response] for _, _, response in score_pairs],
        )
        scores = dict(zip((name for name, _, _ in score_pairs), scored))
        results.append(build_result(bundle, scores, model, revision))
    write_outputs(
        args.output,
        run_metadata(args, records, model, revision, params, results),
        results,
    )
    print(f"Wrote {len(results)} rows to {args.output} and {json_path}")


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-score saved responses (local or global ΔLP)."
    )
    parser.add_argument("--scope", choices=("local", "global"), required=True)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Response JSON file from src.generation.generate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV; the per-token JSON is written next to it. Neither may exist yet.",
    )
    args = parser.parse_args(arguments)
    if args.output.suffix != ".csv":
        parser.error("--output must be a .csv file")
    return args


if __name__ == "__main__":
    run(parse_args())
