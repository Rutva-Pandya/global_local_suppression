#!/usr/bin/env python
"""Generate model responses for global-local suppression stimuli."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = (
    "stimulus_id",
    "scenario_id",
    "domain",
    "suppression_type",
    "condition",
    "scenario",
    "prompt",
)
# Copied into a response record only when the stimulus row has a value.
OPTIONAL_COLUMNS = (
    "wording_id",
    "operation",
    "instruction",
    "donor_scenario_id",
    "donor_topic",
)
MODEL_REVISIONS = {
    "meta-llama/Llama-3.1-8B-Instruct": "0e9e39f249a16976918f6564b8830bc894c89659",
    "meta-llama/Llama-3.1-70B-Instruct": "1605565b47bb9346c5515c34102e054115b4f98b",
    "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    "Qwen/Qwen2.5-72B-Instruct": "495f39366efef23836d0cfae4fbe635880d2be31",
}
MAX_NEW_TOKENS = 512
STOP_MARKER = "END_REPLY"
SYSTEM_PROMPT = (
    "You are responding as the person in the situation. "
    "Write exactly one spoken first-person reply (that can be multiple sentences long) that the person would say aloud. "
    "Do not describe or narrate actions. "
    "Do not explain what the response is doing. "
    f"After the reply, write {STOP_MARKER}."
)


def read_stimuli(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        stimuli = list(reader)
    for index, row in enumerate(stimuli):
        row["row_index"] = index
    return stimuli


def scenario_ids_in_order(stimuli: list[dict[str, Any]]) -> list[str]:
    """Scenario IDs in the order they first appear."""
    return list(dict.fromkeys(row["scenario_id"] for row in stimuli))


def select_stimuli(
    stimuli: list[dict[str, Any]],
    conditions: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Keep only the given conditions, then only the first `limit` scenarios."""
    if conditions is not None:
        unknown = sorted(set(conditions) - {row["condition"] for row in stimuli})
        if unknown:
            raise ValueError(f"Unknown conditions: {', '.join(unknown)}")
        stimuli = [row for row in stimuli if row["condition"] in conditions]
    if limit is not None:
        first_scenarios = scenario_ids_in_order(stimuli)[:limit]
        stimuli = [row for row in stimuli if row["scenario_id"] in first_scenarios]
    return stimuli


def build_prompt_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    user_prompt = f"Situation:\n{row['scenario']}\n\nRequest:\n{row['prompt']}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def render_model_input(tokenizer: Any, prompt_messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )


def clean_response(response: str) -> str:
    """Cut the reply at the stop marker, if the model wrote one."""
    return response.split(STOP_MARKER, maxsplit=1)[0].strip()


# torch and transformers are imported inside the two functions below, so this
# file (and the judge and tests that import it) can be used without them.
def load_model_and_tokenizer(model_name: str, revision: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, device_map="auto", dtype=torch.bfloat16
    )
    model.eval()
    return model, tokenizer


def generate_responses(
    model: Any,
    tokenizer: Any,
    model_input: str,
    params: dict[str, Any],
    sample_seeds: list[int],
) -> tuple[list[tuple[str, int]], int]:
    """Sample one response per seed.

    Returns (response, response_tokens) for each seed, and the prompt length.
    """
    import torch
    from transformers import set_seed

    encoded = tokenizer(model_input, return_tensors="pt").to(model.device)
    prompt_tokens = encoded["input_ids"].shape[-1]
    # Llama has no padding token; reuse end-of-sequence instead.
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    responses = []
    for seed in sample_seeds:
        set_seed(seed)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=params["max_new_tokens"],
                do_sample=params["do_sample"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                top_k=params["top_k"],
                pad_token_id=pad_token_id,
            )
        response_ids = generated[0, prompt_tokens:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        responses.append((clean_response(response), response_ids.numel()))
    return responses, prompt_tokens


def generation_params(args: argparse.Namespace) -> dict[str, Any]:
    """Sampling settings, stored with every response."""
    return {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "samples": args.samples,
        "base_seed": args.seed,
    }


def run(args: argparse.Namespace) -> None:
    if args.output.exists() and not args.resume:
        raise FileExistsError(
            f"{args.output} already exists; pass --resume to continue it"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_stimuli = read_stimuli(args.stimuli)
    stimuli = select_stimuli(all_stimuli, args.conditions, args.limit)
    stimuli_sha256 = hashlib.sha256(args.stimuli.read_bytes()).hexdigest()
    params = generation_params(args)
    revision = MODEL_REVISIONS[args.model]
    # Every response also stores its scenario's baseline prompt.
    baselines = {
        row["scenario_id"]: row for row in all_stimuli if row["condition"] == "baseline"
    }
    # Seeds come from scenario positions in the full selection, so a resumed run
    # gives every stimulus the same seeds as an uninterrupted run.
    scenario_ids = scenario_ids_in_order(stimuli)
    print(f"Selected {len(stimuli)} of {len(all_stimuli)} stimuli from {args.stimuli}")

    records = []
    if args.resume and args.output.exists():
        records = json.loads(args.output.read_text(encoding="utf-8"))
        for record in records:
            settings = (
                record["model"],
                record["requested_model_revision"],
                record["generation_params"],
                record["stimuli_sha256"],
            )
            if settings != (args.model, revision, params, stimuli_sha256):
                raise ValueError(
                    f"{args.output} has a different model, sampling or stimulus file; "
                    "cannot resume"
                )
        finished = {record["stimulus_id"] for record in records}
        stimuli = [row for row in stimuli if row["stimulus_id"] not in finished]
        print(
            f"Resuming {args.output}: keeping {len(records)} responses; "
            f"{len(stimuli)} stimuli left"
        )
        if not stimuli:
            return

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model, revision)
    model_revision = model.config._commit_hash or revision
    print(f"Writing responses to {args.output}")

    for done, row in enumerate(stimuli, start=1):
        prompt_messages = build_prompt_messages(row)
        model_input = render_model_input(tokenizer, prompt_messages)
        baseline = baselines[row["scenario_id"]]
        baseline_messages = build_prompt_messages(baseline)
        baseline_input = render_model_input(tokenizer, baseline_messages)
        # All conditions of a scenario share its seeds:
        # seed + (scenario position) * samples + sample index.
        first_seed = args.seed + scenario_ids.index(row["scenario_id"]) * args.samples
        sample_seeds = [first_seed + i for i in range(args.samples)]
        responses, prompt_tokens = generate_responses(
            model, tokenizer, model_input, params, sample_seeds
        )

        for sample_index, (response, response_tokens) in enumerate(responses):
            record = {
                "row_index": row["row_index"],
                "stimulus_id": row["stimulus_id"],
                "scenario_id": row["scenario_id"],
                "domain": row["domain"],
                "suppression_type": row["suppression_type"],
                "condition": row["condition"],
                "model": args.model,
                "model_revision": model_revision,
                "generation_params": params,
                "prompt_messages": prompt_messages,
                "model_input": model_input,
                "response": response,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "requested_model_revision": revision,
                "stimuli_source": str(args.stimuli),
                "stimuli_sha256": stimuli_sha256,
            }
            for column in OPTIONAL_COLUMNS:
                if row.get(column):
                    record[column] = row[column]
            record["sample_index"] = sample_index
            record["sample_seed"] = sample_seeds[sample_index]
            record["baseline_stimulus_id"] = baseline["stimulus_id"]
            record["baseline_prompt_messages"] = baseline_messages
            record["baseline_model_input"] = baseline_input
            records.append(record)

        # Rewritten after every stimulus, so an interrupted run keeps its progress.
        # Written through a temporary file so the output is never half-written.
        temporary_path = args.output.with_name(f"{args.output.name}.tmp")
        temporary_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary_path.replace(args.output)
        print(
            f"Wrote {done}/{len(stimuli)}: {row['stimulus_id']} ({len(responses)} responses)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate responses for a stimulus CSV with a Hugging Face model."
    )
    parser.add_argument(
        "--model", required=True, choices=MODEL_REVISIONS, help="Paper model to use."
    )
    parser.add_argument("--stimuli", type=Path, required=True, help="Stimulus CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file; it must not exist yet unless --resume is passed.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing --output file, skipping stimuli it already holds.",
    )
    parser.add_argument("--conditions", nargs="+", help="Only these conditions.")
    parser.add_argument(
        "--limit", type=int, help="Only the first N scenarios, for quick tests."
    )
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260719)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
