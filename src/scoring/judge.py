"""Rate saved responses for expertise or fact use with an LLM judge (GPT-OSS-120B on vLLM)."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

from src.generation import generate

SCORE_MIN = 1
SCORE_MAX = 7
REQUEST_TIMEOUT = 300  # seconds per judge request
REQUEST_RETRIES = 8  # retries after a network error or a busy server
RETRY_BACKOFF = 2.0  # seconds before the first retry; doubles on each retry
RETRY_MAX_DELAY = 60.0  # seconds
TRANSIENT_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}
PARSE_RETRIES = 2  # extra requests when a reply has no valid score
EXPERTISE_SYSTEM_PROMPT = 'Task:\n\nYou are an expert evaluator of written responses. You will be given a scenario and a response in relation to that scenario. Your task is to rate the level of expertise the response reveals about the writer.\n\nEvaluation Criteria:\n\nRate only the expertise the text reveals about the writer. Do not grade on helpfulness, politeness, or writing style. Score the response on a scale of 1 to 7.\n\nScale Representation:\n\n* 1 = No Expertise: This response is likely to be written by a writer who has no domain familiarity.\n* 4 = Average Expertise: This response is likely to be written by a writer who has ordinary domain familiarity, roughly what an average non-specialist would know.\n* 7 = Deep Expertise: This response is likely to be written by a writer who has expert-level domain familiarity.\n\nScores 2 and 3 fall between 1 and 4; scores 5 and 6 fall between 4 and 7. Higher numbers mean more expertise revealed. Choose the score that best matches the evidence; do not target any particular score distribution.\n\nEvaluation steps:\n\n1. Read the topic, scenario, and response carefully and understand the item/topic/question that is being described or answered.\n\n2. Provide a note that explains your understanding of the response. Rate the level of expertise the response reveals, using the scale above.\n\n3. Provide a brief explanation for your rating, citing specific evidence from the response. Identify the adjacent score that was the closest alternative and explain why it was not selected.\n\nOutput:\n\nReturn one valid JSON object and nothing else:\n\n{\n    "notes": "<2-3 sentences explaining your understanding of the response>",\n    "rationale": "<2-3 sentences explaining the rating>",\n    "expertise_score": <integer 1-7>\n}\n'
FACT_USAGE_SYSTEM_PROMPT = 'Task:\n\nYou are an expert evaluator of written responses. You will be given a scenario, a fact, and a response in relation to that scenario. Your task is to rate how much of the information given in the fact is used by the response.\n\nEvaluation Criteria:\n\nRate only how much the response uses the information provided in the fact. A response uses the fact when its content draws on the specific information the fact provides, whether or not it repeats the fact word-for-word. Domain-typical statements that could have been produced without the fact are not, by themselves, evidence of fact use. Do not grade on helpfulness, politeness, or any information unrelated to the fact. Score the response on a scale of 1 to 7.\n\nScale Representation:\n\n* 1 = No Use: No substantive detail from the fact is detectably incorporated.\n* 4 = Partial Use: Some substantive details from the fact are incorporated, while a meaningful portion is absent.\n* 7 = Full Use: Most or all substantive information from the fact is incorporated.\n\nScores 2 and 3 fall between 1 and 4; scores 5 and 6 fall between 4 and 7. Higher numbers mean the response uses more of the information the fact provides. Choose the score that best matches the evidence; do not target any particular score distribution.\n\nEvaluation steps:\n\n1. Read the topic, scenario, fact, and response carefully and understand what information the fact provides.\n\n2. Provide a note that explains your understanding of the response. Rate how much the response uses the information in the fact, using the scale above.\n\n3. Provide a brief explanation for your rating, citing specific evidence from the response. Identify the adjacent score that was the closest alternative and explain why it was not selected.\n\nOutput:\n\nReturn one valid JSON object and nothing else:\n\n{\n    "notes": "<2-3 sentences explaining your understanding of the response>",\n    "rationale": "<2-3 sentences explaining the rating>",\n    "fact_usage": <integer 1-7>\n}\n'
SYSTEM_PROMPTS = {"expertise": EXPERTISE_SYSTEM_PROMPT, "fact_usage": FACT_USAGE_SYSTEM_PROMPT}
# The key that holds the score in the judge's JSON reply.
SCORE_KEYS = {"expertise": "expertise_score", "fact_usage": "fact_usage"}
# The ratings collected for each (suppression_type, condition) of the stimulus files.
JUDGE_TASKS = {
    ("default_baseline", "baseline"): ("expertise", "fact_usage"),
    ("global", "high_knowledge"): ("expertise",),
    ("global", "low_knowledge"): ("expertise",),
    ("global", "suppression"): ("expertise",),
    ("local", "fact_use_explicit"): ("fact_usage",),
    ("local", "fact_suppression_explicit"): ("fact_usage",),
}


def read_stimuli(path: Path) -> dict[str, dict[str, str]]:
    """Read a stimulus CSV and index its rows by stimulus_id."""
    rows = generate.read_stimuli(path)
    stimuli = {row["stimulus_id"]: row for row in rows}
    if len(stimuli) != len(rows):
        raise ValueError(f"{path} has duplicate stimulus IDs")
    return stimuli


def local_fact(row: dict[str, str]) -> str:
    """The fact of a local prompt is the text before its use/ignore instruction."""
    fact, instruction, _question = row["prompt"].partition(f" {row['instruction']} ")
    if not instruction or not fact.strip():
        raise ValueError(f"Cannot find the fact in the prompt of {row['stimulus_id']}")
    return fact.strip()


def local_facts_by_scenario(stimuli: dict[str, dict[str, str]]) -> dict[str, str]:
    """Map each scenario to its local fact; all local rows of a scenario must agree."""
    facts: dict[str, str] = {}
    for row in stimuli.values():
        if row["suppression_type"] != "local":
            continue
        fact = local_fact(row)
        if facts.setdefault(row["scenario_id"], fact) != fact:
            raise ValueError(f"The local rows of {row['scenario_id']} give different facts")
    return facts


def read_model_outputs(paths: list[Path]) -> list[dict]:
    """Read response files written by src.generation.generate, noting each response's file."""
    responses = []
    seen = set()
    for path in paths:
        for response in json.loads(path.read_text(encoding="utf-8")):
            key = (response["model"], response["stimulus_id"], response["sample_index"])
            if key in seen:
                raise ValueError(f"Duplicate model response across inputs: {key}")
            seen.add(key)
            responses.append({**response, "_model_output_source": str(path)})
    return responses


def judge_tasks(stimulus: dict[str, str], requested_task: str | None = None) -> tuple[str, ...]:
    """The ratings to collect for a stimulus, or only `requested_task` if it applies."""
    tasks = JUDGE_TASKS[stimulus["suppression_type"], stimulus["condition"]]
    if requested_task is None:
        return tasks
    return tuple(task for task in tasks if task == requested_task)


def build_judge_messages(
    stimulus: dict[str, str], response: str, local_facts: dict[str, str], task: str
) -> list[dict[str, str]]:
    """The judge sees the topic, the scenario, the fact (fact use only) and the response,
    never the condition or the prompt the model answered."""
    if task == "expertise":
        user_prompt = f'Topic: {stimulus["topic"]}\nScenario: {stimulus["scenario"]}\n\nResponse to evaluate:\n"""\n{response}\n"""'
    else:
        fact = local_facts[stimulus["scenario_id"]]
        user_prompt = f'Topic: {stimulus["topic"]}\nScenario: {stimulus["scenario"]}\nFact: {fact}\n\nResponse to evaluate:\n"""\n{response}\n"""'
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[task]},
        {"role": "user", "content": user_prompt},
    ]


def parse_judge_output(raw_output: str | None, score_key: str) -> dict:
    """Read the score, notes and rationale from the first JSON object in a judge reply.

    Returns {"error": ...} instead when the reply holds no valid score.
    """
    if not isinstance(raw_output, str) or not raw_output.strip():
        return {"error": "missing judge message content"}
    start = raw_output.find("{")
    if start == -1:
        return {"error": "no JSON object found"}
    try:
        parsed, _end = json.JSONDecoder().raw_decode(raw_output, start)
    except json.JSONDecodeError as error:
        return {"error": str(error)}
    score = parsed.get(score_key)
    if isinstance(score, bool) or not isinstance(score, int) or not SCORE_MIN <= score <= SCORE_MAX:
        return {"error": f"{score_key} must be an integer from {SCORE_MIN} to {SCORE_MAX}"}
    return {
        score_key: score,
        "notes": str(parsed.get("notes", "")).strip(),
        "rationale": str(parsed.get("rationale", "")).strip(),
    }


def request_judgment(api_base: str, payload: dict) -> dict:
    """Send one chat-completion request, retrying network errors and busy-server replies."""
    for attempt in range(1, REQUEST_RETRIES + 2):
        request = urllib.request.Request(
            f"{api_base.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                body = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            if error.code not in TRANSIENT_HTTP_STATUS_CODES or attempt > REQUEST_RETRIES:
                message = error.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"judge request failed: HTTP {error.code}: {message}") from error
            problem = f"HTTP {error.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt > REQUEST_RETRIES:
                raise RuntimeError(f"judge request failed: {error}") from error
            problem = error
        delay = min(RETRY_BACKOFF * 2 ** (attempt - 1) + random.uniform(0, RETRY_BACKOFF), RETRY_MAX_DELAY)
        print(f"Judge request failed ({problem}); retry {attempt}/{REQUEST_RETRIES} in {delay:.1f}s")
        time.sleep(delay)
    choice = body["choices"][0]
    return {
        "content": choice["message"].get("content"),
        "response_id": body.get("id"),
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage"),
        "request_attempts": attempt,
        "raw_response": body,
    }


def judge_sample(api_base: str, payload: dict, sample_index: int, score_key: str, label: str) -> dict:
    """Collect one rating, asking again up to PARSE_RETRIES times when a reply has no valid score.

    Earlier invalid replies are kept under "failed_attempts". If every reply is
    invalid, the last one is returned and counts as an invalid sample.
    """
    failed_attempts = []
    for attempt in range(1, PARSE_RETRIES + 2):
        response = request_judgment(api_base, payload)
        sample = {"sample_index": sample_index, "raw_output": response.get("content")}
        sample.update(parse_judge_output(response.get("content"), score_key))
        for field in ("response_id", "finish_reason", "usage", "request_attempts"):
            if response.get(field) is not None:
                sample[field] = response[field]
        if "error" not in sample:
            break
        sample["raw_response"] = response.get("raw_response")
        if attempt > PARSE_RETRIES:
            print(f"Invalid sample after {attempt} attempts for {label}: {sample['error']}; storing invalid sample")
            break
        print(f"Invalid sample attempt {attempt}/{PARSE_RETRIES + 1} for {label}: {sample['error']}; retrying")
        # Keep the invalid reply, error first, without its sample index.
        failed_attempts.append(
            {"error": sample["error"], **{key: value for key, value in sample.items() if key not in ("sample_index", "error")}}
        )
    if failed_attempts:
        sample["failed_attempts"] = failed_attempts
    return sample


def judge_sampling(args: argparse.Namespace) -> dict:
    """The judge settings stored in every record and checked by --resume."""
    return {
        "n": args.samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
    }


def output_record(
    args: argparse.Namespace,
    response: dict,
    stimulus: dict[str, str],
    task: str,
    samples: list[dict],
    fact_provenance: dict[str, str],
) -> dict:
    """The record for one response and task: its ratings and where everything came from."""
    score_key = SCORE_KEYS[task]
    scores = [sample[score_key] for sample in samples if score_key in sample]
    record = {
        "run_name": args.run_name,
        "stimulus_id": response["stimulus_id"],
        "scenario_id": stimulus["scenario_id"],
        "topic": stimulus["topic"],
        "condition": stimulus["condition"],
        "suppression_type": stimulus["suppression_type"],
        "judge_task": task,
        "judge_prompt_sha256": hashlib.sha256(SYSTEM_PROMPTS[task].encode("utf-8")).hexdigest(),
        "evaluated_model": response["model"],
        "judge_model": args.judge_model,
        "score_scale": {"min": SCORE_MIN, "max": SCORE_MAX},
        "sampling": judge_sampling(args),
        "n_valid": len(scores),
        "n_invalid": len(samples) - len(scores),
        "samples": samples,
    }
    # Stored only when they have a value. With --fact-stimuli, the donor fields come
    # from the fact file's rows.
    provenance = {
        "evaluated_model_revision": response.get("model_revision"),
        "requested_model_revision": response.get("requested_model_revision"),
        "model_output_source": response.get("_model_output_source"),
        "response_generation_params": response.get("generation_params"),
        "stimuli_source": response.get("stimuli_source"),
        "stimuli_sha256": response.get("stimuli_sha256"),
        "wording_id": response.get("wording_id"),
        "operation": response.get("operation"),
        "donor_scenario_id": stimulus.get("donor_scenario_id"),
        "donor_topic": stimulus.get("donor_topic"),
        **fact_provenance,
    }
    record.update({key: value for key, value in provenance.items() if value not in (None, "")})
    record["response_sample_index"] = response["sample_index"]
    record["response_sample_seed"] = response["sample_seed"]
    record[f"{score_key}_mean"] = statistics.fmean(scores) if scores else None
    return record


def write_output_records(path: Path, records: list[dict]) -> None:
    """Write through a temporary file so an interrupted run never leaves a half-written file."""
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def run(args: argparse.Namespace) -> None:
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"{args.output} already exists; pass --resume to continue it")
    stimuli = read_stimuli(args.stimuli)
    if args.fact_stimuli is None:
        local_facts = local_facts_by_scenario(stimuli)
        fact_provenance = {}
    else:
        # The paper's control baseline: baseline responses generated from --stimuli,
        # rated for use of the donor facts in another stimulus file.
        if args.judge_task != "fact_usage":
            raise ValueError("--fact-stimuli requires --judge-task fact_usage")
        fact_stimuli = read_stimuli(args.fact_stimuli)
        local_facts = local_facts_by_scenario(fact_stimuli)
        fact_stimuli_sha256 = hashlib.sha256(args.fact_stimuli.read_bytes()).hexdigest()
        fact_provenance = {
            row["scenario_id"]: {
                "donor_scenario_id": row.get("donor_scenario_id"),
                "donor_topic": row.get("donor_topic"),
                "fact_stimuli_source": str(args.fact_stimuli),
                "fact_stimuli_sha256": fact_stimuli_sha256,
            }
            for row in fact_stimuli.values()
            if row["suppression_type"] == "local"
        }

    responses = read_model_outputs(args.model_outputs)
    stimuli_sha256 = hashlib.sha256(args.stimuli.read_bytes()).hexdigest()
    if any(response.get("stimuli_sha256") != stimuli_sha256 for response in responses):
        raise ValueError(f"Model outputs were not generated from the supplied stimuli file: {args.stimuli}")
    if args.conditions:
        responses = [response for response in responses if response["condition"] in args.conditions]
    if args.limit:
        first_scenarios = set(generate.scenario_ids_in_order(responses)[: args.limit])
        responses = [response for response in responses if response["scenario_id"] in first_scenarios]
    jobs = [
        (response, task)
        for response in responses
        for task in judge_tasks(stimuli[response["stimulus_id"]], args.judge_task)
    ]
    if not jobs:
        raise ValueError("None of the selected responses needs the requested rating")

    records = []
    if args.resume and args.output.exists():
        records = json.loads(args.output.read_text(encoding="utf-8"))
        for record in records:
            if (record["run_name"], record["judge_model"], record["sampling"]) != (
                args.run_name, args.judge_model, judge_sampling(args)
            ):
                raise ValueError(f"{args.output} has a different run name, judge model or sampling; cannot resume")
        # Records with an invalid rating are judged again from scratch.
        records = [record for record in records if record["n_invalid"] == 0]
        finished = {
            (record["evaluated_model"], record["stimulus_id"], record["response_sample_index"],
             record["response_sample_seed"], record["judge_task"])
            for record in records
        }
        jobs = [
            (response, task)
            for response, task in jobs
            if (response["model"], response["stimulus_id"], response["sample_index"],
                response["sample_seed"], task) not in finished
        ]
        print(f"Resuming {args.output}: keeping {len(records)} finished records")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Judging {len(jobs)} response-rating pairs with {args.judge_model} at {args.api_base}; writing {args.output}")
    for number, (response, task) in enumerate(jobs, start=1):
        stimulus = stimuli[response["stimulus_id"]]
        payload = {
            "model": args.judge_model,
            "messages": build_judge_messages(stimulus, response["response"], local_facts, task),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "reasoning_effort": args.reasoning_effort,
        }
        label = f"{response['stimulus_id']} sample {response['sample_index']} ({task})"
        samples = [
            judge_sample(args.api_base, payload, sample_index, SCORE_KEYS[task], label)
            for sample_index in range(args.samples)
        ]
        record = output_record(args, response, stimulus, task, samples, fact_provenance.get(stimulus["scenario_id"], {}))
        records.append(record)
        write_output_records(args.output, records)
        print(f"Wrote {number}/{len(jobs)}: {label}; valid={record['n_valid']}/{args.samples}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rate saved responses for expertise or fact use with an LLM judge.")
    parser.add_argument("--stimuli", type=Path, default=Path("data/critical_stimuli.csv"),
                        help="The stimulus CSV the responses were generated from.")
    parser.add_argument("--model-outputs", type=Path, nargs="+", required=True,
                        help="Response JSON files from src.generation.generate.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Judge output JSON; it must not exist yet unless --resume is passed.")
    parser.add_argument("--run-name", required=True, help="Name stored in every judge record.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an existing --output file; records with an invalid rating are redone.")
    parser.add_argument("--judge-task", choices=("expertise", "fact_usage"),
                        help="Collect only this rating (by default, every rating the condition needs).")
    parser.add_argument("--conditions", nargs="+", help="Judge only responses to these conditions.")
    parser.add_argument("--limit", type=int, help="Judge only the first N scenarios (for smoke tests).")
    parser.add_argument("--fact-stimuli", type=Path,
                        help="Take the local facts from this CSV instead of --stimuli (the paper's control baseline).")
    parser.add_argument("--judge-model", default="openai/gpt-oss-120b")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--samples", type=int, default=5, help="Judge ratings per response.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--reasoning-effort", choices=("minimal", "low", "medium", "high"), default="low")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
