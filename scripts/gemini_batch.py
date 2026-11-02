#!/usr/bin/env python
"""Submit, check, and download a Gemini batch job, with explicit spend confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.select_irrelevant_controls import (
    DEFAULT_MODEL,
    sha256_file,
    write_json,
)

DOWNLOADABLE_STATES = ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED")


def validate_request_file(path: Path) -> int:
    """Check every line is a {key, request} object with a unique key; return the count."""
    keys: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("key") if isinstance(row, dict) else None
            if (
                not isinstance(key, str)
                or not key
                or not isinstance(row.get("request"), dict)
            ):
                raise ValueError(f"{path} line {line_number} is not a {{key, request}}")
            if key in keys:
                raise ValueError(f"Duplicate request key in {path}: {key}")
            keys.add(key)
    if not keys:
        raise ValueError(f"No requests found in {path}")
    return len(keys)


def submit(
    client: Any,
    requests_path: Path,
    model: str,
    *,
    confirm_spend: bool,
    display_name: str,
) -> dict[str, Any]:
    request_count = validate_request_file(requests_path)
    if not confirm_spend:
        raise RuntimeError(
            "Submission is disabled unless --confirm-spend is provided explicitly"
        )
    uploaded = client.files.upload(
        file=requests_path,
        config={"display_name": display_name, "mime_type": "jsonl"},
    )
    job = client.batches.create(
        model=model,
        src=uploaded.name,
        config={"display_name": display_name},
    )
    return {
        "job_name": job.name,
        "model": model,
        "display_name": display_name,
        "request_count": request_count,
        "requests_file": str(requests_path),
        "requests_sha256": sha256_file(requests_path),
        "uploaded_file": uploaded.name,
    }


def download(client: Any, job_name: str, output_path: Path) -> str:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    job = client.batches.get(name=job_name)
    if job.state.name not in DOWNLOADABLE_STATES:
        raise RuntimeError(f"Job {job_name} is {job.state.name}; nothing to download")
    content = client.files.download(file=job.dest.file_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(content)
    return job.state.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser(
        "submit", help="Upload requests and create a job."
    )
    submit_parser.add_argument("--requests", type=Path, required=True)
    submit_parser.add_argument("--model", default=DEFAULT_MODEL)
    submit_parser.add_argument("--display-name", required=True)
    submit_parser.add_argument(
        "--job-record",
        type=Path,
        help="Where to save the job record (default: next to the requests file).",
    )
    submit_parser.add_argument("--confirm-spend", action="store_true")

    status_parser = subparsers.add_parser("status", help="Print a job's state.")
    status_parser.add_argument("--job-name", required=True)

    download_parser = subparsers.add_parser(
        "download", help="Save a finished job's result file."
    )
    download_parser.add_argument("--job-name", required=True)
    download_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from google import genai

    # Reads GEMINI_API_KEY from the environment; the key is never logged or saved.
    client = genai.Client()
    if args.command == "submit":
        job_record = args.job_record or args.requests.with_suffix(".job.json")
        if job_record.exists():
            raise FileExistsError(
                f"{job_record} exists; a job was already submitted for these requests"
            )
        record = submit(
            client,
            args.requests,
            args.model,
            confirm_spend=args.confirm_spend,
            display_name=args.display_name,
        )
        write_json(job_record, record)
        print(f"Submitted {record['request_count']} requests as {record['job_name']}")
        print(f"Job record: {job_record}")
    elif args.command == "status":
        job = client.batches.get(name=args.job_name)
        print(f"{job.name}: {job.state.name}")
    else:
        state = download(client, args.job_name, args.output)
        print(f"Downloaded {state} results to {args.output}")


if __name__ == "__main__":
    main()
