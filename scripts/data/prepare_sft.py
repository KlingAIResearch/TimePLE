#!/usr/bin/env python3
"""Build public TimePLE SFT JSONL from common temporal-grounding schemas."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def message_text(row: dict[str, Any], role: str) -> str:
    for message in row.get("messages", []):
        if isinstance(message, dict) and message.get("role") == role:
            content = message.get("content", "")
            if isinstance(content, str):
                return content.strip()
    fallback = "prompt" if role == "user" else "answer"
    return str(row.get(fallback, "")).strip()


def segments(row: dict[str, Any]) -> list[list[float]]:
    value = row.get("time_gt", row.get("segments", row.get("timestamps", [])))
    if isinstance(value, dict):
        value = value.get("time_gt", [])
    result: list[list[float]] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                start, end = float(item[0]), float(item[1])
                result.append([min(start, end), max(start, end)])
    return result


def normalized(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    user = message_text(row, "user")
    assistant = message_text(row, "assistant")
    spans = segments(row)
    if not user or not assistant or not spans:
        return None
    if "<video>" not in user:
        user = f"<video>{user}"
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "videos": [f"videos/sample_{index:04d}.mp4"],
        "time_gt": spans,
    }


def sample_rows(path: Path, count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    valid_count = 0
    for row in iter_jsonl(path):
        candidate = normalized(row, valid_count)
        if candidate is None:
            continue
        valid_count += 1
        if len(reservoir) < count:
            reservoir.append(candidate)
        else:
            position = rng.randrange(valid_count)
            if position < count:
                reservoir[position] = candidate
    if len(reservoir) < count:
        raise SystemExit(f"Requested {count} samples but only found {len(reservoir)} valid records")
    # Re-number after sampling so neither source order nor source identifiers survive.
    for index, row in enumerate(reservoir):
        row["videos"] = [f"videos/sample_{index:04d}.mp4"]
    return reservoir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = sample_rows(args.input, args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} example SFT records to {args.output}")


if __name__ == "__main__":
    main()
