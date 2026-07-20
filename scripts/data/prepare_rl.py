#!/usr/bin/env python3
"""Convert public TimePLE SFT records to the EasyR1 JSONL schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            row = json.loads(line)
            messages = {item["role"]: item["content"] for item in row["messages"]}
            prompt = messages["user"]
            if prompt.startswith("<video>"):
                prompt = "<video>\n" + prompt[len("<video>") :].lstrip()
            converted = {
                "sample_id": f"sample_{count:04d}",
                "prompt": prompt,
                "videos": list(row["videos"]),
                "ground_truth": {
                    "time_gt": row["time_gt"],
                    "reference_answer": messages["assistant"],
                },
            }
            target.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1
    print(f"wrote {count} example RL records to {args.output}")


if __name__ == "__main__":
    main()
