#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline_core import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert dataset file format between final(JSONL) and debug(JSON).")
    parser.add_argument("--input", required=True, help="Input file path.")
    parser.add_argument("--output", required=True, help="Output file path.")
    parser.add_argument(
        "--src-format",
        choices=("final", "debug"),
        required=True,
        help="Source format: final=JSONL, debug=pretty JSON.",
    )
    parser.add_argument(
        "--dst-format",
        choices=("final", "debug"),
        required=True,
        help="Destination format: final=JSONL, debug=pretty JSON.",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indent when dst-format=debug.")
    return parser.parse_args()


def _indent_block(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" if line else line for line in text.splitlines())


def _normalize_output_path_for_format(path: Path, fmt: str) -> Path:
    target = fmt.strip().lower()
    if target == "final":
        return path if path.suffix.lower() == ".jsonl" else path.with_suffix(".jsonl")
    if target == "debug":
        return path if path.suffix.lower() == ".json" else path.with_suffix(".json")
    raise ValueError(f"Unsupported format for path normalization: {fmt}")


def jsonl_to_pretty_json(input_path: Path, output_path: Path, indent: int = 2) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as writer:
        writer.write("[\n")
        first = True
        for _, row in read_jsonl(input_path):
            serialized = json.dumps(row, ensure_ascii=False, indent=indent)
            if not first:
                writer.write(",\n")
            writer.write(_indent_block(serialized, spaces=2))
            first = False
            count += 1
        writer.write("\n]\n")
    return count


def pretty_json_to_jsonl(input_path: Path, output_path: Path) -> int:
    with input_path.open("r", encoding="utf-8") as reader:
        payload = json.load(reader)

    rows: list[dict[str, Any]]
    if isinstance(payload, list):
        rows = [item for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict):
        rows = [payload]
    else:
        raise ValueError("debug JSON must be a list of objects or a single object.")

    write_jsonl(output_path, rows, append=False)
    return len(rows)


def convert_file_format(
    input_path: str | Path,
    output_path: str | Path,
    src_format: str,
    dst_format: str,
    indent: int = 2,
) -> int:
    src = str(src_format).strip().lower()
    dst = str(dst_format).strip().lower()
    if src not in ("final", "debug"):
        raise ValueError(f"Invalid src_format={src_format}, expected final/debug.")
    if dst not in ("final", "debug"):
        raise ValueError(f"Invalid dst_format={dst_format}, expected final/debug.")

    input_file = Path(input_path)
    output_file = _normalize_output_path_for_format(Path(output_path), dst)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if src == dst:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(input_file.read_text(encoding="utf-8"), encoding="utf-8")
        if src == "final":
            return sum(1 for _ in read_jsonl(output_file))
        with output_file.open("r", encoding="utf-8") as reader:
            payload = json.load(reader)
        if isinstance(payload, list):
            return len(payload)
        return 1 if isinstance(payload, dict) else 0

    if src == "final" and dst == "debug":
        return jsonl_to_pretty_json(input_file, output_file, indent=indent)
    if src == "debug" and dst == "final":
        return pretty_json_to_jsonl(input_file, output_file)

    raise ValueError(f"Unsupported conversion: {src} -> {dst}")


def main() -> None:
    args = parse_args()
    rows = convert_file_format(
        input_path=args.input,
        output_path=args.output,
        src_format=args.src_format,
        dst_format=args.dst_format,
        indent=args.indent,
    )
    print(
        f"[DONE] format converted: input={args.input} output={args.output} "
        f"src={args.src_format} dst={args.dst_format} rows={rows}"
    )


if __name__ == "__main__":
    main()
