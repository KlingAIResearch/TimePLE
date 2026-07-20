#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline_core import normalize_intervals, read_jsonl

VIDEO_PATH_KEYS = ("video_path", "video", "video_file", "media_path", "path")
QUERY_KEYS = ("query", "question", "instruction", "prompt", "text")
ANSWER_KEYS = ("answer", "response", "target", "label")
TIME_GT_KEYS = ("time_gt", "gt", "timestamps", "timestamp", "time_span", "span")
SOURCE_KEYS = ("source", "dataset", "dataset_name")
DATA_TYPE_KEYS = ("data_type", "task_type", "type")
SAMPLE_ID_KEYS = ("sample_id", "id", "uid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 1: Normalize raw JSONL to a unified grounding schema.")
    parser.add_argument("--input", required=True, help="Raw input JSONL path.")
    parser.add_argument("--output", required=True, help="Normalized output JSONL path.")
    parser.add_argument(
        "--video-base-dir",
        default="",
        help="Prepend this base dir to relative video_path values.",
    )
    parser.add_argument("--source-name", default="TimePLE", help="Fallback value for `source`.")
    parser.add_argument("--data-type", default="grounding", help="Fallback value for `data_type`.")
    parser.add_argument(
        "--answer-template",
        default="",
        help="Fallback value for `answer` when missing (default: empty string).",
    )
    parser.add_argument(
        "--no-placeholders",
        action="store_true",
        help="Do not include model/iou placeholder fields in output.",
    )
    parser.add_argument(
        "--drop-sample-id",
        action="store_true",
        help="Do not write `sample_id` in output.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Only process first N valid samples.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop immediately when any invalid row is encountered.",
    )
    return parser.parse_args()


def first_non_empty(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def normalize_one_row(
    row: dict[str, Any],
    line_no: int,
    video_base_dir: str,
    source_name: str,
    data_type: str,
    answer_template: str,
    with_placeholders: bool,
    keep_sample_id: bool,
) -> dict[str, Any]:
    video_path = first_non_empty(row, VIDEO_PATH_KEYS)
    query = first_non_empty(row, QUERY_KEYS)
    answer = first_non_empty(row, ANSWER_KEYS) or answer_template
    gt_raw = first_non_empty(row, TIME_GT_KEYS)
    time_gt = normalize_intervals(gt_raw)

    if not video_path:
        raise ValueError(f"missing video_path at line {line_no}")
    if not query:
        raise ValueError(f"missing query at line {line_no}")
    if not time_gt:
        raise ValueError(f"missing or invalid time_gt at line {line_no}")

    video_path_str = str(video_path).strip()
    if (
        video_base_dir
        and video_path_str
        and not Path(video_path_str).is_absolute()
        and not video_path_str.startswith(("http://", "https://", "file://"))
    ):
        video_path_str = str(Path(video_base_dir) / video_path_str)

    normalized: dict[str, Any] = {
        "source": first_non_empty(row, SOURCE_KEYS) or source_name,
        "data_type": first_non_empty(row, DATA_TYPE_KEYS) or data_type,
        "video_path": video_path_str,
        "query": str(query).strip(),
        "answer": str(answer).strip(),
        "time_gt": time_gt,
    }

    if with_placeholders:
        normalized.update(
            {
                "gemini3pro_tgt": row.get("gemini3pro_tgt"),
                "gemini3pro_pred_intervals": row.get("gemini3pro_pred_intervals"),
                "qwen3vl_30b_tgt": row.get("qwen3vl_30b_tgt"),
                "qwen3vl_30b_pred_intervals": row.get("qwen3vl_30b_pred_intervals"),
                "iou": row.get("iou"),
                "iou_gemini3pro": row.get("iou_gemini3pro"),
                "iou_qwen3vl_30b": row.get("iou_qwen3vl_30b"),
            }
        )

    if keep_sample_id:
        sample_id = first_non_empty(row, SAMPLE_ID_KEYS) or f"sample_{line_no}"
        normalized["sample_id"] = str(sample_id)

    return normalized


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    skipped = 0
    with_placeholders = not args.no_placeholders
    keep_sample_id = not args.drop_sample_id

    with output_path.open("w", encoding="utf-8") as writer:
        for line_no, row in read_jsonl(input_path):
            total += 1
            try:
                normalized = normalize_one_row(
                    row=row,
                    line_no=line_no,
                    video_base_dir=str(args.video_base_dir).strip(),
                    source_name=args.source_name,
                    data_type=args.data_type,
                    answer_template=args.answer_template,
                    with_placeholders=with_placeholders,
                    keep_sample_id=keep_sample_id,
                )
            except Exception as exc:
                skipped += 1
                if args.strict:
                    raise
                print(f"[WARN] skip line {line_no}: {exc}")
                continue

            writer.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            kept += 1

            if args.max_samples > 0 and kept >= args.max_samples:
                break

    print(
        f"[DONE] normalized dataset: input={input_path} output={output_path} "
        f"total={total} kept={kept} skipped={skipped}"
    )


if __name__ == "__main__":
    main()
