#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = CURRENT_DIR.parent / "train_building"
for import_path in (CURRENT_DIR, TRAIN_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from benchmark_utils import load_rows, load_yaml, normalize_intervals, save_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply human-reviewed benchmark corrections.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", default="")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    runtime = dict(config.get("runtime", {}))
    source = Path(args.input or runtime.get("review_export_jsonl") or runtime.get("input_jsonl", "")).expanduser().resolve()
    target = Path(args.output or runtime.get("review_apply_output_jsonl") or source.with_name(f"{source.stem}.corrected{source.suffix}")).expanduser().resolve()
    output = []
    counts = {"keep": 0, "modify": 0, "delete": 0, "unreviewed": 0}
    for row in load_rows(source):
        review = row.get("manual_review")
        if not isinstance(review, dict) or not review.get("decision"):
            output.append(row)
            counts["unreviewed"] += 1
            continue
        decision = str(review.get("decision"))
        if decision == "delete":
            counts["delete"] += 1
            continue
        row_copy = dict(row)
        if decision == "modify":
            final_query = str(review.get("final_query", "")).strip()
            final_segment = normalize_intervals(review.get("final_segment"))
            if not final_query or not final_segment:
                raise ValueError(f"Modified sample requires final_query and final_segment: {review.get('sample_id')}")
            row_copy["query"] = final_query
            if "gt_timestamps" in row_copy:
                row_copy["gt_timestamps"] = final_segment
            else:
                row_copy["time_gt"] = final_segment
            counts["modify"] += 1
        else:
            counts["keep"] += 1
        row_copy["correction_audit"] = review
        row_copy.pop("manual_review", None)
        output.append(row_copy)
    save_rows(target, output)
    print(f"[DONE] output={target} counts={counts}")


if __name__ == "__main__":
    main()
