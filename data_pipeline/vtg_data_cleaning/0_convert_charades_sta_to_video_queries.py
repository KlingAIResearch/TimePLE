#!/usr/bin/env python3
"""
Convert Charades-STA flat annotations into per-video grouped VTG format.

Input sample (flat):
{
  "video_id": "3MSZA",
  "video_name": "3MSZA.mp4",
  "start_time": 24.3,
  "end_time": 30.4,
  "duration": 6.1,
  "query": "person turn a light on."
}

Output sample (grouped by video):
{
  "video_id": "3MSZA",
  "video_name": "3MSZA.mp4",
  "query_count": 2,
  "queries": [
    {
      "query_id": "3MSZA#000000",
      "query": "...",
      "time_gt": {"start": 24.3, "end": 30.4},
      "time_duration": 6.1,
      "source_index": 0
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from common import ensure_parent_dir, round_float

logger = logging.getLogger(__name__)


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Charades-STA flat annotations to per-video format")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input Charades-STA JSON path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_pipeline/vtg_data_cleaning/output/charades_sta_test_by_video.json",
        help="Output grouped JSON path",
    )
    parser.add_argument("--round-digits", type=int, default=3, help="Round floats to N digits (-1 keeps raw float)")
    parser.add_argument("--sort-videos", action="store_true", help="Sort output videos by video_id")
    parser.add_argument("--compact", action="store_true", help="Save compact JSON (no indent)")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def _as_non_empty_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _valid_range(start: float, end: float) -> bool:
    return start >= 0 and end > start


def convert_rows(
    rows: List[Dict],
    round_digits: int,
    sort_videos: bool = False,
) -> Dict:
    videos = defaultdict(lambda: {"video_id": "", "video_name": "", "query_count": 0, "queries": []})

    total_rows = len(rows)
    dropped_missing_video = 0
    dropped_missing_query = 0
    dropped_invalid_time = 0

    for row_idx, row in enumerate(rows):
        video_id = _as_non_empty_text(row.get("video_id"))
        video_name = _as_non_empty_text(row.get("video_name")) or (f"{video_id}.mp4" if video_id else "")
        query = _as_non_empty_text(row.get("query"))

        if not video_id:
            dropped_missing_video += 1
            continue
        if not query:
            dropped_missing_query += 1
            continue

        try:
            start = float(row.get("start_time"))
            end = float(row.get("end_time"))
        except (TypeError, ValueError):
            dropped_invalid_time += 1
            continue

        if not _valid_range(start, end):
            dropped_invalid_time += 1
            continue

        duration_raw = row.get("duration")
        try:
            duration = float(duration_raw) if duration_raw is not None else (end - start)
        except (TypeError, ValueError):
            duration = end - start

        if duration <= 0:
            duration = end - start

        current = videos[video_id]
        if not current["video_id"]:
            current["video_id"] = video_id
        if not current["video_name"]:
            current["video_name"] = video_name

        query_index = len(current["queries"])
        query_item = {
            "query_id": f"{video_id}#{query_index:06d}",
            "query": query,
            "time_gt": {
                "start": round_float(start, round_digits),
                "end": round_float(end, round_digits),
            },
            "time_duration": round_float(duration, round_digits),
            "source_index": row_idx,
        }
        current["queries"].append(query_item)

    output_videos = list(videos.values())
    for video in output_videos:
        video["query_count"] = len(video["queries"])

    if sort_videos:
        output_videos = sorted(output_videos, key=lambda item: item["video_id"])

    kept_rows = sum(video["query_count"] for video in output_videos)
    stats = {
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "dropped_rows": total_rows - kept_rows,
        "dropped_missing_video": dropped_missing_video,
        "dropped_missing_query": dropped_missing_query,
        "dropped_invalid_time": dropped_invalid_time,
        "video_count": len(output_videos),
    }

    return {
        "dataset": "charades_sta",
        "schema_version": "vtg_by_video_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
        "videos": output_videos,
    }


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    input_path = Path(args.input)
    output_path = Path(args.output)

    logger.info("Loading input: %s", input_path)
    with input_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list):
        raise ValueError("Input JSON must be a list of flat annotations")

    logger.info("Converting %d rows...", len(rows))
    converted = convert_rows(rows, round_digits=args.round_digits, sort_videos=args.sort_videos)
    converted["source_path"] = str(input_path)

    ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=None if args.compact else 2)

    logger.info("Saved: %s", output_path)
    logger.info("Stats: %s", converted["stats"])


if __name__ == "__main__":
    main()
