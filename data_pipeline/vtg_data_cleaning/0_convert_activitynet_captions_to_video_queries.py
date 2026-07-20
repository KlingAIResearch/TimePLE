#!/usr/bin/env python3
"""
Convert ActivityNet Captions annotations into per-video grouped VTG format.

Input sample (row-level, one row per video):
{
  "video_id": "v_QOlSCBRmfWY",
  "video": "v_QOlSCBRmfWY.mp4",
  "duration": 82.73,
  "timestamps": [[0.83, 19.86], [17.37, 60.81]],
  "sentences": ["...", "..."]
}

Output sample (grouped by video):
{
  "video_id": "v_QOlSCBRmfWY",
  "video_name": "v_QOlSCBRmfWY.mp4",
  "query_count": 2,
  "queries": [
    {
      "query_id": "v_QOlSCBRmfWY#000000",
      "query": "...",
      "time_gt": {"start": 0.83, "end": 19.86},
      "time_duration": 19.03,
      "source_index": 0,
      "source_sentence_index": 0
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
from typing import Dict, List, Optional, Tuple

from common import ensure_parent_dir, round_float

logger = logging.getLogger(__name__)

Segment = Tuple[float, float]


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ActivityNet Captions annotations to per-video VTG format")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input ActivityNet Captions annotation JSON path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_pipeline/vtg_data_cleaning/output/activitynet_captions/activitynet_captions_val1_by_video.json",
        help="Output grouped JSON path",
    )
    parser.add_argument("--round-digits", type=int, default=-1, help="Round floats to N digits (-1 keeps raw float)")
    parser.add_argument("--clip-to-duration", default=True, help="Clip timestamp end by row `duration` when possible")
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


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _valid_range(start: float, end: float) -> bool:
    return start >= 0 and end > start


def _parse_segment(raw_segment: object) -> Optional[Segment]:
    if not isinstance(raw_segment, list) or len(raw_segment) != 2:
        return None
    start = _safe_float(raw_segment[0])
    end = _safe_float(raw_segment[1])
    if start is None or end is None:
        return None
    return start, end


def convert_rows(
    rows: List[Dict],
    round_digits: int,
    clip_to_duration: bool = False,
    sort_videos: bool = False,
) -> Dict:
    videos = defaultdict(lambda: {"video_id": "", "video_name": "", "query_count": 0, "queries": []})

    total_rows = len(rows)
    total_candidate_queries = 0
    dropped_missing_video = 0
    dropped_invalid_structure = 0
    dropped_missing_query = 0
    dropped_invalid_time = 0
    rows_with_pair_mismatch = 0
    dropped_mismatched_pairs = 0

    for row_idx, row in enumerate(rows):
        video_id = _as_non_empty_text(row.get("video_id"))
        video_name = _as_non_empty_text(row.get("video"))
        if not video_id and video_name:
            video_id = Path(video_name).stem
        if not video_name and video_id:
            video_name = f"{video_id}.mp4"

        if not video_id:
            dropped_missing_video += 1
            continue

        sentences = row.get("sentences")
        timestamps = row.get("timestamps")
        if not isinstance(sentences, list) or not isinstance(timestamps, list):
            dropped_invalid_structure += 1
            continue

        if len(sentences) != len(timestamps):
            rows_with_pair_mismatch += 1
            dropped_mismatched_pairs += abs(len(sentences) - len(timestamps))

        pair_count = min(len(sentences), len(timestamps))
        total_candidate_queries += pair_count
        if pair_count <= 0:
            continue

        duration = _safe_float(row.get("duration"))
        if duration is not None and duration <= 0:
            duration = None

        current = videos[video_id]
        if not current["video_id"]:
            current["video_id"] = video_id
        if not current["video_name"]:
            current["video_name"] = video_name

        for sentence_idx in range(pair_count):
            query = _as_non_empty_text(sentences[sentence_idx])
            if not query:
                dropped_missing_query += 1
                continue

            segment = _parse_segment(timestamps[sentence_idx])
            if segment is None:
                dropped_invalid_time += 1
                continue
            start, end = segment

            if clip_to_duration and duration is not None:
                start = max(0.0, start)
                end = min(end, duration)

            if not _valid_range(start, end):
                dropped_invalid_time += 1
                continue

            query_index = len(current["queries"])
            query_item = {
                "query_id": f"{video_id}#{query_index:06d}",
                "query": query,
                "time_gt": {
                    "start": round_float(start, round_digits),
                    "end": round_float(end, round_digits),
                },
                "time_duration": round_float(end - start, round_digits),
                "source_index": row_idx,
                "source_sentence_index": sentence_idx,
            }
            current["queries"].append(query_item)

    output_videos = list(videos.values())
    for video in output_videos:
        video["query_count"] = len(video["queries"])

    if sort_videos:
        output_videos = sorted(output_videos, key=lambda item: item["video_id"])

    kept_queries = sum(video["query_count"] for video in output_videos)
    stats = {
        "total_rows": total_rows,
        "total_candidate_queries": total_candidate_queries,
        "kept_queries": kept_queries,
        "dropped_queries": total_candidate_queries - kept_queries,
        "dropped_missing_video": dropped_missing_video,
        "dropped_invalid_structure": dropped_invalid_structure,
        "dropped_missing_query": dropped_missing_query,
        "dropped_invalid_time": dropped_invalid_time,
        "rows_with_pair_mismatch": rows_with_pair_mismatch,
        "dropped_mismatched_pairs": dropped_mismatched_pairs,
        "video_count": len(output_videos),
    }

    return {
        "dataset": "activitynet_captions",
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
        raise ValueError("Input JSON must be a list of ActivityNet video entries")

    logger.info("Converting %d rows...", len(rows))
    converted = convert_rows(
        rows,
        round_digits=args.round_digits,
        clip_to_duration=args.clip_to_duration,
        sort_videos=args.sort_videos,
    )
    converted["source_path"] = str(input_path)

    ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=None if args.compact else 2)

    logger.info("Saved: %s", output_path)
    logger.info("Stats: %s", converted["stats"])


if __name__ == "__main__":
    main()
