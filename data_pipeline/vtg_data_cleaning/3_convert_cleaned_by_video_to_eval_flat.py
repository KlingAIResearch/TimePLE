#!/usr/bin/env python3
"""
Convert VTG cleaned by-video annotations to evaluator-consumable formats.

Input supports:
1) dict with key `videos`
2) list of video objects

Supported output formats:
- time_codec_jsonl (default):
  each line is a sample with fields `messages`, `videos`, `segment_gt`
- charades_flat_json:
  a flat JSON list with fields `video_id`, `video_name`, `query`,
  `start_time`, `end_time`, `duration`
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import ensure_parent_dir, round_float

logger = logging.getLogger(__name__)
Segment = Tuple[float, float]


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert cleaned by-video VTG JSON to time_codec JSONL or flat eval JSON"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data_pipeline/vtg_data_cleaning/output/charades_sta/3_cleaned_charades_sta_test_iou0p99_final.json",
        help="Input cleaned by-video JSON path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_pipeline/vtg_data_cleaning/output/charades_sta/6_cleaned_charades_sta_test_iou0p99_time_codec.jsonl",
        help="Output path (.jsonl for time_codec_jsonl, .json for charades_flat_json)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="time_codec_jsonl",
        choices=["time_codec_jsonl", "charades_flat_json"],
        help="Output format",
    )
    parser.add_argument(
        "--round-digits",
        type=int,
        default=3,
        help="Round float fields to N digits (-1 keeps raw float)",
    )
    parser.add_argument(
        "--sort-rows",
        action="store_true",
        help="Sort rows by (video_id, start_time, end_time, query)",
    )
    parser.add_argument(
        "--disable-video-tag",
        action="store_true",
        help="Do not prepend '<video>\\n' to user content in time_codec output",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Save compact JSON for charades_flat_json output",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def _as_non_empty_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_segment(query_item: Dict[str, Any]) -> Optional[Segment]:
    time_gt = query_item.get("time_gt")

    if isinstance(time_gt, dict):
        start = _as_float(time_gt.get("start"))
        end = _as_float(time_gt.get("end"))
        if start is not None and end is not None:
            return start, end

    if isinstance(time_gt, (list, tuple)) and len(time_gt) >= 2:
        start = _as_float(time_gt[0])
        end = _as_float(time_gt[1])
        if start is not None and end is not None:
            return start, end

    start = _as_float(query_item.get("start_time"))
    end = _as_float(query_item.get("end_time"))
    if start is not None and end is not None:
        return start, end

    segment_gt = query_item.get("segment_gt")
    if isinstance(segment_gt, dict):
        start = _as_float(segment_gt.get("start"))
        end = _as_float(segment_gt.get("end"))
        if start is not None and end is not None:
            return start, end

    if isinstance(segment_gt, (list, tuple)) and len(segment_gt) >= 2:
        start = _as_float(segment_gt[0])
        end = _as_float(segment_gt[1])
        if start is not None and end is not None:
            return start, end

    return None


def _valid_range(start: float, end: float) -> bool:
    return start >= 0 and end > start


def _load_videos(raw_data: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_data, dict):
        videos = raw_data.get("videos")
        if isinstance(videos, list):
            return videos
        raise ValueError("Input dict must contain key `videos` with a list value")

    if isinstance(raw_data, list):
        return raw_data

    raise ValueError("Input JSON must be a dict(with `videos`) or a list")


def convert_by_video_to_flat(
    videos: List[Dict[str, Any]],
    round_digits: int,
    sort_rows: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rows: List[Dict[str, Any]] = []

    total_videos = len(videos)
    total_queries = 0
    kept_rows = 0
    dropped_missing_video = 0
    dropped_missing_query = 0
    dropped_missing_segment = 0
    dropped_invalid_segment = 0
    dropped_invalid_video_object = 0
    dropped_invalid_query_object = 0

    for video_index, video in enumerate(videos):
        if not isinstance(video, dict):
            dropped_invalid_video_object += 1
            continue

        video_id = _as_non_empty_text(video.get("video_id"))
        video_name = _as_non_empty_text(video.get("video_name"))

        if not video_id and video_name:
            video_id = Path(video_name).stem
        if not video_name and video_id:
            video_name = f"{video_id}.mp4"

        queries = video.get("queries", [])
        if not isinstance(queries, list):
            continue

        for query_index, query_item in enumerate(queries):
            total_queries += 1

            if not isinstance(query_item, dict):
                dropped_invalid_query_object += 1
                continue

            if not video_id or not video_name:
                dropped_missing_video += 1
                continue

            query_text = _as_non_empty_text(query_item.get("query"))
            if not query_text:
                dropped_missing_query += 1
                continue

            segment = _extract_segment(query_item)
            if segment is None:
                dropped_missing_segment += 1
                continue

            start_time, end_time = segment
            if not _valid_range(start_time, end_time):
                dropped_invalid_segment += 1
                continue

            duration = _as_float(query_item.get("time_duration"))
            if duration is None or duration <= 0:
                duration = end_time - start_time

            row = {
                "video_id": video_id,
                "video_name": video_name,
                "query": query_text,
                "start_time": round_float(start_time, round_digits),
                "end_time": round_float(end_time, round_digits),
                "duration": round_float(duration, round_digits),
                "source_video_index": video_index,
                "source_query_index": query_index,
            }

            query_id = _as_non_empty_text(query_item.get("query_id"))
            if query_id:
                row["query_id"] = query_id

            source_index = query_item.get("source_index")
            if source_index is not None:
                row["source_index"] = source_index

            rows.append(row)
            kept_rows += 1

    if sort_rows:
        rows = sorted(
            rows,
            key=lambda item: (
                item.get("video_id", ""),
                item.get("start_time", 0.0),
                item.get("end_time", 0.0),
                item.get("query", ""),
            ),
        )

    stats = {
        "total_videos": total_videos,
        "total_queries": total_queries,
        "kept_rows": kept_rows,
        "dropped_rows": total_queries - kept_rows,
        "dropped_missing_video": dropped_missing_video,
        "dropped_missing_query": dropped_missing_query,
        "dropped_missing_segment": dropped_missing_segment,
        "dropped_invalid_segment": dropped_invalid_segment,
        "dropped_invalid_video_object": dropped_invalid_video_object,
        "dropped_invalid_query_object": dropped_invalid_query_object,
    }

    return rows, stats


def convert_flat_to_time_codec_jsonl_rows(
    rows: List[Dict[str, Any]],
    add_video_tag: bool = True,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []

    for row in rows:
        query_text = _as_non_empty_text(row.get("query"))
        if add_video_tag:
            user_content = f"<video>\n{query_text}"
        else:
            user_content = query_text

        sample: Dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": user_content,
                }
            ],
            "videos": [row["video_name"]],
            "segment_gt": [row["start_time"], row["end_time"]],
        }

        if row.get("video_id"):
            sample["video_id"] = row["video_id"]
        if row.get("query_id"):
            sample["query_id"] = row["query_id"]
        if row.get("source_index") is not None:
            sample["source_index"] = row["source_index"]
        if row.get("duration") is not None:
            sample["time_duration"] = row["duration"]

        samples.append(sample)

    return samples


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    input_path = Path(args.input)
    output_path = Path(args.output)

    logger.info("Loading input: %s", input_path)
    with input_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    videos = _load_videos(raw_data)
    logger.info("Loaded %d videos", len(videos))

    rows, stats = convert_by_video_to_flat(
        videos=videos,
        round_digits=args.round_digits,
        sort_rows=args.sort_rows,
    )

    ensure_parent_dir(output_path)
    if args.output_format == "time_codec_jsonl":
        samples = convert_flat_to_time_codec_jsonl_rows(
            rows=rows,
            add_video_tag=not args.disable_video_tag,
        )
        with output_path.open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False))
                f.write("\n")
        logger.info("Saved time_codec JSONL data: %s", output_path)
        logger.info("Generated samples: %d", len(samples))
    else:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=None if args.compact else 2)
        logger.info("Saved flat eval JSON data: %s", output_path)

    logger.info("Stats: %s", stats)


if __name__ == "__main__":
    main()
