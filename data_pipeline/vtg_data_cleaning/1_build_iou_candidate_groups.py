#!/usr/bin/env python3
"""
Build potential duplicate query groups per video using time_gt IoU threshold.

The script reads per-video VTG data (from convert_charades_sta_to_video_queries.py),
then forms candidate groups by:
1) building an overlap graph where edge(i, j) exists if IoU(time_gt_i, time_gt_j) >= threshold
2) extracting groups as maximal cliques (default) or connected components
3) attaching the maximal common intersection segment for each group
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from common import (
    bron_kerbosch_maximal_cliques,
    calculate_iou,
    connected_components,
    ensure_parent_dir,
    round_float,
    segment_intersection,
    segment_union,
)

logger = logging.getLogger(__name__)


Segment = Tuple[float, float]


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build IoU-based candidate duplicate groups for VTG queries")
    parser.add_argument(
        "--input",
        type=str,
        # default="data_pipeline/vtg_data_cleaning/output/charades_sta_test_by_video.json",
        default="data_pipeline/vtg_data_cleaning/output/activitynet_captions/activitynet_captions_val1_by_video.json",

        help="Input per-video JSON path",
    )
    parser.add_argument(
        "--output",
        type=str,
        # default="data_pipeline/vtg_data_cleaning/output/charades_sta_test_iou_groups_iou0p99.json",
        default="data_pipeline/vtg_data_cleaning/output/activitynet_captions/activitynet_captions_val1__iou_groups_iou0p99.json",

        help="Output group JSON path",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.99, help="IoU threshold for candidate edges")
    parser.add_argument("--min-group-size", type=int, default=2, help="Minimum group size")
    parser.add_argument(
        "--group-mode",
        type=str,
        default="clique",
        choices=["clique", "connected"],
        help="Group extraction mode: strict clique or looser connected component",
    )
    parser.add_argument("--round-digits", type=int, default=3, help="Round float fields to N digits")
    parser.add_argument("--skip-empty", action="store_true", help="Skip videos with no candidate groups")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def _get_segment(query_item: Dict) -> Optional[Segment]:
    time_gt = query_item.get("time_gt")
    if not isinstance(time_gt, dict):
        return None
    start = time_gt.get("start")
    end = time_gt.get("end")
    try:
        start_f = float(start)
        end_f = float(end)
    except (TypeError, ValueError):
        return None
    if start_f < 0 or end_f <= start_f:
        return None
    return start_f, end_f


def _segment_to_dict(segment: Optional[Segment], round_digits: int) -> Optional[Dict[str, float]]:
    if segment is None:
        return None
    start, end = segment
    return {
        "start": round_float(start, round_digits),
        "end": round_float(end, round_digits),
        "duration": round_float(end - start, round_digits),
    }


def _query_record(query: Dict, video_id: str, query_idx: int) -> Dict:
    return {
        "query_id": query.get("query_id", f"{video_id}#{query_idx:06d}"),
        "query": query.get("query", ""),
        "time_gt": query.get("time_gt"),
        "time_duration": query.get("time_duration"),
    }


def build_video_groups(
    video_item: Dict,
    iou_threshold: float,
    min_group_size: int,
    group_mode: str,
    round_digits: int,
) -> Dict:
    video_id = video_item.get("video_id", "")
    video_name = video_item.get("video_name", "")
    queries = video_item.get("queries", [])
    if not isinstance(queries, list):
        queries = []

    valid_segments: Dict[int, Segment] = {}
    for idx, query in enumerate(queries):
        segment = _get_segment(query)
        if segment is not None:
            valid_segments[idx] = segment

    adj: Dict[int, Set[int]] = {idx: set() for idx in valid_segments.keys()}
    iou_map: Dict[Tuple[int, int], float] = {}
    valid_indices = sorted(valid_segments.keys())
    for left in range(len(valid_indices)):
        i = valid_indices[left]
        for right in range(left + 1, len(valid_indices)):
            j = valid_indices[right]
            iou = calculate_iou(valid_segments[i], valid_segments[j])
            if iou >= iou_threshold:
                adj[i].add(j)
                adj[j].add(i)
                iou_map[(i, j)] = iou

    if group_mode == "connected":
        raw_groups = connected_components(adj, min_size=min_group_size)
    else:
        raw_groups = bron_kerbosch_maximal_cliques(adj, min_size=min_group_size)

    groups = []
    covered_indices: Set[int] = set()
    query_id_by_index = {
        idx: queries[idx].get("query_id", f"{video_id}#{idx:06d}")
        for idx in range(len(queries))
    }
    for group_id, member_indices in enumerate(raw_groups):
        covered_indices.update(member_indices)

        pairwise_iou = []
        iou_values: List[float] = []
        member_segments = [valid_segments[idx] for idx in member_indices if idx in valid_segments]

        for i_pos in range(len(member_indices)):
            for j_pos in range(i_pos + 1, len(member_indices)):
                i = member_indices[i_pos]
                j = member_indices[j_pos]
                iou = iou_map.get((i, j))
                if iou is None:
                    iou = iou_map.get((j, i))
                if iou is None and i in valid_segments and j in valid_segments:
                    iou = calculate_iou(valid_segments[i], valid_segments[j])
                if iou is None:
                    iou = 0.0
                iou_values.append(iou)
                pairwise_iou.append(
                    {
                        "left_query_id": query_id_by_index.get(i),
                        "right_query_id": query_id_by_index.get(j),
                        "iou": round_float(iou, round_digits),
                    }
                )

        max_intersection = segment_intersection(member_segments)
        union_segment = segment_union(member_segments)

        members = []
        for global_idx in member_indices:
            query = queries[global_idx]
            members.append(_query_record(query, video_id, global_idx))

        group_stats = {
            "member_count": len(member_indices),
            "pair_count": len(pairwise_iou),
            "min_pair_iou": round_float(min(iou_values), round_digits) if iou_values else None,
            "max_pair_iou": round_float(max(iou_values), round_digits) if iou_values else None,
            "avg_pair_iou": round_float(sum(iou_values) / len(iou_values), round_digits) if iou_values else None,
        }

        groups.append(
            {
                "group_id": group_id,
                "members": members,
                "max_intersection_segment": _segment_to_dict(max_intersection, round_digits),
                "union_segment": _segment_to_dict(union_segment, round_digits),
                "pairwise_iou": pairwise_iou,
                "stats": group_stats,
            }
        )

    ungrouped_query = [
        _query_record(queries[idx], video_id, idx)
        for idx in sorted(set(range(len(queries))) - covered_indices)
    ]

    return {
        "video_id": video_id,
        "video_name": video_name,
        "query_count": len(queries),
        "valid_segment_query_count": len(valid_segments),
        "group_count": len(groups),
        "ungrouped_query_count": len(ungrouped_query),
        "ungrouped_query": ungrouped_query,
        "groups": groups,
    }


def _load_videos(data: object) -> List[Dict]:
    if isinstance(data, dict) and isinstance(data.get("videos"), list):
        return data["videos"]
    if isinstance(data, list):
        return data
    raise ValueError("Input JSON must be a dict with `videos` or a list of videos")


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    input_path = Path(args.input)
    output_path = Path(args.output)

    logger.info("Loading input: %s", input_path)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    videos = _load_videos(data)
    logger.info("Total videos: %d", len(videos))

    output_videos: List[Dict] = []
    total_groups = 0
    for video in videos:
        result = build_video_groups(
            video,
            iou_threshold=args.iou_threshold,
            min_group_size=args.min_group_size,
            group_mode=args.group_mode,
            round_digits=args.round_digits,
        )
        total_groups += result["group_count"]
        if args.skip_empty and result["group_count"] == 0:
            continue
        output_videos.append(result)

    summary = {
        "source_path": str(input_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "iou_threshold": args.iou_threshold,
            "min_group_size": args.min_group_size,
            "group_mode": args.group_mode,
            "round_digits": args.round_digits,
            "skip_empty": args.skip_empty,
        },
        "stats": {
            "input_video_count": len(videos),
            "output_video_count": len(output_videos),
            "total_group_count": total_groups,
            "videos_with_groups": sum(1 for v in output_videos if v["group_count"] > 0),
        },
        "videos": output_videos,
    }

    ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Saved: %s", output_path)
    logger.info("Stats: %s", summary["stats"])


if __name__ == "__main__":
    main()
