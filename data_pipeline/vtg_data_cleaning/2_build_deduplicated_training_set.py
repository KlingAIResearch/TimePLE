#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from common import ensure_parent_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic IoU-deduplicated VTG dataset.")
    parser.add_argument("--input-by-video", required=True)
    parser.add_argument("--input-groups", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--representative-policy",
        choices=("longest_query", "longest_duration", "first"),
        default="longest_query",
    )
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def query_id(query: dict[str, Any], index: int) -> str:
    return str(query.get("query_id") or query.get("id") or index)


def duration(query: dict[str, Any]) -> float:
    segment = query.get("time_gt")
    if not isinstance(segment, dict):
        return 0.0
    try:
        return max(0.0, float(segment.get("end")) - float(segment.get("start")))
    except (TypeError, ValueError):
        return 0.0


def representative_key(item: tuple[int, dict[str, Any]], policy: str) -> tuple[Any, ...]:
    index, query = item
    if policy == "longest_query":
        return (-len(str(query.get("query", "")).strip()), -duration(query), index)
    if policy == "longest_duration":
        return (-duration(query), -len(str(query.get("query", "")).strip()), index)
    return (index,)


def group_member_ids(group_video: dict[str, Any]) -> list[list[str]]:
    result: list[list[str]] = []
    for group in group_video.get("groups", []):
        if not isinstance(group, dict):
            continue
        ids: list[str] = []
        for member in group.get("members", []):
            if not isinstance(member, dict):
                continue
            value = member.get("query_id")
            if value is not None:
                ids.append(str(value))
        if len(ids) > 1:
            result.append(ids)
    return result


def connected_components(groups: list[list[str]]) -> list[set[str]]:
    adjacency: dict[str, set[str]] = {}
    for group in groups:
        members = set(group)
        for member in members:
            adjacency.setdefault(member, set()).update(members - {member})
    components: list[set[str]] = []
    seen: set[str] = set()
    for node in sorted(adjacency):
        if node in seen:
            continue
        stack = [node]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.add(current)
            stack.extend(adjacency.get(current, set()) - seen)
        components.append(component)
    return components


def main() -> None:
    args = parse_args()
    source = load_json(args.input_by_video)
    grouped = load_json(args.input_groups)
    group_by_video = {
        str(video.get("video_id")): video
        for video in grouped.get("videos", [])
        if isinstance(video, dict)
    }
    output = copy.deepcopy(source)
    audit: list[dict[str, Any]] = []
    total_before = 0
    total_after = 0

    for video in output.get("videos", []):
        if not isinstance(video, dict):
            continue
        queries = [item for item in video.get("queries", []) if isinstance(item, dict)]
        total_before += len(queries)
        indexed = [(index, query) for index, query in enumerate(queries)]
        by_id = {query_id(query, index): (index, query) for index, query in indexed}
        components = connected_components(group_member_ids(group_by_video.get(str(video.get("video_id")), {})))
        dropped: set[str] = set()
        selections: list[dict[str, Any]] = []
        for component in components:
            candidates = [by_id[item_id] for item_id in component if item_id in by_id]
            if len(candidates) < 2:
                continue
            selected_index, selected_query = sorted(candidates, key=lambda item: representative_key(item, args.representative_policy))[0]
            selected_id = query_id(selected_query, selected_index)
            removed_ids = sorted(component - {selected_id})
            dropped.update(removed_ids)
            selections.append({"component": sorted(component), "selected_query_id": selected_id, "dropped_query_ids": removed_ids})
        video["queries"] = [query for index, query in indexed if query_id(query, index) not in dropped]
        video["query_count"] = len(video["queries"])
        total_after += len(video["queries"])
        if selections:
            audit.append({"video_id": video.get("video_id"), "selections": selections})

    output["deduplication"] = {
        "method": "deterministic_iou_components",
        "representative_policy": args.representative_policy,
        "input_query_count": total_before,
        "output_query_count": total_after,
        "dropped_query_count": total_before - total_after,
        "audit": audit,
    }
    output_path = Path(args.output)
    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] input={total_before} output={total_after} dropped={total_before - total_after}")


if __name__ == "__main__":
    main()
