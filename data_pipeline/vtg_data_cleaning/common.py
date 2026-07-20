#!/usr/bin/env python3
"""Common utilities for VTG dataset cleaning tools."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


Segment = Tuple[float, float]


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def round_float(value: float, digits: int) -> float:
    value = float(value)
    if digits < 0:
        return value
    return round(value, digits)


def calculate_iou(seg1: Segment, seg2: Segment) -> float:
    start1, end1 = seg1
    start2, end2 = seg2
    intersection = max(0.0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    if union <= 0:
        return 0.0
    return intersection / union


def segment_intersection(segments: List[Segment]) -> Optional[Segment]:
    if not segments:
        return None
    start = max(seg[0] for seg in segments)
    end = min(seg[1] for seg in segments)
    if end <= start:
        return None
    return start, end


def segment_union(segments: List[Segment]) -> Optional[Segment]:
    if not segments:
        return None
    start = min(seg[0] for seg in segments)
    end = max(seg[1] for seg in segments)
    if end <= start:
        return None
    return start, end


def bron_kerbosch_maximal_cliques(adj: Dict[int, Set[int]], min_size: int = 2) -> List[List[int]]:
    """Find maximal cliques from an undirected graph."""
    cliques: List[List[int]] = []
    nodes = set(adj.keys())

    def _bk(r: Set[int], p: Set[int], x: Set[int]) -> None:
        if not p and not x:
            if len(r) >= min_size:
                cliques.append(sorted(r))
            return
        if not p:
            return
        pivot_candidates = p | x
        pivot = None
        if pivot_candidates:
            pivot = max(pivot_candidates, key=lambda n: len(adj.get(n, set())))
        excluded = adj.get(pivot, set()) if pivot is not None else set()
        for v in list(p - excluded):
            _bk(r | {v}, p & adj.get(v, set()), x & adj.get(v, set()))
            p.remove(v)
            x.add(v)

    _bk(set(), set(nodes), set())
    return sorted(cliques, key=lambda g: (-len(g), g))


def connected_components(adj: Dict[int, Set[int]], min_size: int = 2) -> List[List[int]]:
    components: List[List[int]] = []
    visited: Set[int] = set()
    for node in sorted(adj.keys()):
        if node in visited:
            continue
        stack = [node]
        component: List[int] = []
        visited.add(node)
        while stack:
            cur = stack.pop()
            component.append(cur)
            for nxt in adj.get(cur, set()):
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        if len(component) >= min_size:
            components.append(sorted(component))
    return sorted(components, key=lambda g: (-len(g), g))


def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None
