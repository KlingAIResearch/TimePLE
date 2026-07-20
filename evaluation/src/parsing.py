from __future__ import annotations

import logging
import re


LOGGER = logging.getLogger(__name__)


def parse_time_to_seconds(time_str: str) -> float | None:
    time_str = time_str.strip()

    if ":" not in time_str:
        try:
            return float(time_str)
        except ValueError:
            return None

    parts = time_str.split(":")
    try:
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
    except (ValueError, IndexError):
        return None
    return None


def parse_timestamp_response(response: str) -> list[tuple[float, float]] | None:
    if not response:
        return None

    timestamps: list[tuple[float, float]] = []

    patterns = [
        r"\[(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)\s*,\s*(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)\]",
        r"\[(\d{1,2}:\d{2}(?:\.\d+)?)\s*,\s*(\d{1,2}:\d{2}(?:\.\d+)?)\]",
        r"\[\s*(\d+\.?\d*)\s*(?:s|sec|secs|second|seconds)?\s*,\s*(\d+\.?\d*)\s*(?:s|sec|secs|second|seconds)?\s*\]",
        r"\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\]",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response, flags=re.IGNORECASE)
        for match in matches:
            if len(match) != 2:
                continue
            left, right = match
            start = parse_time_to_seconds(left) if ":" in left else float(left)
            end = parse_time_to_seconds(right) if ":" in right else float(right)
            if start is None or end is None:
                continue
            if start >= 0 and end > start:
                timestamps.append((float(start), float(end)))
        if timestamps:
            return timestamps

    fallback_patterns = [
        r"(\d+\.?\d*)\s*(?:s|seconds)?\s*[-~to]+\s*(\d+\.?\d*)\s*(?:s|seconds)?",
        r"(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)\s*[-~to]+\s*(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)",
    ]
    for pattern in fallback_patterns:
        matches = re.findall(pattern, response)
        for match in matches:
            if len(match) != 2:
                continue
            left, right = match
            start = parse_time_to_seconds(left)
            end = parse_time_to_seconds(right)
            if start is None or end is None:
                continue
            if start >= 0 and end > start:
                timestamps.append((float(start), float(end)))
        if timestamps:
            return timestamps

    LOGGER.debug("Failed to parse timestamp response: %s", response)
    return None
