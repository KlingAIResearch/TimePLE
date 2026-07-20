from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

GROUNDING_PROMPT_TEMPLATE = """You are an expert in video temporal grounding. Based on the video content and the Query, first provide an objective understanding of the video
content, then give the accurate temporal segment for the Query in the video.
Query: {query}
Requirements:
1) event_timeline is the model's objective understanding of the entire video content and must be based only on video evidence.
2) event_timeline is a list. Each event must contain description, start_time, and end_time.
3) event_timeline should cover the major events in the video in chronological order, with temporal boundaries as accurate as possible.
4) query_prediction is the final temporal localization prediction for the original Query and should best match the Query.
5) If the Query does not match the video content, set query_prediction.start_time and query_prediction.end_time to null.
6) All timestamps must use absolute seconds on the video timeline.
Output format:
{
"event_timeline": [
{
"description": "...",
"start_time": xx.xx,
"end_time": xx.xx
}
],
"query_prediction": {
"start_time": xx.xx,
"end_time": xx.xx
}
}
"""

_CODE_BLOCK_JSON_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)


def read_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no} in {file_path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_no} in {file_path} is not a JSON object.")
            yield line_no, row


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]], append: bool = False) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with file_path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_seconds(token: Any) -> float | None:
    if token is None:
        return None
    if isinstance(token, (int, float)):
        value = float(token)
        return value if math.isfinite(value) else None
    text = str(token).strip().lower()
    text = text.rstrip(",.;")
    text = text.replace("seconds", "").replace("second", "").replace("secs", "").replace("sec", "")
    text = text.replace("s", "") if text.endswith("s") and ":" not in text else text
    text = text.strip()
    if not text:
        return None

    if ":" in text:
        parts = text.split(":")
        if len(parts) not in (2, 3):
            return None
        try:
            parts_float = [float(part) for part in parts]
        except ValueError:
            return None
        if len(parts_float) == 2:
            minutes, seconds = parts_float
            return minutes * 60.0 + seconds
        hours, minutes, seconds = parts_float
        return hours * 3600.0 + minutes * 60.0 + seconds

    text = text.replace(",", ".")
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def normalize_intervals(raw: Any) -> list[list[float]]:
    pairs: list[tuple[float, float]] = []

    if raw is None:
        return []

    if isinstance(raw, str):
        try:
            return normalize_intervals(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            return []

    if isinstance(raw, dict):
        start = next((raw.get(key) for key in ("start", "start_time", "s") if raw.get(key) is not None), None)
        end = next((raw.get(key) for key in ("end", "end_time", "e") if raw.get(key) is not None), None)
        interval = _clean_interval(start, end)
        return [interval] if interval else []

    if isinstance(raw, Sequence):
        if len(raw) == 2 and _is_scalar(raw[0]) and _is_scalar(raw[1]):
            interval = _clean_interval(raw[0], raw[1])
            return [interval] if interval else []

        scalar_buffer: list[Any] = []
        for item in raw:
            if isinstance(item, dict):
                interval = _clean_interval(
                    item.get("start") or item.get("start_time") or item.get("s"),
                    item.get("end") or item.get("end_time") or item.get("e"),
                )
                if interval:
                    pairs.append((interval[0], interval[1]))
                continue

            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                if len(item) >= 2:
                    interval = _clean_interval(item[0], item[1])
                    if interval:
                        pairs.append((interval[0], interval[1]))
                continue

            if _is_scalar(item):
                scalar_buffer.append(item)

        if scalar_buffer and len(scalar_buffer) % 2 == 0:
            for idx in range(0, len(scalar_buffer), 2):
                interval = _clean_interval(scalar_buffer[idx], scalar_buffer[idx + 1])
                if interval:
                    pairs.append((interval[0], interval[1]))

    merged = merge_intervals([[start, end] for start, end in pairs])
    return merged


def normalize_event_timeline(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []

    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue

        event_id = item.get("event_id", idx)
        event_description = _to_text(
            item.get("event_description")
            or item.get("description")
            or item.get("event")
            or item.get("text")
            or ""
        )
        start_seconds = to_seconds(next(
            (item.get(key) for key in ("start_time", "start_seconds", "start", "s") if item.get(key) is not None),
            None,
        ))
        end_seconds = to_seconds(next(
            (item.get(key) for key in ("end_time", "end_seconds", "end", "e") if item.get(key) is not None),
            None,
        ))

        if start_seconds is None or end_seconds is None:
            continue
        if not (math.isfinite(start_seconds) and math.isfinite(end_seconds)):
            continue
        if start_seconds < 0 or end_seconds < 0:
            continue
        if end_seconds < start_seconds:
            start_seconds, end_seconds = end_seconds, start_seconds
        if end_seconds == start_seconds:
            continue

        normalized.append(
            {
                "description": event_description,
                "start_time": round(float(start_seconds), 6),
                "end_time": round(float(end_seconds), 6),
            }
        )

    return normalized


def temporal_iou(gt_intervals: Any, pred_intervals: Any) -> float:
    gt = merge_intervals(normalize_intervals(gt_intervals))
    pred = merge_intervals(normalize_intervals(pred_intervals))

    if not gt and not pred:
        return 1.0
    if not gt or not pred:
        return 0.0

    intersection = _intersection_length(gt, pred)
    union = total_interval_length(gt) + total_interval_length(pred) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def total_interval_length(intervals: Sequence[Sequence[float]]) -> float:
    return sum(max(0.0, float(end) - float(start)) for start, end in intervals)


def merge_intervals(intervals: Sequence[Sequence[float]], tolerance: float = 1e-8) -> list[list[float]]:
    cleaned: list[list[float]] = []
    for item in intervals:
        if len(item) < 2:
            continue
        interval = _clean_interval(item[0], item[1])
        if interval:
            cleaned.append(interval)

    if not cleaned:
        return []

    cleaned.sort(key=lambda pair: pair[0])
    merged: list[list[float]] = [cleaned[0]]
    for start, end in cleaned[1:]:
        last = merged[-1]
        if start <= last[1] + tolerance:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return merged


def build_grounding_prompt(query: str, prompt_template: str | None = None) -> str:
    query_text = query.strip()
    template = (prompt_template or GROUNDING_PROMPT_TEMPLATE).strip()
    if "{query}" in template:
        return template.replace("{query}", query_text)
    return f"{template}\n\nQuery:\n{query_text}"


def parse_model_output_with_metadata(text: str) -> dict[str, Any]:
    normalized_text = (text or "").strip()
    if not normalized_text:
        return {
            "intervals": [],
            "raw_text": "",
            "reason": "",
            "event_timeline": [],
            "refined_query": "",
            "payload": None,
        }

    payload = _extract_json_payload(normalized_text)
    intervals: list[list[float]] = []
    reason = ""
    event_timeline: list[dict[str, Any]] = []
    refined_query = ""

    if payload:
        interval_candidates = (
            payload.get("query_prediction")
            or payload.get("refined_segment")
        )
        intervals = normalize_intervals(interval_candidates)
        reason = _to_text(payload.get("reason") or "")
        event_timeline = normalize_event_timeline(payload.get("event_timeline"))
        refined_query = _to_text(payload.get("refined_query") or "")

    return {
        "intervals": intervals,
        "raw_text": normalized_text,
        "reason": reason,
        "event_timeline": event_timeline,
        "refined_query": refined_query,
        "payload": payload,
    }


def count_jsonl_lines(path: str | Path) -> int:
    file_path = Path(path)
    if not file_path.exists():
        return 0
    with file_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    candidates: list[str] = []
    for match in _CODE_BLOCK_JSON_RE.finditer(text):
        candidates.append(match.group(1))

    bracket_json = _extract_first_balanced_json(text)
    if bracket_json:
        candidates.append(bracket_json)

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _extract_first_balanced_json(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _intersection_length(a_intervals: Sequence[Sequence[float]], b_intervals: Sequence[Sequence[float]]) -> float:
    i = 0
    j = 0
    overlap = 0.0
    while i < len(a_intervals) and j < len(b_intervals):
        a_start, a_end = a_intervals[i]
        b_start, b_end = b_intervals[j]
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        if end > start:
            overlap += end - start
        if a_end < b_end:
            i += 1
        else:
            j += 1
    return overlap


def _clean_interval(start_token: Any, end_token: Any) -> list[float] | None:
    start = to_seconds(start_token)
    end = to_seconds(end_token)
    if start is None or end is None:
        return None
    if not (math.isfinite(start) and math.isfinite(end)):
        return None
    if end < start:
        start, end = end, start
    if end == start:
        return None
    if start < 0 or end < 0:
        return None
    return [round(start, 6), round(end, 6)]


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, str))
