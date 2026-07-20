from __future__ import annotations

import json
import math
import re
from typing import Any


REWARD_NAME = "timeple_iou_dfl_format_reward"
REWARD_TYPE = "batch"

ANSWER_BLOCK_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
STRICT_SINGLE_TIMESPAN_ANSWER_PATTERN = re.compile(
    r"^\s*<answer>\s*<\|TIMESPAN\|>\s*</answer>\s*$",
    re.IGNORECASE | re.DOTALL,
)
TIMESPAN_TOKEN_PATTERN = re.compile(re.escape("<|TIMESPAN|>"))


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return default
        if len(value) == 1:
            return _to_float(value[0], default=default)
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def _safe_json_loads(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return None


def _normalize_segments(value: Any) -> list[tuple[float, float]]:
    if value is None:
        return []
    if isinstance(value, str):
        maybe_json = _safe_json_loads(value)
        if maybe_json is not None:
            value = maybe_json
    if isinstance(value, dict):
        value = value.get("time_gt", value.get("segments", value.get("timestamps", [])))
    if not isinstance(value, (list, tuple)):
        return []

    segments: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            start = float(item[0])
            end = float(item[1])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        if end < start:
            start, end = end, start
        if end > start:
            segments.append((start, end))
    return segments


def _extract_answer_text(response: str) -> str:
    match = ANSWER_BLOCK_PATTERN.search(response)
    if match is None:
        return response
    return match.group(1)


def _timespan_count(response: str) -> int:
    return len(TIMESPAN_TOKEN_PATTERN.findall(_extract_answer_text(response)))


def _loss_to_reward(loss_value: Any, scale: float) -> float:
    loss = max(0.0, _to_float(loss_value))
    scale = max(float(scale), 1e-6)
    return _clamp(1.0 / (1.0 + loss / scale))


def _select_iou_score(
    *,
    decoded_iou: float,
    span_iou: float,
    decoded_valid: float,
    metrics_valid: float,
    iou_source: str,
) -> tuple[float, float]:
    source = iou_source.strip().lower()
    if source == "codec":
        return (span_iou, 2.0) if metrics_valid > 0.0 else (0.0, 0.0)
    if source == "best":
        candidates: list[tuple[float, float]] = []
        if decoded_valid > 0.0:
            candidates.append((decoded_iou, 1.0))
        if metrics_valid > 0.0:
            candidates.append((span_iou, 2.0))
        if not candidates:
            return 0.0, 0.0
        return max(candidates, key=lambda item: item[0])

    if decoded_valid > 0.0:
        return decoded_iou, 1.0
    if source in {"decoded_or_codec", "auto"} and metrics_valid > 0.0:
        return span_iou, 2.0
    return 0.0, 0.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    iou_weight: float = 0.80,
    dfl_weight: float = 0.20,
    format_weight: float = 0.30,
    dfl_scale: float = 2.5,
    iou_source: str = "decoded",
    exact_count_required: bool = False,
    invalid_localization_reward: float = 0.0,
) -> list[dict[str, float]]:
    scores: list[dict[str, float]] = []

    for reward_input in reward_inputs:
        response = str(reward_input.get("response", ""))
        gt_segments = _normalize_segments(reward_input.get("ground_truth"))
        gt_segment_count = len(gt_segments)
        response_timespan_count = _timespan_count(response)

        raw_timespan_count = reward_input.get("timeple_timespan_count")
        if raw_timespan_count is None:
            timespan_count = response_timespan_count
        else:
            timespan_count = int(round(_to_float(raw_timespan_count, default=0.0)))

        metrics_valid = 1.0 if _to_float(reward_input.get("timeple_metrics_valid"), default=0.0) > 0.5 else 0.0
        decoded_valid = 1.0 if _to_float(reward_input.get("timeple_decoded_valid"), default=0.0) > 0.5 else 0.0
        pred_valid = 1.0 if timespan_count > 0 else 0.0
        pred_single_valid = 1.0 if timespan_count == 1 else 0.0
        gt_valid = 1.0 if gt_segment_count == 1 else 0.0

        span_iou = _clamp(_to_float(reward_input.get("timeple_span_iou")))
        decoded_iou = _clamp(_to_float(reward_input.get("timeple_decoded_iou")))
        iou_raw, iou_source_id = _select_iou_score(
            decoded_iou=decoded_iou,
            span_iou=span_iou,
            decoded_valid=decoded_valid,
            metrics_valid=metrics_valid,
            iou_source=iou_source,
        )
        dfl_loss = max(0.0, _to_float(reward_input.get("timeple_dfl_loss")))
        dfl_reward = _loss_to_reward(dfl_loss, dfl_scale)

        localization_valid = gt_valid > 0.0 and pred_valid > 0.0 and iou_source_id > 0.0
        if exact_count_required:
            localization_valid = localization_valid and pred_single_valid > 0.0

        if localization_valid:
            iou_reward = _clamp(iou_raw)
            dfl_score = dfl_reward if metrics_valid > 0.0 else 0.0
        else:
            iou_reward = 0.0
            dfl_score = 0.0

        answer_has_timespan = 1.0 if TIMESPAN_TOKEN_PATTERN.search(_extract_answer_text(response)) else 0.0
        format_score = answer_has_timespan
        strict_format = 1.0 if STRICT_SINGLE_TIMESPAN_ANSWER_PATTERN.fullmatch(response) else 0.0

        localization_reward = (
            max(float(iou_weight), 0.0) * iou_reward
            + max(float(dfl_weight), 0.0) * dfl_score
        )
        if not localization_valid:
            localization_reward = _clamp(invalid_localization_reward)
        format_bonus = max(float(format_weight), 0.0) * format_score
        overall = localization_reward + format_bonus

        score = {
            "overall": overall,
            "localization": localization_reward,
            "miou": iou_reward,
            "dfl": dfl_score,
            "format": format_score,
            "format_bonus": format_bonus,
            "strict_answer_format": strict_format,
            "valid": 1.0 if localization_valid else 0.0,
            "pred_valid": pred_valid,
            "pred_single_valid": pred_single_valid,
            "gt_valid": gt_valid,
            "metrics_valid": metrics_valid,
            "decoded_valid": decoded_valid,
            "decode_source": _to_float(reward_input.get("timeple_decode_source")),
            "iou_source_id": iou_source_id,
            "timespan_count": float(timespan_count),
            "gt_segment_count": float(gt_segment_count),
            "answer_has_timespan": answer_has_timespan,
            "count_exact": 1.0 if timespan_count == 1 else 0.0,
            "timeple_dfl_loss": dfl_loss,
            "timeple_span_iou": span_iou,
            "timeple_decoded_iou": decoded_iou,
            "timeple_pred_start": _to_float(reward_input.get("timeple_pred_start")),
            "timeple_pred_end": _to_float(reward_input.get("timeple_pred_end")),
        }
        for key, value in reward_input.items():
            if key.startswith("timeple_debug_"):
                score[key] = _to_float(value)

        scores.append(score)

    return scores
