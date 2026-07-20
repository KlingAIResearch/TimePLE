#!/usr/bin/env python3
from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Any

from pipeline_core import normalize_event_timeline, normalize_intervals, read_jsonl, temporal_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute teacher IoUs, filter by dual-teacher agreement, and construct event-consensus samples."
    )
    parser.add_argument("--input", required=True, help="JSONL from step2 inference.")
    parser.add_argument("--output", required=True, help="Audited JSONL containing existing and consensus samples.")
    parser.add_argument("--precision", type=int, default=6)
    parser.add_argument("--iou-alias", choices=("gemini", "qwen", "best"), default="gemini")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=2000)
    parser.add_argument(
        "--filter-thresholds",
        default="0.1,0.3,0.5,0.7",
        help="Each filtered file retains a sample only when both teacher IoUs reach its threshold.",
    )
    parser.add_argument("--disable-consensus", action="store_true")
    parser.add_argument(
        "--consensus-temporal-iou-threshold",
        type=float,
        default=0.5,
        help="Minimum cross-teacher event IoU for a new grounded-sample candidate.",
    )
    parser.add_argument(
        "--consensus-semantic-threshold",
        type=float,
        default=0.5,
        help="Minimum normalized description similarity for a new grounded-sample candidate.",
    )
    return parser.parse_args()


def parse_thresholds(raw: str) -> list[float]:
    thresholds = sorted({float(item.strip()) for item in raw.split(",") if item.strip()})
    if any(value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("IoU thresholds must be within [0, 1]")
    return thresholds


def threshold_output_path(output_path: Path, threshold: float) -> Path:
    tag = f"{threshold:g}".replace(".", "p")
    return output_path.with_name(f"{output_path.stem}.both_iou_ge_{tag}{output_path.suffix}")


def round_score(value: float, precision: int) -> float:
    return round(float(value), precision)


def event_description(event: dict[str, Any]) -> str:
    return str(event.get("description") or event.get("event_description") or event.get("event") or "").strip()


def event_interval(event: dict[str, Any]) -> list[list[float]]:
    return normalize_intervals(
        {
            "start_time": event.get("start_time", event.get("start_seconds")),
            "end_time": event.get("end_time", event.get("end_seconds")),
        }
    )


def description_similarity(left: str, right: str) -> float:
    left_norm = " ".join(re.findall(r"\w+", left.lower(), flags=re.UNICODE))
    right_norm = " ".join(re.findall(r"\w+", right.lower(), flags=re.UNICODE))
    if not left_norm or not right_norm:
        return 0.0
    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    sequence = SequenceMatcher(None, left_norm, right_norm).ratio()
    return max(jaccard, sequence)


def match_consensus_events(
    gemini_timeline: Any,
    qwen_timeline: Any,
    temporal_threshold: float,
    semantic_threshold: float,
) -> list[dict[str, Any]]:
    gemini_events = normalize_event_timeline(gemini_timeline)
    qwen_events = normalize_event_timeline(qwen_timeline)
    candidates: list[dict[str, Any]] = []

    for gemini_index, gemini_event in enumerate(gemini_events):
        gemini_span = event_interval(gemini_event)
        gemini_text = event_description(gemini_event)
        if not gemini_span or not gemini_text:
            continue
        for qwen_index, qwen_event in enumerate(qwen_events):
            qwen_span = event_interval(qwen_event)
            qwen_text = event_description(qwen_event)
            if not qwen_span or not qwen_text:
                continue
            event_iou = temporal_iou(gemini_span, qwen_span)
            semantic_score = description_similarity(gemini_text, qwen_text)
            if event_iou < temporal_threshold or semantic_score < semantic_threshold:
                continue
            candidates.append(
                {
                    "gemini_index": gemini_index,
                    "qwen_index": qwen_index,
                    "gemini_event": gemini_event,
                    "qwen_event": qwen_event,
                    "gemini_span": gemini_span,
                    "qwen_span": qwen_span,
                    "temporal_iou": event_iou,
                    "semantic_score": semantic_score,
                }
            )

    candidates.sort(key=lambda item: (-item["temporal_iou"], -item["semantic_score"], item["gemini_index"], item["qwen_index"]))
    used_gemini: set[int] = set()
    used_qwen: set[int] = set()
    matches: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate["gemini_index"] in used_gemini or candidate["qwen_index"] in used_qwen:
            continue
        used_gemini.add(candidate["gemini_index"])
        used_qwen.add(candidate["qwen_index"])
        matches.append(candidate)
    return matches


def build_consensus_row(parent: dict[str, Any], match: dict[str, Any], ordinal: int, precision: int) -> dict[str, Any]:
    gemini_span = match["gemini_span"][0]
    qwen_span = match["qwen_span"][0]
    consensus_span = [[
        round_score((gemini_span[0] + qwen_span[0]) / 2.0, precision),
        round_score((gemini_span[1] + qwen_span[1]) / 2.0, precision),
    ]]
    gemini_text = event_description(match["gemini_event"])
    qwen_text = event_description(match["qwen_event"])
    consensus_query = gemini_text if len(gemini_text) >= len(qwen_text) else qwen_text
    parent_id = str(parent.get("sample_id") or parent.get("id") or "sample")
    gemini_iou = temporal_iou(consensus_span, match["gemini_span"])
    qwen_iou = temporal_iou(consensus_span, match["qwen_span"])
    return {
        "source": f"{parent.get('source', 'unknown')}:teacher_consensus",
        "data_type": parent.get("data_type", "grounding"),
        "video_path": parent.get("video_path", ""),
        "query": consensus_query,
        "answer": "The event occurs at <TIME_STAMP>.",
        "time_gt": consensus_span,
        "sample_id": f"{parent_id}#teacher_consensus_{ordinal:03d}",
        "sample_kind": "teacher_event_consensus",
        "parent_sample_id": parent_id,
        "gemini3pro_pred_intervals": match["gemini_span"],
        "qwen3vl_30b_pred_intervals": match["qwen_span"],
        "iou_gemini3pro": round_score(gemini_iou, precision),
        "iou_qwen3vl_30b": round_score(qwen_iou, precision),
        "iou": round_score(min(gemini_iou, qwen_iou), precision),
        "best_model": "teacher_consensus",
        "best_iou": round_score(max(gemini_iou, qwen_iou), precision),
        "min_teacher_iou": round_score(min(gemini_iou, qwen_iou), precision),
        "teacher_consensus": {
            "gemini_description": gemini_text,
            "qwen_description": qwen_text,
            "cross_teacher_temporal_iou": round_score(match["temporal_iou"], precision),
            "description_similarity": round_score(match["semantic_score"], precision),
            "interval_policy": "mean_boundaries",
        },
    }


def passes_both_teachers(row: dict[str, Any], threshold: float) -> bool:
    return float(row.get("iou_gemini3pro", 0.0)) >= threshold and float(row.get("iou_qwen3vl_30b", 0.0)) >= threshold


def write_row(row: dict[str, Any], writer: Any, threshold_writers: dict[float, Any], threshold_counts: dict[float, int]) -> None:
    serialized = json.dumps(row, ensure_ascii=False) + "\n"
    writer.write(serialized)
    for threshold, threshold_writer in threshold_writers.items():
        if passes_both_teachers(row, threshold):
            threshold_writer.write(serialized)
            threshold_counts[threshold] += 1


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.consensus_temporal_iou_threshold <= 1.0:
        raise ValueError("--consensus-temporal-iou-threshold must be within [0, 1]")
    if not 0.0 <= args.consensus_semantic_threshold <= 1.0:
        raise ValueError("--consensus-semantic-threshold must be within [0, 1]")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = parse_thresholds(args.filter_thresholds)
    threshold_writers = {threshold: threshold_output_path(output_path, threshold).open("w", encoding="utf-8") for threshold in thresholds}
    threshold_counts = {threshold: 0 for threshold in thresholds}
    processed = 0
    consensus_count = 0

    try:
        with output_path.open("w", encoding="utf-8") as writer:
            for _, row in read_jsonl(input_path):
                if args.max_samples > 0 and processed >= args.max_samples:
                    break
                gt_intervals = normalize_intervals(row.get("time_gt"))
                gemini_intervals = normalize_intervals(row.get("gemini3pro_pred_intervals"))
                qwen_intervals = normalize_intervals(row.get("qwen3vl_30b_pred_intervals"))
                iou_gemini = temporal_iou(gt_intervals, gemini_intervals)
                iou_qwen = temporal_iou(gt_intervals, qwen_intervals)
                row["gemini3pro_pred_intervals"] = gemini_intervals
                row["qwen3vl_30b_pred_intervals"] = qwen_intervals
                row["iou_gemini3pro"] = round_score(iou_gemini, args.precision)
                row["iou_qwen3vl_30b"] = round_score(iou_qwen, args.precision)
                row["min_teacher_iou"] = round_score(min(iou_gemini, iou_qwen), args.precision)
                row["best_iou"] = round_score(max(iou_gemini, iou_qwen), args.precision)
                row["best_model"] = "gemini3pro" if iou_gemini >= iou_qwen else "qwen3vl_30b"
                row["iou"] = row[f"iou_{'gemini3pro' if args.iou_alias == 'gemini' else 'qwen3vl_30b'}"] if args.iou_alias != "best" else row["best_iou"]
                write_row(row, writer, threshold_writers, threshold_counts)

                if not args.disable_consensus:
                    matches = match_consensus_events(
                        row.get("gemini3pro_event_timeline"),
                        row.get("qwen3vl_30b_event_timeline"),
                        args.consensus_temporal_iou_threshold,
                        args.consensus_semantic_threshold,
                    )
                    for ordinal, match in enumerate(matches, start=1):
                        consensus_row = build_consensus_row(row, match, ordinal, args.precision)
                        write_row(consensus_row, writer, threshold_writers, threshold_counts)
                        consensus_count += 1

                processed += 1
                if args.progress_every > 0 and processed % args.progress_every == 0:
                    print(f"[INFO] processed={processed} consensus={consensus_count}")
    finally:
        for threshold_writer in threshold_writers.values():
            threshold_writer.close()

    print(f"[DONE] existing={processed} consensus={consensus_count} output={output_path}")
    for threshold in thresholds:
        print(f"[DONE] both_teacher_iou>={threshold:g} rows={threshold_counts[threshold]} output={threshold_output_path(output_path, threshold)}")


if __name__ == "__main__":
    main()
