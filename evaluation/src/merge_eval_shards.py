#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import load_yaml_with_base
from metrics_loader import load_evaluation_metrics_class, load_qvhighlights_metrics_class
from parsing import parse_timestamp_response


EvaluationMetrics = load_evaluation_metrics_class(PROJECT_ROOT)
QVHighlightsMetrics = load_qvhighlights_metrics_class(PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge eval_suite shard outputs.")
    parser.add_argument("--config", required=True, help="Rendered job config YAML.")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Concrete timestamped output dir. If omitted, uses `output.resume_from` or latest run under `output.dir`.",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _resolve_run_dir(cfg: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override).resolve()
    resume_from = cfg.get("output", {}).get("resume_from")
    if resume_from:
        return Path(str(resume_from)).resolve()

    root = Path(str(cfg["output"]["dir"])).resolve()
    candidates = [path for path in root.iterdir() if path.is_dir()] if root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {root}")
    return sorted(candidates)[-1]


def _as_intervals(value: Any) -> list[tuple[float, float]] | None:
    if value is None:
        return None
    return [(float(start), float(end)) for start, end in value]


def _jsonable_intervals(value: list[tuple[float, float]] | None) -> list[list[float]] | None:
    if value is None:
        return None
    return [[float(start), float(end)] for start, end in value]


def _build_metrics(cfg: dict[str, Any]) -> Any:
    metrics_cfg = dict(cfg.get("metrics", {}))
    metrics_profile = str(metrics_cfg.get("profile", "")).strip().lower()
    dataset_name = str(cfg.get("dataset", {}).get("name", "")).strip().lower()

    if metrics_profile == "qvhighlights" or dataset_name.startswith("qvhighlights"):
        return QVHighlightsMetrics(
            recall_thresholds=list(metrics_cfg.get("recall_thresholds", [0.5, 0.7])),
            map_thresholds=list(metrics_cfg.get("map_thresholds", [0.5, 0.75])),
        )

    return EvaluationMetrics(
        iou_thresholds=list(metrics_cfg.get("iou_thresholds", [0.3, 0.5, 0.7])),
        gt_duration_bucket_edges=metrics_cfg.get("gt_duration_bucket_edges"),
    )


def _repair_prediction_row(row: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if row.get("pred_timestamps") is not None:
        return row, False
    if row.get("error"):
        return row, False

    response_text = str(row.get("response_text") or "").strip()
    if not response_text:
        return row, False

    repaired_pred_timestamps = parse_timestamp_response(response_text)
    if repaired_pred_timestamps is None:
        return row, False

    repaired_row = dict(row)
    repaired_row["pred_timestamps"] = _jsonable_intervals(repaired_pred_timestamps)
    repaired_row["pred_timestamps_source"] = "reparsed_response_text"
    return repaired_row, True


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_yaml_with_base(config_path)
    run_dir = _resolve_run_dir(cfg, args.run_dir)

    shard_paths = sorted(run_dir.glob("predictions.rank*.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard prediction files found in {run_dir}")

    merged_rows: list[dict[str, Any]] = []
    repaired_rows = 0
    for shard_path in shard_paths:
        for row in _read_jsonl(shard_path):
            repaired_row, was_repaired = _repair_prediction_row(row)
            merged_rows.append(repaired_row)
            if was_repaired:
                repaired_rows += 1
    merged_rows.sort(key=lambda row: row.get("global_index", 0))

    metrics = _build_metrics(cfg)
    for row in merged_rows:
        pred_timestamps = _as_intervals(row.get("pred_timestamps"))
        gt_timestamps = _as_intervals(row.get("gt_timestamps")) or []
        if isinstance(metrics, QVHighlightsMetrics):
            metrics.add(pred_timestamps, None, gt_timestamps)
            segment_ious = None
        else:
            segment_ious = metrics.add(pred_timestamps, gt_timestamps)
        row["segment_ious"] = segment_ious

    merged_predictions_path = run_dir / "predictions.merged.jsonl"
    with merged_predictions_path.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = metrics.get_summary()
    summary["total_prediction_rows"] = len(merged_rows)
    summary["repaired_prediction_rows"] = repaired_rows
    summary["run_dir"] = str(run_dir)
    summary["model_name"] = cfg["model"]["name"]
    summary["dataset_name"] = cfg["dataset"]["name"]

    merged_metrics_path = run_dir / "metrics.merged.json"
    with merged_metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Merged predictions: {merged_predictions_path}")
    print(f"Merged metrics: {merged_metrics_path}")


if __name__ == "__main__":
    main()
