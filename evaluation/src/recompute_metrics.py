#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_BUCKET_EDGES = (0.0, 10.0, 30.0)
SELECTION_RULE = "max_valid_prediction_records_then_latest_run_timestamp"
RUN_METRICS_FILENAME = "metrics.recomputed_3gt_buckets.json"
BENCHMARK_SUMMARY_FILENAME = "metrics.summary.csv"
BENCHMARK_ALL_RUNS_FILENAME = "metrics.all_runs.csv"


@dataclass(frozen=True)
class RunInfo:
    suite_name: str
    model_name: str
    run_dir: Path
    prediction_paths: tuple[Path, ...]
    record_count: int
    timestamp: str
    run_mtime: float = 0.0


def _load_metrics_class(module_file: str, class_name: str):
    metrics_path = PROJECT_ROOT / "utils" / module_file
    module_name = f"eval_suite_recompute_{Path(module_file).stem}_module"
    spec = importlib.util.spec_from_file_location(module_name, metrics_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load metrics module from {metrics_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _load_evaluation_metrics_class():
    metrics_path = PROJECT_ROOT / "utils" / "metrics.py"
    spec = importlib.util.spec_from_file_location("eval_suite_recompute_metrics_module", metrics_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load metrics module from {metrics_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EvaluationMetrics


EvaluationMetrics = _load_evaluation_metrics_class()
QVHighlightsMetrics = _load_metrics_class("qvhighlights_metrics.py", "QVHighlightsMetrics")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _count_jsonl_records(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            json.loads(line)
            count += 1
    return count


def _count_json_array_records(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array in {path}")
    return len(payload)


def _count_prediction_records(path: Path) -> int:
    if path.suffix == ".jsonl":
        return _count_jsonl_records(path)
    return _count_json_array_records(path)


def _iter_jsonl_records(paths: Sequence[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected JSON object in {path}")
                yield payload


def _iter_json_array_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array in {path}")
    for record in payload:
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object in {path}")
        yield record


def _iter_prediction_records(paths: Sequence[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        if path.suffix == ".jsonl":
            yield from _iter_jsonl_records((path,))
        else:
            yield from _iter_json_array_records(path)


def _normalize_intervals(value: Any) -> list[tuple[float, float]]:
    if not value:
        return []

    intervals: list[tuple[float, float]] = []
    for item in value:
        if item is None:
            continue
        try:
            if len(item) < 2:
                continue
            start = float(item[0])
            end = float(item[1])
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        intervals.append((start, end))
    return intervals


def _uses_qvhighlights_metrics(run: RunInfo, record: dict[str, Any]) -> bool:
    dataset_name = str(record.get("dataset_name") or "").lower()
    return "qvhighlights" in run.suite_name.lower() or dataset_name.startswith("qvhighlights")


def _should_skip_discovery_path(path: Path, eval_root: Path) -> bool:
    try:
        parts = path.relative_to(eval_root).parts
    except ValueError:
        return False
    return any(
        part == "_generated"
        or part == "_summaries"
        or part.startswith("recomputed_metrics")
        for part in parts
    )


def _prediction_paths_for_run(run_dir: Path) -> tuple[Path, ...]:
    # Prefer merged rows when present because merge_eval_shards may have repaired
    # unparsable raw responses into pred_timestamps.
    merged_path = run_dir / "predictions.merged.jsonl"
    if merged_path.exists():
        return (merged_path,)

    rank_paths = tuple(sorted(run_dir.glob("predictions.rank*.jsonl")))
    if rank_paths:
        return rank_paths

    json_path = run_dir / "predictions.json"
    if json_path.exists():
        return (json_path,)

    return ()


def _benchmark_model_from_run_dir(eval_root: Path, run_dir: Path) -> tuple[str, str]:
    try:
        rel_parts = run_dir.relative_to(eval_root).parts
    except ValueError:
        rel_parts = ()
    if len(rel_parts) >= 3:
        return rel_parts[0], rel_parts[1]
    if len(rel_parts) == 2:
        return rel_parts[0], rel_parts[1]
    if len(rel_parts) == 1:
        return rel_parts[0], run_dir.parent.name
    return run_dir.parent.parent.name, run_dir.parent.name


def discover_runs(eval_root: Path) -> list[RunInfo]:
    eval_root = eval_root.resolve()
    run_dirs: set[Path] = set()
    for pattern in (
        "predictions.merged.jsonl",
        "predictions.rank*.jsonl",
        "predictions.json",
    ):
        for prediction_path in eval_root.rglob(pattern):
            if _should_skip_discovery_path(prediction_path, eval_root):
                continue
            run_dirs.add(prediction_path.parent)

    runs: list[RunInfo] = []
    for run_dir in sorted(run_dirs):
        paths = _prediction_paths_for_run(run_dir)
        if not paths:
            continue
        benchmark_name, model_name = _benchmark_model_from_run_dir(eval_root, run_dir)
        record_count = sum(_count_prediction_records(path) for path in paths)
        runs.append(
            RunInfo(
                suite_name=benchmark_name,
                model_name=model_name,
                run_dir=run_dir,
                prediction_paths=tuple(sorted(paths)),
                record_count=record_count,
                timestamp=run_dir.name,
                run_mtime=run_dir.stat().st_mtime,
            )
        )
    return runs


def select_best_runs(runs: Sequence[RunInfo]) -> list[RunInfo]:
    selected: dict[tuple[str, str], RunInfo] = {}
    for run in runs:
        key = (run.suite_name, run.model_name)
        current = selected.get(key)
        if current is None or _selection_key(run) > _selection_key(current):
            selected[key] = run
    return sorted(selected.values(), key=lambda item: (item.suite_name, item.model_name))


def _selection_key(run: RunInfo) -> tuple[int, str, float, str]:
    return (run.record_count, run.timestamp, run.run_mtime, str(run.run_dir))


def recompute_run_metrics(
    run: RunInfo,
    *,
    bucket_edges: Sequence[float] = DEFAULT_BUCKET_EDGES,
    iou_thresholds: Sequence[float] = (0.3, 0.5, 0.7),
) -> dict[str, Any]:
    first_record: dict[str, Any] | None = None
    metrics: Any | None = None
    uses_qvhighlights_metrics = "qvhighlights" in run.suite_name.lower()

    for record in _iter_prediction_records(run.prediction_paths):
        if first_record is None:
            first_record = record
            uses_qvhighlights_metrics = _uses_qvhighlights_metrics(run, record)
            if uses_qvhighlights_metrics:
                metrics = QVHighlightsMetrics()
            else:
                metrics = EvaluationMetrics(
                    iou_thresholds=list(iou_thresholds),
                    gt_duration_bucket_edges=list(bucket_edges),
                )
        gt_timestamps = _normalize_intervals(record.get("gt_timestamps"))
        pred_timestamps = _normalize_intervals(record.get("pred_timestamps"))
        if uses_qvhighlights_metrics:
            metrics.add(pred_timestamps, None, gt_timestamps)
        else:
            metrics.add(pred_timestamps, gt_timestamps)

    if metrics is None:
        metrics = EvaluationMetrics(
            iou_thresholds=list(iou_thresholds),
            gt_duration_bucket_edges=list(bucket_edges),
        )

    summary = metrics.get_summary()
    summary["requested_metrics"] = _extract_requested_metrics(summary)
    summary.update(
        {
            "benchmark_name": run.suite_name,
            "suite_name": run.suite_name,
            "model_name": run.model_name,
            "model_display_name": (first_record or {}).get("model_display_name"),
            "runner_family": (first_record or {}).get("runner_family"),
            "dataset_name": (first_record or {}).get("dataset_name"),
            "run_dir": str(run.run_dir),
            "prediction_files": [str(path) for path in run.prediction_paths],
            "prediction_record_count": run.record_count,
            "selected_run_timestamp": run.timestamp,
            "selection_rule": SELECTION_RULE,
            "gt_duration_bucket_edges": [float(edge) for edge in bucket_edges],
        }
    )
    return summary


def _extract_requested_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    breakdown = summary.get("gt_duration_breakdown", {})
    short = breakdown.get("Short", {}).get("mean_iou") if "Short" in breakdown else None
    if short is None:
        short = summary.get("mIoU_S")
    medium = breakdown.get("Medium", {}).get("mean_iou") if "Medium" in breakdown else None
    if medium is None:
        medium = summary.get("mIoU_M")
    long = breakdown.get("Long", {}).get("mean_iou") if "Long" in breakdown else None
    if long is None:
        long = summary.get("mIoU_L")
    return {
        "mean_iou": summary.get("mean_iou"),
        "mIoU_S": short,
        "mIoU_M": medium,
        "mIoU_L": long,
    }


def _flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    breakdown = summary.get("gt_duration_breakdown", {})
    requested_metrics = summary.get("requested_metrics") or _extract_requested_metrics(summary)
    row = {
        "benchmark_name": summary.get("benchmark_name") or summary.get("suite_name"),
        "model_name": summary.get("model_name"),
        "dataset_name": summary.get("dataset_name"),
        "model_display_name": summary.get("model_display_name"),
        "runner_family": summary.get("runner_family"),
        "run_timestamp": summary.get("selected_run_timestamp"),
        "run_dir": summary.get("run_dir"),
        "prediction_record_count": summary.get("prediction_record_count"),
        "total_samples": summary.get("total_samples"),
        "parsed_samples": summary.get("parsed_samples"),
        "parse_rate": summary.get("parse_rate"),
        "evaluated_samples": summary.get("evaluated_samples"),
        "mean_iou": summary.get("mean_iou"),
        "mIoU_S": requested_metrics.get("mIoU_S"),
        "mIoU_M": requested_metrics.get("mIoU_M"),
        "mIoU_L": requested_metrics.get("mIoU_L"),
        "gt_Short_mean_iou": breakdown.get("Short", {}).get("mean_iou"),
        "gt_Medium_mean_iou": breakdown.get("Medium", {}).get("mean_iou"),
        "gt_Long_mean_iou": breakdown.get("Long", {}).get("mean_iou"),
        "segment_mean_iou": summary.get("segment_mean_iou"),
    }
    for key in (
        "iou@0.3_rate",
        "iou@0.5_rate",
        "iou@0.7_rate",
        "R@1,IoU=0.5",
        "R@1,IoU=0.7",
        "mAP@IoU=0.5",
        "mAP@IoU=0.75",
    ):
        row[key] = summary.get(key)
    return row


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_benchmark_summaries(
    *,
    eval_root: Path,
    selected_summaries: Sequence[dict[str, Any]],
    all_summaries: Sequence[dict[str, Any]],
) -> None:
    selected_by_benchmark: dict[str, list[dict[str, Any]]] = {}
    all_by_benchmark: dict[str, list[dict[str, Any]]] = {}

    for summary in selected_summaries:
        benchmark = str(summary.get("benchmark_name") or summary.get("suite_name"))
        selected_by_benchmark.setdefault(benchmark, []).append(summary)
    for summary in all_summaries:
        benchmark = str(summary.get("benchmark_name") or summary.get("suite_name"))
        all_by_benchmark.setdefault(benchmark, []).append(summary)

    for benchmark, summaries in sorted(selected_by_benchmark.items()):
        benchmark_dir = eval_root.joinpath(*benchmark.split("/"))
        selected_rows = [
            _flatten_summary(summary)
            for summary in sorted(
                summaries,
                key=lambda item: (
                    str(item.get("model_name") or ""),
                    str(item.get("selected_run_timestamp") or ""),
                ),
            )
        ]
        _write_csv(benchmark_dir / BENCHMARK_SUMMARY_FILENAME, selected_rows)

        all_rows = [
            _flatten_summary(summary)
            for summary in sorted(
                all_by_benchmark.get(benchmark, []),
                key=lambda item: (
                    str(item.get("model_name") or ""),
                    str(item.get("selected_run_timestamp") or ""),
                    str(item.get("run_dir") or ""),
                ),
            )
        ]
        _write_csv(benchmark_dir / BENCHMARK_ALL_RUNS_FILENAME, all_rows)


def write_outputs(
    *,
    output_dir: Path,
    eval_root: Path,
    all_runs: Sequence[RunInfo],
    selected_runs: Sequence[RunInfo],
    summaries: Sequence[dict[str, Any]],
    bucket_edges: Sequence[float],
    write_run_metrics: bool,
    write_benchmark_summaries: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    selected_paths = {run.run_dir.resolve() for run in selected_runs}
    selected_summaries = [
        summary
        for summary in summaries
        if Path(str(summary["run_dir"])).resolve() in selected_paths
    ]

    payload = {
        "generated_at_utc": generated_at,
        "eval_root": str(eval_root.resolve()),
        "selection_rule": SELECTION_RULE,
        "gt_duration_bucket_edges": [float(edge) for edge in bucket_edges],
        "total_candidate_runs": len(all_runs),
        "selected_run_count": len(selected_runs),
        "run_metrics": list(summaries),
        "selected_metrics": list(selected_summaries),
    }
    with (output_dir / "recomputed_metrics_3gt_buckets.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)

    _write_csv(
        output_dir / "recomputed_metrics_3gt_buckets.csv",
        [_flatten_summary(summary) for summary in selected_summaries],
    )
    _write_csv(
        output_dir / "recomputed_metrics_3gt_buckets.all_runs.csv",
        [_flatten_summary(summary) for summary in summaries],
    )

    _write_csv(
        output_dir / "selected_runs.csv",
        [
            {
                "benchmark_name": run.suite_name,
                "model_name": run.model_name,
                "run_dir": str(run.run_dir),
                "prediction_record_count": run.record_count,
                "timestamp": run.timestamp,
            }
            for run in selected_runs
        ],
    )
    _write_csv(
        output_dir / "all_run_candidates.csv",
        [
            {
                "benchmark_name": run.suite_name,
                "model_name": run.model_name,
                "run_dir": str(run.run_dir),
                "prediction_record_count": run.record_count,
                "timestamp": run.timestamp,
            }
            for run in all_runs
        ],
    )

    if write_run_metrics:
        for summary in summaries:
            run_metrics_path = Path(str(summary["run_dir"])) / RUN_METRICS_FILENAME
            run_payload = dict(summary)
            run_payload["recomputed_at_utc"] = generated_at
            with run_metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(run_payload, handle, ensure_ascii=False, indent=2, default=_json_default)

    if write_benchmark_summaries:
        _write_benchmark_summaries(
            eval_root=eval_root,
            selected_summaries=selected_summaries,
            all_summaries=summaries,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute mIoU metrics from eval/suite prediction files."
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=PROJECT_ROOT / "eval" / "suite",
        help="Root directory containing eval suite outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for aggregate JSON/CSV outputs. Defaults to eval_root/_summaries/<timestamp>.",
    )
    parser.add_argument(
        "--no-run-metrics",
        action="store_true",
        help=f"Do not write {RUN_METRICS_FILENAME} into run directories.",
    )
    parser.add_argument(
        "--no-benchmark-summaries",
        action="store_true",
        help=(
            f"Do not write {BENCHMARK_SUMMARY_FILENAME} and "
            f"{BENCHMARK_ALL_RUNS_FILENAME} into benchmark directories."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = eval_root / "_summaries" / timestamp
    output_dir = output_dir.resolve()

    all_runs = discover_runs(eval_root)
    selected_runs = select_best_runs(all_runs)
    summaries = [
        recompute_run_metrics(run, bucket_edges=DEFAULT_BUCKET_EDGES)
        for run in all_runs
    ]
    write_outputs(
        output_dir=output_dir,
        eval_root=eval_root,
        all_runs=all_runs,
        selected_runs=selected_runs,
        summaries=summaries,
        bucket_edges=DEFAULT_BUCKET_EDGES,
        write_run_metrics=not args.no_run_metrics,
        write_benchmark_summaries=not args.no_benchmark_summaries,
    )

    print(f"Discovered candidate runs: {len(all_runs)}")
    print(f"Selected runs: {len(selected_runs)}")
    print(f"Wrote aggregate outputs to: {output_dir}")
    if not args.no_run_metrics:
        print(f"Wrote per-run metrics file: {RUN_METRICS_FILENAME}")
    if not args.no_benchmark_summaries:
        print(
            "Wrote per-benchmark CSV files: "
            f"{BENCHMARK_SUMMARY_FILENAME}, {BENCHMARK_ALL_RUNS_FILENAME}"
        )


if __name__ == "__main__":
    main()
