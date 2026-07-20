#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


def _sanitize_import_path() -> None:
    """Prefer environment packages over repo-local top-level modules.

    The repository root contains a local `transformers/` package snapshot.
    When this script is launched from the repo root, Python can resolve that
    package ahead of the eval environment's site-packages, which breaks models
    that require newer upstream Transformers support such as Gemma 4.

    We only need `evaluation/src` on `sys.path` for local imports, so it is
    safe to remove the current working directory and project root here.
    """

    os.environ.setdefault("PYTHONSAFEPATH", "1")

    cwd = Path.cwd().resolve()
    blocked_paths = {str(cwd), str(PROJECT_ROOT)}
    sanitized_path: list[str] = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            resolved_entry = str(Path(entry).resolve())
        except OSError:
            resolved_entry = entry
        if resolved_entry in blocked_paths:
            continue
        sanitized_path.append(entry)

    sys.path[:] = sanitized_path
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))


_sanitize_import_path()

from benchmark_loader import BenchmarkSample, load_benchmark_samples
from common import load_yaml_with_base, safe_dump
from distributed import DistributedContext, load_distributed_context
from metrics_loader import load_evaluation_metrics_class, load_qvhighlights_metrics_class
from parsing import parse_timestamp_response
from runners import build_runner


LOGGER = logging.getLogger("eval_suite")
EvaluationMetrics = load_evaluation_metrics_class(PROJECT_ROOT)
QVHighlightsMetrics = load_qvhighlights_metrics_class(PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run modular vLLM benchmark evaluation.")
    parser.add_argument("--config", required=True, help="Path to rendered job config YAML.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap after sharding, useful for smoke tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and benchmark, then exit before model initialization.",
    )
    return parser.parse_args()


def setup_logging(log_level: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )


def _intervals_to_jsonable(
    value: list[tuple[float, float]] | None,
) -> list[list[float]] | None:
    if value is None:
        return None
    return [[float(start), float(end)] for start, end in value]


def _load_processed_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    processed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            sample_id = payload.get("sample_id")
            if sample_id is not None:
                processed.add(str(sample_id))
    return processed


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _build_output_dir(cfg: dict[str, Any]) -> Path:
    resume_from = cfg.get("output", {}).get("resume_from")
    if resume_from:
        return Path(str(resume_from)).resolve()

    root = Path(str(cfg["output"]["dir"])).resolve()
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return root / timestamp


def _build_record(
    sample: BenchmarkSample,
    *,
    cfg: dict[str, Any],
    context: DistributedContext,
    response_text: str,
    response_error: str,
    response_meta: dict[str, Any],
    pred_timestamps: list[tuple[float, float]] | None,
    segment_ious: list[float] | None,
) -> dict[str, Any]:
    return {
        "global_index": sample.global_index,
        "sample_id": sample.sample_id,
        "query": sample.query,
        "video_path": sample.video_path,
        "gt_timestamps": _intervals_to_jsonable(sample.gt_timestamps),
        "pred_timestamps": _intervals_to_jsonable(pred_timestamps),
        "segment_ious": segment_ious,
        "response_text": response_text,
        "error": response_error,
        "response_meta": response_meta,
        "dataset_name": cfg["dataset"]["name"],
        "model_name": cfg["model"]["name"],
        "model_display_name": cfg["model"].get("display_name"),
        "runner_family": cfg["model"].get("runner", {}).get("family"),
        "rank": context.rank,
    }


def _save_metrics(
    metrics: Any,
    *,
    cfg: dict[str, Any],
    context: DistributedContext,
    output_dir: Path,
    processed_count: int,
    assigned_samples: int,
) -> None:
    summary = metrics.get_summary()
    summary["processed_samples"] = processed_count
    summary["assigned_samples"] = assigned_samples
    summary["rank"] = context.rank
    summary["world_size"] = context.world_size
    summary["model_name"] = cfg["model"]["name"]
    summary["dataset_name"] = cfg["dataset"]["name"]
    summary["output_dir"] = str(output_dir)

    with context.metrics_path(output_dir).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


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


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_yaml_with_base(config_path)

    tensor_parallel_size = int(
        cfg.get("inference", {}).get("engine", {}).get("tensor_parallel_size", 1)
    )
    distributed_cfg = dict(cfg.get("distributed", {}))
    context = load_distributed_context(
        distributed_cfg,
        tensor_parallel_size=tensor_parallel_size,
    )

    output_dir = _build_output_dir(cfg)
    setup_logging(str(cfg.get("output", {}).get("log_level", "INFO")), output_dir)

    LOGGER.info("Config: %s", config_path)
    LOGGER.info(
        "Distributed context: enabled=%s world_size=%s rank=%s local_rank=%s local_world_size=%s cuda_visible_devices=%s",
        context.enabled,
        context.world_size,
        context.rank,
        context.local_rank,
        context.local_world_size,
        context.assigned_cuda_visible_devices or os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )

    samples = load_benchmark_samples(cfg["dataset"], config_base_dir=config_path.parent)
    assigned_samples = [sample for sample in samples if context.owns_index(sample.global_index)]
    if args.max_samples is not None:
        assigned_samples = assigned_samples[: max(args.max_samples, 0)]

    LOGGER.info(
        "Loaded %s total samples, %s assigned to rank %s",
        len(samples),
        len(assigned_samples),
        context.rank,
    )

    manifest_path = output_dir / "job_manifest.yaml"
    if context.is_rank0 and not manifest_path.exists():
        safe_dump(manifest_path, cfg)

    runner = build_runner(cfg, config_path=config_path)
    LOGGER.info("Resolved runner family: %s", cfg["model"].get("runner", {}).get("family"))

    if args.dry_run:
        LOGGER.info("Dry run complete. No model was initialized.")
        return

    predictions_path = context.predictions_path(output_dir)
    failed_path = context.failed_path(output_dir)
    resume_enabled = bool(cfg.get("output", {}).get("resume_from"))
    processed_ids = _load_processed_sample_ids(predictions_path) if resume_enabled else set()

    pending_samples = [sample for sample in assigned_samples if sample.sample_id not in processed_ids]
    LOGGER.info(
        "Resume enabled=%s, processed_samples=%s, pending_samples=%s",
        resume_enabled,
        len(processed_ids),
        len(pending_samples),
    )

    metrics = _build_metrics(cfg)

    batch_size = max(1, int(cfg.get("inference", {}).get("batching", {}).get("batch_size", 1)))
    save_every = max(1, int(cfg.get("output", {}).get("save_every", 20)))
    save_predictions = bool(cfg.get("output", {}).get("save_predictions", True))
    save_failed_samples = bool(cfg.get("output", {}).get("save_failed_samples", True))

    processed_count = len(processed_ids)
    for batch_start in range(0, len(pending_samples), batch_size):
        batch = pending_samples[batch_start : batch_start + batch_size]
        responses = runner.predict_batch(batch)

        for sample, response in zip(batch, responses):
            time_codec_predictions = response.metadata.get("time_codec_predictions")
            cis_codec_predictions = response.metadata.get("cis_codec_predictions")
            timeed_predictions = response.metadata.get("timeed_predictions")
            generic_codec_predictions = response.metadata.get("codec_predictions")
            if response.error:
                pred_timestamps = None
            elif (
                time_codec_predictions
                or cis_codec_predictions
                or timeed_predictions
                or generic_codec_predictions
            ):
                codec_predictions = (
                    time_codec_predictions
                    or cis_codec_predictions
                    or timeed_predictions
                    or generic_codec_predictions
                )
                pred_timestamps = [
                    (float(start), float(end))
                    for start, end in codec_predictions
                ]
            else:
                pred_timestamps = parse_timestamp_response(response.raw_text)
            if isinstance(metrics, QVHighlightsMetrics):
                metrics.add(pred_timestamps, None, sample.gt_timestamps)
                segment_ious = None
            else:
                segment_ious = metrics.add(pred_timestamps, sample.gt_timestamps)
            record = _build_record(
                sample,
                cfg=cfg,
                context=context,
                response_text=response.raw_text,
                response_error=response.error,
                response_meta=response.metadata,
                pred_timestamps=pred_timestamps,
                segment_ious=segment_ious,
            )

            if save_predictions:
                _append_jsonl(predictions_path, record)
            if save_failed_samples and (response.error or pred_timestamps is None):
                _append_jsonl(failed_path, record)

            processed_count += 1

        if processed_count % save_every == 0 or processed_count == len(assigned_samples):
            _save_metrics(
                metrics,
                cfg=cfg,
                context=context,
                output_dir=output_dir,
                processed_count=processed_count,
                assigned_samples=len(assigned_samples),
            )
            LOGGER.info(
                "Rank %s progress: %s / %s",
                context.rank,
                processed_count,
                len(assigned_samples),
            )

    _save_metrics(
        metrics,
        cfg=cfg,
        context=context,
        output_dir=output_dir,
        processed_count=processed_count,
        assigned_samples=len(assigned_samples),
    )
    LOGGER.info("Finished evaluation for rank %s. Output dir: %s", context.rank, output_dir)

    llm_init_error = getattr(runner, "_llm_init_error", None)
    if llm_init_error:
        LOGGER.error(
            "Fatal model initialization error encountered for rank %s: %s",
            context.rank,
            llm_init_error,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
