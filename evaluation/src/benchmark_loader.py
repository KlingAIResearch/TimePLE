from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common import get_nested_value, resolve_ref


@dataclass(slots=True)
class BenchmarkSample:
    global_index: int
    sample_id: str
    query: str
    video_path: str
    gt_timestamps: list[tuple[float, float]]
    raw_sample: dict[str, Any] = field(default_factory=dict)


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"Expected JSON object at {path}:{line_no}, got {type(payload).__name__}."
                    )
                records.append(payload)
        return records

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            items = payload.get("data")
            if not isinstance(items, list):
                raise ValueError(
                    f"Expected top-level list or `data` list in {path}, got {type(payload).__name__}."
                )
            records = items
        else:
            raise ValueError(
                f"Expected top-level JSON list or dict in {path}, got {type(payload).__name__}."
            )
        for idx, item in enumerate(records):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Expected JSON object at index {idx} in {path}, got {type(item).__name__}."
                )
        return records

    raise ValueError(f"Unsupported benchmark file suffix for {path}. Expected .json or .jsonl.")


def _coerce_gt_timestamps(value: Any) -> list[tuple[float, float]]:
    if value is None:
        return []

    if isinstance(value, (list, tuple)) and len(value) == 2 and all(
        isinstance(item, (int, float)) for item in value
    ):
        start, end = float(value[0]), float(value[1])
        return [(start, end)]

    if isinstance(value, (list, tuple)):
        intervals: list[tuple[float, float]] = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"Invalid interval item: {item!r}")
            start, end = float(item[0]), float(item[1])
            intervals.append((start, end))
        return intervals

    raise ValueError(f"Unsupported GT timestamp value: {value!r}")


def _resolve_benchmark_path(dataset_cfg: dict[str, Any], config_base_dir: Path) -> Path:
    dataset_path = Path(str(dataset_cfg["path"]))
    if dataset_path.is_absolute():
        resolved = dataset_path
    else:
        resolved = resolve_ref(config_base_dir, str(dataset_path))

    annotation_file = dataset_cfg.get("annotation_file")
    if annotation_file:
        resolved = resolved / str(annotation_file)
    return resolved


def _resolve_video_path(video_value: str, video_base_dir: str | None) -> str:
    video_path = Path(video_value)
    if video_path.is_absolute():
        return str(video_path)
    if video_base_dir:
        base_path = Path(video_base_dir)
        return str((base_path / video_path).resolve())
    return str(video_path)


def load_benchmark_samples(
    dataset_cfg: dict[str, Any],
    *,
    config_base_dir: Path,
) -> list[BenchmarkSample]:
    sample_schema = dataset_cfg.get("sample_schema") or {}
    sample_id_field = str(sample_schema.get("sample_id_field", "sample_id"))
    query_field = str(sample_schema.get("query_field", "query"))
    gt_field = sample_schema.get("gt_field")
    gt_start_field = sample_schema.get("gt_start_field")
    gt_end_field = sample_schema.get("gt_end_field")
    video_field = str(sample_schema.get("video_field", "video_path"))

    dataset_file = _resolve_benchmark_path(dataset_cfg, config_base_dir)
    records = _read_records(dataset_file)

    start_idx = int(dataset_cfg.get("start_idx", 0) or 0)
    end_idx = dataset_cfg.get("end_idx")
    if end_idx is not None:
        end_idx = int(end_idx)
        records = records[start_idx:end_idx]
    else:
        records = records[start_idx:]

    video_base_dir = dataset_cfg.get("video_base_dir")
    samples: list[BenchmarkSample] = []
    for local_index, record in enumerate(records):
        global_index = start_idx + local_index
        try:
            sample_id_value = get_nested_value(record, sample_id_field)
        except KeyError:
            sample_id_value = global_index

        query_value = get_nested_value(record, query_field)
        if gt_field is not None:
            gt_value = get_nested_value(record, str(gt_field))
            gt_timestamps = _coerce_gt_timestamps(gt_value)
        elif gt_start_field is not None and gt_end_field is not None:
            start_value = get_nested_value(record, str(gt_start_field))
            end_value = get_nested_value(record, str(gt_end_field))
            gt_timestamps = [(float(start_value), float(end_value))]
        else:
            raise ValueError(
                "sample_schema requires either `gt_field` or both "
                "`gt_start_field` and `gt_end_field`."
            )
        video_value = get_nested_value(record, video_field)

        sample = BenchmarkSample(
            global_index=global_index,
            sample_id=str(sample_id_value),
            query=str(query_value),
            video_path=_resolve_video_path(str(video_value), video_base_dir),
            gt_timestamps=gt_timestamps,
            raw_sample=record,
        )
        samples.append(sample)

    return samples
