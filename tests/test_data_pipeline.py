from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "data_pipeline/train_building/pipeline_core.py"


def _load_pipeline_core():
    spec = importlib.util.spec_from_file_location("timeple_pipeline_core", CORE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_temporal_iou() -> None:
    core = _load_pipeline_core()
    assert core.temporal_iou([0.0, 10.0], [5.0, 15.0]) == 1.0 / 3.0
    assert core.temporal_iou([0.0, 1.0], [2.0, 3.0]) == 0.0


def test_model_output_parser_accepts_json_fence() -> None:
    core = _load_pipeline_core()
    parsed = core.parse_model_output_with_metadata(
        '```json\n{"event_timeline": [], "query_prediction": {"start_time": 1, "end_time": 2}}\n```'
    )
    assert parsed["intervals"] == [[1.0, 2.0]]
