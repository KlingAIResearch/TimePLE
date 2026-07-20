from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_SRC = ROOT / "evaluation" / "src"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_evaluation_metrics_include_duration_breakdown() -> None:
    metrics_module = _load_module("timeple_eval_metrics_test", EVAL_SRC / "metrics.py")
    metrics = metrics_module.EvaluationMetrics(iou_thresholds=[0.3, 0.5, 0.7])
    metrics.add([(1.0, 5.0)], [(2.0, 6.0)])

    summary = metrics.get_summary()
    assert summary["mean_iou"] == 0.6
    assert summary["mIoU_S"] == 0.6
    assert summary["mIoU_M"] is None


def test_public_suite_renders_project_data_paths(tmp_path: Path) -> None:
    sys.path.insert(0, str(EVAL_SRC))
    try:
        run_eval_suite = _load_module(
            "timeple_run_eval_suite_test", EVAL_SRC / "run_eval_suite.py"
        )
        rendered = run_eval_suite.render_suite(
            ROOT / "evaluation/configs/suites/charades_sta.yaml",
            output_root_override=str(tmp_path),
            selected_models={"timeple_8b"},
        )
    finally:
        sys.path.remove(str(EVAL_SRC))

    generated = rendered["generated_configs"][0]
    payload = run_eval_suite.load_yaml_with_base(generated)
    assert payload["dataset"]["path"] == str(
        (ROOT / "data/benchmarks/charades_sta/charades_sta_test.json").resolve()
    )
    assert payload["model"]["runner"]["family"] == "qwen3_vl_timeple"
    assert payload["inference"]["video"]["message_kwargs"]["fps"] == 2.0
    assert payload["inference"]["video"]["message_kwargs"]["max_frames"] == 200
