from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_evaluation_metrics_class(project_root: Path):
    del project_root
    metrics_path = Path(__file__).resolve().with_name("metrics.py")
    module = _load_module("eval_suite_metrics_module", metrics_path)
    return module.EvaluationMetrics


def load_qvhighlights_metrics_class(project_root: Path):
    del project_root
    metrics_path = Path(__file__).resolve().with_name("qvhighlights_metrics.py")
    module = _load_module("eval_suite_qvhighlights_metrics_module", metrics_path)
    return module.QVHighlightsMetrics
