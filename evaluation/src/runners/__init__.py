from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from runners.base import BaseRunner


RUNNER_IMPORTS = {
    (None, "qwen3_vl"): ("runners.qwen_family", "Qwen3VLRunner"),
    (None, "qwen3_vl_timeple"): ("runners.qwen_family", "QwenTimePLERunner"),
}


def _load_runner_cls(family: str, *, kind: str | None):
    target = RUNNER_IMPORTS.get((kind, family)) or RUNNER_IMPORTS.get((None, family))
    if target is None:
        raise ValueError(f"Unsupported public runner family={family!r} kind={kind!r}")
    module_name, class_name = target
    return getattr(import_module(module_name), class_name)


def build_runner(config: dict[str, Any], *, config_path: Path) -> BaseRunner:
    runner_cfg = dict(config.get("model", {}).get("runner", {}))
    kind = runner_cfg.get("kind")
    family = runner_cfg.get("family")
    if not family:
        raise ValueError("Missing `model.runner.family` in job config.")
    runner_cls = _load_runner_cls(str(family), kind=str(kind) if kind else None)
    return runner_cls(config, config_path=config_path)
