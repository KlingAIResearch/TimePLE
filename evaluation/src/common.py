from __future__ import annotations

import copy
import shlex
from pathlib import Path
from typing import Any

import yaml

DEFAULT_EVAL_SUITE_OUTPUT_ROOT = "evaluation/outputs"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object at {path}, got {type(payload).__name__}.")
    return payload


def merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml_with_base(path: Path) -> dict[str, Any]:
    payload = load_yaml(path)
    base_ref = payload.pop("__base__", None)
    if not base_ref:
        return payload

    base_path = Path(base_ref)
    if not base_path.is_absolute():
        base_path = (path.parent / base_path).resolve()
    base_payload = load_yaml_with_base(base_path)
    return merge_dict(base_payload, payload)


def resolve_ref(base_dir: Path, ref: str) -> Path:
    ref_path = Path(ref)
    if ref_path.is_absolute():
        return ref_path
    return (base_dir / ref_path).resolve()


def safe_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def get_nested_value(data: dict[str, Any], field: str) -> Any:
    value: Any = data
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(field)
        value = value[part]
    return value


def normalize_benchmark_name(name: str) -> str:
    value = str(name or "").strip()
    if value == "charades_sta_manual_review_20260224":
        return "charades_sta_manual_review"
    return value or "unknown_benchmark"


def suite_name_from_config(suite_cfg: dict[str, Any], suite_path: Path) -> str:
    suite_section = suite_cfg.get("suite", {})
    if isinstance(suite_section, dict):
        value = str(suite_section.get("name") or "").strip()
        if value:
            return value
    return suite_path.stem


def resolve_suite_output_base(
    project_root: Path,
    suite_cfg: dict[str, Any],
    output_root_override: str | None = None,
) -> Path:
    suite_section = suite_cfg.get("suite", {})
    output_root = output_root_override
    if output_root is None and isinstance(suite_section, dict):
        output_root = suite_section.get("output_root")
    if not output_root:
        output_root = DEFAULT_EVAL_SUITE_OUTPUT_ROOT

    output_root_path = Path(str(output_root))
    if output_root_path.is_absolute():
        return output_root_path.resolve()
    return (project_root / output_root_path).resolve()


def suite_workspace_dir(output_base: Path, suite_name: str) -> Path:
    return output_base / "_generated" / suite_name


def benchmark_output_dir(output_base: Path, benchmark_name: str) -> Path:
    return output_base / normalize_benchmark_name(benchmark_name)


def job_output_dir(output_base: Path, benchmark_name: str, model_name: str) -> Path:
    return benchmark_output_dir(output_base, benchmark_name) / str(model_name)
