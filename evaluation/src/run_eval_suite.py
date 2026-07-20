#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (
    job_output_dir,
    load_yaml_with_base,
    merge_dict,
    normalize_benchmark_name,
    resolve_ref,
    resolve_suite_output_base,
    safe_dump,
    shell_join,
    suite_name_from_config,
    suite_workspace_dir,
)
from prompt_resolver import load_prompt_profile, resolve_prompt_profile, validate_prompt_config


def resolve_dataset_runtime_paths(dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve committed project-relative data paths in generated job configs."""
    resolved = copy.deepcopy(dataset_cfg)
    for key in ("path", "video_base_dir"):
        value = resolved.get(key)
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            resolved[key] = str((PROJECT_ROOT / path).resolve())
    return resolved


def resolve_runner_script(runner_script: str | None) -> Path | None:
    if not runner_script:
        return None
    script_path = Path(runner_script)
    if not script_path.is_absolute():
        script_path = (PROJECT_ROOT / script_path).resolve()
    return script_path


def resolve_python_bin(python_bin: str) -> str:
    candidate = Path(python_bin)
    if candidate.is_absolute():
        return str(candidate)
    if any(sep in python_bin for sep in ("/", "\\")):
        project_candidate = (PROJECT_ROOT / candidate).absolute()
        if project_candidate.exists():
            return str(project_candidate)
    return python_bin


def build_command(
    python_bin: str,
    runner_kind: str,
    runner_script: str | None,
    config_path: Path,
) -> list[str]:
    python_cmd = [resolve_python_bin(python_bin)]
    if runner_kind in {"eval_suite_vllm", "eval_suite_transformers"}:
        script = runner_script or (
            "evaluation/src/run_vllm_benchmark.py"
            if runner_kind == "eval_suite_vllm"
            else "evaluation/src/run_transformers_benchmark.py"
        )
        python_cmd.append("-P")
    elif runner_kind == "builtin_eval":
        script = runner_script or "eval_temporal_grounding.py"
    elif runner_kind == "builtin_vllm_eval":
        script = runner_script or "eval_temporal_grounding_vllm.py"
    elif runner_kind == "external_manifest":
        if not runner_script:
            raise ValueError("external_manifest runner requires `runner.script`.")
        script = runner_script
        python_cmd.append("-P")
    else:
        raise ValueError(f"Unsupported runner kind: {runner_kind}")

    script_path = resolve_runner_script(script)
    if script_path is None:
        raise ValueError(f"Unable to resolve runner script for kind={runner_kind}")

    return [*python_cmd, str(script_path), "--config", str(config_path)]


def filter_suite_jobs(
    jobs: list[dict[str, Any]],
    selected_models: set[str] | None,
) -> list[dict[str, Any]]:
    if not selected_models:
        return jobs

    available_jobs = {
        str(job["name"])
        for job in jobs
        if isinstance(job, dict) and "name" in job
    }
    unknown_jobs = sorted(selected_models - available_jobs)
    if unknown_jobs:
        raise ValueError(
            "Unknown job name(s) requested via --models: "
            f"{unknown_jobs}. Available jobs: {sorted(available_jobs)}"
        )

    return [job for job in jobs if str(job.get("name")) in selected_models]


def build_prompt_config(
    *,
    suite_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    dataset_path: Path,
    model_cfg: dict[str, Any],
    job_cfg: dict[str, Any],
) -> dict[str, Any]:
    prompt_cfg = copy.deepcopy(suite_cfg.get("defaults", {}).get("prompt", {}))
    prompt_profile_path: Path | None = None

    prompt_profile_ref = dataset_cfg["dataset"].get("prompt_profile_ref")
    if prompt_profile_ref:
        prompt_profile_path = resolve_ref(dataset_path.parent, str(prompt_profile_ref))
        prompt_profile = load_prompt_profile(prompt_profile_path)
        resolved_prompt_cfg = resolve_prompt_profile(
            prompt_profile,
            profile_path=prompt_profile_path,
            model_cfg=model_cfg["model"],
            job_cfg=job_cfg,
        )
        prompt_cfg = merge_dict(prompt_cfg, resolved_prompt_cfg)

    validate_prompt_config(
        prompt_cfg=prompt_cfg,
        profile_path=prompt_profile_path,
        matched_selectors=None,
        model_cfg=model_cfg["model"],
        job_cfg=job_cfg,
    )

    return prompt_cfg


def compose_eval_suite_vllm_payload(
    suite_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    dataset_path: Path,
    model_cfg: dict[str, Any],
    job_cfg: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    defaults = suite_cfg.get("defaults", {})
    eval_cfg = model_cfg["model"].get("eval", {})
    payload: dict[str, Any] = {
        "dataset": resolve_dataset_runtime_paths(dataset_cfg["dataset"]),
        "model": copy.deepcopy(model_cfg["model"]),
        "prompt": build_prompt_config(
            suite_cfg=suite_cfg,
            dataset_cfg=dataset_cfg,
            dataset_path=dataset_path,
            model_cfg=model_cfg,
            job_cfg=job_cfg,
        ),
        "metrics": copy.deepcopy(defaults.get("metrics", {})),
        "output": copy.deepcopy(defaults.get("output", {})),
        "distributed": copy.deepcopy(defaults.get("distributed", {})),
        "inference": {},
    }
    payload["output"]["dir"] = str(output_dir)
    payload = merge_dict(payload, eval_cfg.get("config", {}))
    payload = merge_dict(payload, job_cfg.get("overrides", {}))

    payload["suite_meta"] = {
        "suite_name": suite_cfg["suite"]["name"],
        "job_name": job_cfg["name"],
        "model_name": model_cfg["model"]["display_name"],
        "runner_kind": model_cfg["model"]["runner"]["kind"],
        "dataset_profile": dataset_cfg["dataset"]["name"],
        "model_profile": model_cfg["model"]["name"],
    }
    payload["official_reference"] = copy.deepcopy(model_cfg["model"].get("official", {}))
    payload["resource_hints"] = copy.deepcopy(model_cfg["model"].get("resources", {}))
    return payload


def compose_builtin_payload(
    suite_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    dataset_path: Path,
    model_cfg: dict[str, Any],
    job_cfg: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    defaults = suite_cfg.get("defaults", {})
    eval_cfg = model_cfg["model"].get("eval", {})
    payload: dict[str, Any] = {
        "dataset": copy.deepcopy(dataset_cfg["dataset"]),
        "prompt": build_prompt_config(
            suite_cfg=suite_cfg,
            dataset_cfg=dataset_cfg,
            dataset_path=dataset_path,
            model_cfg=model_cfg,
            job_cfg=job_cfg,
        ),
        "metrics": copy.deepcopy(defaults.get("metrics", {})),
        "output": copy.deepcopy(defaults.get("output", {})),
    }
    payload["output"]["dir"] = str(output_dir)
    payload = merge_dict(payload, eval_cfg.get("config", {}))
    payload = merge_dict(payload, job_cfg.get("overrides", {}))

    payload["suite_meta"] = {
        "suite_name": suite_cfg["suite"]["name"],
        "job_name": job_cfg["name"],
        "model_name": model_cfg["model"]["display_name"],
        "runner_kind": model_cfg["model"]["runner"]["kind"],
        "dataset_profile": dataset_cfg["dataset"]["name"],
        "model_profile": model_cfg["model"]["name"],
    }
    payload["official_reference"] = copy.deepcopy(model_cfg["model"].get("official", {}))
    payload["resource_hints"] = copy.deepcopy(model_cfg["model"].get("resources", {}))
    return payload


def compose_external_manifest(
    suite_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    dataset_path: Path,
    model_cfg: dict[str, Any],
    job_cfg: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    defaults = suite_cfg.get("defaults", {})
    eval_cfg = model_cfg["model"].get("eval", {})
    payload: dict[str, Any] = {
        "dataset": copy.deepcopy(dataset_cfg["dataset"]),
        "prompt": build_prompt_config(
            suite_cfg=suite_cfg,
            dataset_cfg=dataset_cfg,
            dataset_path=dataset_path,
            model_cfg=model_cfg,
            job_cfg=job_cfg,
        ),
        "metrics": copy.deepcopy(defaults.get("metrics", {})),
        "output": copy.deepcopy(defaults.get("output", {})),
        "backend": {},
        "parsing": {},
    }
    payload["output"]["dir"] = str(output_dir)
    payload = merge_dict(payload, eval_cfg.get("config", {}))
    payload = merge_dict(payload, job_cfg.get("overrides", {}))

    payload["suite_meta"] = {
        "suite_name": suite_cfg["suite"]["name"],
        "job_name": job_cfg["name"],
        "model_name": model_cfg["model"]["display_name"],
        "runner_kind": model_cfg["model"]["runner"]["kind"],
        "dataset_profile": dataset_cfg["dataset"]["name"],
        "model_profile": model_cfg["model"]["name"],
        "status": "manifest_only",
    }
    payload["runner"] = copy.deepcopy(model_cfg["model"]["runner"])
    payload["model"] = {
        key: copy.deepcopy(value)
        for key, value in model_cfg["model"].items()
        if key not in {"runner", "eval"}
    }
    payload["resource_hints"] = copy.deepcopy(model_cfg["model"].get("resources", {}))
    return payload


def render_suite(
    suite_path: Path,
    output_root_override: str | None = None,
    *,
    selected_models: set[str] | None = None,
    python_bin_override: str | None = None,
) -> dict[str, Any]:
    suite_cfg = load_yaml_with_base(suite_path)
    if "suite" not in suite_cfg:
        raise ValueError(f"`suite` section missing in {suite_path}.")
    if "dataset_ref" not in suite_cfg:
        raise ValueError(f"`dataset_ref` missing in {suite_path}.")
    if "jobs" not in suite_cfg or not isinstance(suite_cfg["jobs"], list):
        raise ValueError(f"`jobs` must be a list in {suite_path}.")

    suite_dir = suite_path.parent
    dataset_path = resolve_ref(suite_dir, str(suite_cfg["dataset_ref"]))
    dataset_cfg = load_yaml_with_base(dataset_path)

    output_base_path = resolve_suite_output_base(
        PROJECT_ROOT,
        suite_cfg,
        output_root_override,
    )
    suite_name = suite_name_from_config(suite_cfg, suite_path)
    benchmark_name = normalize_benchmark_name(dataset_cfg["dataset"]["name"])
    benchmark_root_path = output_base_path / benchmark_name
    workspace_dir = suite_workspace_dir(output_base_path, suite_name)
    generated_dir = workspace_dir / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)

    execution_cfg = suite_cfg.get("defaults", {}).get("execution", {})
    default_python_bin = str(execution_cfg.get("python", "python"))
    if python_bin_override:
        default_python_bin = python_bin_override
    jobs = filter_suite_jobs(list(suite_cfg["jobs"]), selected_models)

    commands: list[dict[str, Any]] = []
    rendered_configs: list[Path] = []

    for job in jobs:
        if "name" not in job or "model_ref" not in job:
            raise ValueError("Each job requires `name` and `model_ref`.")

        model_path = resolve_ref(suite_dir, str(job["model_ref"]))
        model_cfg = load_yaml_with_base(model_path)
        runner = model_cfg["model"]["runner"]
        runner_kind = runner["kind"]
        python_bin = str(runner.get("python", default_python_bin))

        output_dir = job_output_dir(output_base_path, benchmark_name, str(job["name"]))
        config_path = generated_dir / f"{job['name']}.yaml"

        if runner_kind in {"eval_suite_vllm", "eval_suite_transformers"}:
            payload = compose_eval_suite_vllm_payload(
                suite_cfg=suite_cfg,
                dataset_cfg=dataset_cfg,
                dataset_path=dataset_path,
                model_cfg=model_cfg,
                job_cfg=job,
                output_dir=output_dir,
            )
        elif runner_kind in {"builtin_eval", "builtin_vllm_eval"}:
            payload = compose_builtin_payload(
                suite_cfg=suite_cfg,
                dataset_cfg=dataset_cfg,
                dataset_path=dataset_path,
                model_cfg=model_cfg,
                job_cfg=job,
                output_dir=output_dir,
            )
        elif runner_kind == "external_manifest":
            payload = compose_external_manifest(
                suite_cfg=suite_cfg,
                dataset_cfg=dataset_cfg,
                dataset_path=dataset_path,
                model_cfg=model_cfg,
                job_cfg=job,
                output_dir=output_dir,
            )
        else:
            raise ValueError(f"Unsupported runner kind: {runner_kind}")

        payload.setdefault("suite_meta", {})
        payload["suite_meta"].update(
            {
                "benchmark_name": benchmark_name,
                "output_base": str(output_base_path),
                "benchmark_output_root": str(benchmark_root_path),
                "job_output_root": str(output_dir),
            }
        )
        safe_dump(config_path, payload)
        rendered_configs.append(config_path)

        command = build_command(
            python_bin=python_bin,
            runner_kind=runner_kind,
            runner_script=runner.get("script"),
            config_path=config_path,
        )
        runner_script_path = resolve_runner_script(runner.get("script"))
        is_ready = True
        if runner_kind == "external_manifest":
            is_ready = runner_script_path is not None and runner_script_path.exists()
        commands.append(
            {
                "job_name": job["name"],
                "runner_kind": runner_kind,
                "command": command,
                "manifest": str(config_path),
                "ready": is_ready,
            }
        )

    run_script = workspace_dir / "run_all.sh"
    run_script.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for item in commands:
        lines.append(f"# {item['job_name']} [{item['runner_kind']}]")
        if item["ready"]:
            lines.append(shell_join(item["command"]))
        else:
            lines.append("# pending external runner implementation")
            lines.append(f"# {shell_join(item['command'])}")
        lines.append("")
    run_script.write_text("\n".join(lines), encoding="utf-8")
    os.chmod(run_script, 0o755)

    manifest_path = workspace_dir / "suite_manifest.yaml"
    safe_dump(
        manifest_path,
        {
            "suite": copy.deepcopy(suite_cfg["suite"]),
            "dataset_ref": str(dataset_path),
            "output_base": str(output_base_path),
            "benchmark_name": benchmark_name,
            "benchmark_output_root": str(benchmark_root_path),
            "suite_workspace": str(workspace_dir),
            "jobs": commands,
            "generated_configs": [str(path) for path in rendered_configs],
            "run_script": str(run_script),
            "selected_models": sorted(selected_models) if selected_models else None,
        },
    )

    return {
        "output_root": workspace_dir,
        "output_base": output_base_path,
        "benchmark_name": benchmark_name,
        "benchmark_output_root": benchmark_root_path,
        "suite_workspace": workspace_dir,
        "generated_configs": rendered_configs,
        "run_script": run_script,
        "commands": commands,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render or execute a multi-model evaluation suite.")
    parser.add_argument(
        "--suite",
        required=True,
        help="Path to the suite YAML.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output root override, useful for dry runs or temporary rendering.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of suite job names to render or execute.",
    )
    parser.add_argument(
        "--python-bin",
        default=None,
        help="Python interpreter to use for rendered worker commands.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute rendered commands sequentially after rendering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_path = Path(args.suite).resolve()
    selected_models = set(args.models) if args.models else None
    rendered = render_suite(
        suite_path,
        output_root_override=args.output_root,
        selected_models=selected_models,
        python_bin_override=args.python_bin,
    )

    print(f"Rendered suite workspace: {rendered['suite_workspace']}")
    print(f"Output base: {rendered['output_base']}")
    print(f"Benchmark output root: {rendered['benchmark_output_root']}")
    print(f"Run script: {rendered['run_script']}")
    print("Generated configs:")
    for config_path in rendered["generated_configs"]:
        print(f"  - {config_path}")

    if not args.execute:
        return

    for item in rendered["commands"]:
        if not item["ready"]:
            raise FileNotFoundError(
                f"Runner not implemented for {item['job_name']}: {item['command'][1]}"
            )
        print(f"Executing {item['job_name']}: {shell_join(item['command'])}")
        subprocess.run(item["command"], check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
