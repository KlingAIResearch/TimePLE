from __future__ import annotations

from pathlib import Path
from typing import Any

from common import load_yaml_with_base, merge_dict


def load_prompt_profile(path: Path) -> dict[str, Any]:
    payload = load_yaml_with_base(path)
    prompt_cfg = payload.get("prompt")
    if not isinstance(prompt_cfg, dict):
        raise ValueError(f"`prompt` section missing or invalid in prompt profile: {path}")
    return payload


def resolve_prompt_profile(
    prompt_profile: dict[str, Any],
    *,
    profile_path: Path,
    model_cfg: dict[str, Any],
    job_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt_section = prompt_profile["prompt"]
    resolved: dict[str, Any] = {}
    matched_selectors: list[str] = []

    default_prompt = prompt_section.get("default")
    if default_prompt is not None:
        if not isinstance(default_prompt, dict):
            raise ValueError(
                f"`prompt.default` must be a mapping in prompt profile: {profile_path}"
            )
        resolved = merge_dict(resolved, default_prompt)
        matched_selectors.append("default")

    selector_specs = [
        ("by_model_family", str(model_cfg.get("family", "")).strip()),
        ("by_runner_family", str(model_cfg.get("runner", {}).get("family", "")).strip()),
        ("by_model_name", str(model_cfg.get("name", "")).strip()),
        ("by_job_name", str(job_cfg.get("name", "")).strip() if job_cfg else ""),
    ]

    for selector_name, selector_value in selector_specs:
        if not selector_value:
            continue
        selector_table = prompt_section.get(selector_name)
        if selector_table is None:
            continue
        if not isinstance(selector_table, dict):
            raise ValueError(
                f"`prompt.{selector_name}` must be a mapping in prompt profile: {profile_path}"
            )
        override_cfg = selector_table.get(selector_value)
        if override_cfg is None:
            continue
        if not isinstance(override_cfg, dict):
            raise ValueError(
                f"`prompt.{selector_name}.{selector_value}` must be a mapping "
                f"in prompt profile: {profile_path}"
            )
        resolved = merge_dict(resolved, override_cfg)
        matched_selectors.append(f"{selector_name}:{selector_value}")

    model_name = str(model_cfg.get("name", "")).strip()
    model_groups = prompt_section.get("by_model_groups", [])
    if not isinstance(model_groups, list):
        raise ValueError(
            f"`prompt.by_model_groups` must be a list in prompt profile: {profile_path}"
        )
    for index, group in enumerate(model_groups):
        if not isinstance(group, dict):
            raise ValueError(
                f"`prompt.by_model_groups[{index}]` must be a mapping in: {profile_path}"
            )
        models = group.get("models", [])
        override_cfg = group.get("config", {})
        if not isinstance(models, list) or not all(isinstance(item, str) for item in models):
            raise ValueError(
                f"`prompt.by_model_groups[{index}].models` must be a string list in: "
                f"{profile_path}"
            )
        if not isinstance(override_cfg, dict):
            raise ValueError(
                f"`prompt.by_model_groups[{index}].config` must be a mapping in: "
                f"{profile_path}"
            )
        if model_name and model_name in models:
            resolved = merge_dict(resolved, override_cfg)
            matched_selectors.append(f"by_model_groups:{index}:{model_name}")

    validate_prompt_config(
        prompt_cfg=resolved,
        profile_path=profile_path,
        matched_selectors=matched_selectors,
        model_cfg=model_cfg,
        job_cfg=job_cfg,
    )

    meta = dict(resolved.get("meta", {})) if isinstance(resolved.get("meta"), dict) else {}
    profile_meta = prompt_profile.get("profile")
    profile_name = None
    if isinstance(profile_meta, dict):
        profile_name = str(profile_meta.get("name", "")).strip() or None

    meta.update(
        {
            "profile_name": profile_name,
            "profile_path": str(profile_path),
            "matched_selectors": matched_selectors,
        }
    )
    resolved["meta"] = meta
    return resolved


def validate_prompt_config(
    *,
    prompt_cfg: dict[str, Any],
    profile_path: Path | None,
    matched_selectors: list[str] | None,
    model_cfg: dict[str, Any],
    job_cfg: dict[str, Any] | None,
) -> None:
    context = (
        f"profile={profile_path} model={model_cfg.get('name')} "
        f"runner_family={model_cfg.get('runner', {}).get('family')} "
        f"job={job_cfg.get('name') if job_cfg else None} "
        f"matched={matched_selectors or '[]'}"
    )

    user_prompt = prompt_cfg.get("user_prompt")
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError(
            "Unable to resolve `prompt.user_prompt`; prompt profiles must provide an "
            f"explicit prompt entry. Context: {context}"
        )

    if prompt_cfg.get("enabled_sys_prompt", False):
        system_prompt = prompt_cfg.get("system_prompt")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError(
                "Resolved prompt enables `system_prompt` but no non-empty "
                f"`prompt.system_prompt` was provided. Context: {context}"
            )

    if prompt_cfg.get("enabled_format_prompt", False):
        format_prompt = prompt_cfg.get("format_prompt")
        if not isinstance(format_prompt, str) or not format_prompt.strip():
            raise ValueError(
                "Resolved prompt enables `format_prompt` but no non-empty "
                f"`prompt.format_prompt` was provided. Context: {context}"
            )
