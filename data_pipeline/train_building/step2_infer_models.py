#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
import glob
import heapq
from pathlib import Path
from typing import Any
import yaml

from pipeline_core import (
    GROUNDING_PROMPT_TEMPLATE,
    build_grounding_prompt,
    count_jsonl_lines,
    parse_model_output_with_metadata,
    read_jsonl,
    temporal_iou,
)


LOGGER = logging.getLogger("step2_infer_models")


# Make the self-contained inference backends importable when this file is run
# directly rather than as a package module.
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


@dataclass
class RuntimeConfig:
    input_jsonl: str
    output_jsonl: str
    output_format: str
    models: str
    resume: bool
    overwrite_existing: bool
    skip_missing_video: bool
    max_samples: int
    sleep_seconds: float
    progress_every: int
    flush_every: int


@dataclass
class DistributedConfig:
    enabled: bool
    world_size: int
    rank: int
    local_rank: int
    shard_output: bool
    add_global_index_field: bool
    global_index_field: str
    hostfile: str
    mpi_hostfile: str


@dataclass
class PromptConfig:
    enabled: bool
    system_prompt: str
    query_prompt_template: str


GEMINI_FIELDS = (
    "gemini3pro_tgt",
    "gemini3pro_pred_intervals",
    "gemini3pro_raw_output",
    "gemini3pro_error",
    "gemini3pro_event_timeline",
    "gemini3pro_refined_query",
    "gemini3pro_reason",
    "iou_gemini3pro",
)
QWEN_FIELDS = (
    "qwen3vl_30b_tgt",
    "qwen3vl_30b_pred_intervals",
    "qwen3vl_30b_raw_output",
    "qwen3vl_30b_error",
    "qwen3vl_30b_event_timeline",
    "qwen3vl_30b_refined_query",
    "qwen3vl_30b_reason",
    "iou_qwen3vl_30b",
)


class GeminiKSInferencer:
    """
    使用 inference/gemini_backend.py 进行 Gemini 推理
    """

    def __init__(self, gemini_cfg: dict[str, Any], prompt_cfg: PromptConfig) -> None:
        from inference.gemini_backend import GeminiKSWrapper

        api_cfg = dict(gemini_cfg.get("api", {}))
        gen_cfg = dict(gemini_cfg.get("generation", {}))

        project_id = str(api_cfg.get("project_id", "")).strip()
        if not project_id:
            raise ValueError("Gemini config missing `gemini.api.project_id`.")

        self.system_prompt = prompt_cfg.system_prompt
        self.query_prompt_template = prompt_cfg.query_prompt_template

        self.max_new_tokens = int(gen_cfg.get("max_new_tokens", gen_cfg.get("max_output_tokens", 4096)))
        self.temperature = float(gen_cfg.get("temperature", 0.0))
        self.top_p = float(gen_cfg.get("top_p", 1.0))
        self.max_workers = max(1, int(api_cfg.get("max_workers", 1)))
        timeout_seconds = float(api_cfg.get("timeout", 300.0))
        if timeout_seconds <= 0:
            timeout_seconds = 300.0
        timeout_ms = int(timeout_seconds * 1000)
        self.timeout_seconds = timeout_seconds

        self._api_config = {
            "project_id": project_id,
            "location": api_cfg.get("location", "global"),
            "model_name": api_cfg.get("model_name", "gemini-3-pro-preview"),
            "credentials_path": api_cfg.get("credentials_path", ""),
            "temperature": self.temperature,
            "max_output_tokens": int(gen_cfg.get("max_output_tokens", self.max_new_tokens)),
            "top_p": self.top_p,
            "top_k": int(gen_cfg.get("top_k", 1)),
            "media_resolution": api_cfg.get("media_resolution", "media_resolution_high"),
            "video_fps": float(api_cfg.get("video_fps", 1.0)),
            "timeout": timeout_ms,
            "max_retries": int(api_cfg.get("max_retries", 3)),
            "retry_delay": float(api_cfg.get("retry_delay", 5.0)),
        }

        self._wrapper_cls = GeminiKSWrapper
        self._thread_local = threading.local()

        self._max_retries = int(self._api_config["max_retries"])
        self._retry_delay = float(self._api_config["retry_delay"])
        self._bootstrap_wrapper()
        LOGGER.info(
            "GeminiKSInferencer initialized: model=%s project=%s location=%s max_workers=%s timeout_seconds=%s",
            self._api_config["model_name"],
            self._api_config["project_id"],
            self._api_config["location"],
            self.max_workers,
            self.timeout_seconds,
        )

    def _bootstrap_wrapper(self) -> None:
        wrapper = self._wrapper_cls(
            api_config=self._api_config,
            timeout=int(self._api_config["timeout"]),
            retry=self._max_retries,
            wait=self._retry_delay,
        )
        self._thread_local.wrapper = wrapper

    def _get_wrapper(self) -> Any:
        wrapper = getattr(self._thread_local, "wrapper", None)
        if wrapper is None:
            wrapper = self._wrapper_cls(
                api_config=self._api_config,
                timeout=int(self._api_config["timeout"]),
                retry=self._max_retries,
                wait=self._retry_delay,
            )
            self._thread_local.wrapper = wrapper
        return wrapper

    def predict(self, video_path: str, query: str, *, use_grounding_template: bool = True) -> dict[str, Any]:
        prompt = build_grounding_prompt(query, prompt_template=self.query_prompt_template) if use_grounding_template else query
        wrapper = self._get_wrapper()
        try:
            output_text = wrapper.generate(
                text=prompt,
                video_path=video_path,
                system_prompt=self.system_prompt or None,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            parsed = parse_model_output_with_metadata(output_text)
            return {
                "raw_text": output_text,
                "intervals": parsed["intervals"],
                "response_text": parsed["raw_text"],
                "reason": parsed["reason"],
                "event_timeline": parsed["event_timeline"],
                "refined_query": parsed["refined_query"],
                "error": "",
            }
        except Exception as exc:
            return {
                "raw_text": "",
                "intervals": [],
                "response_text": "",
                "reason": "",
                "event_timeline": [],
                "refined_query": "",
                "error": str(exc),
            }


class QwenVLLMInferencer:
    """
    参考 inference/vllm_backend.py，使用本地 VLLMInferenceEngine 推理 Qwen3VL-30B
    """

    def __init__(self, qwen_cfg: dict[str, Any], prompt_cfg: PromptConfig) -> None:
        self.system_prompt = prompt_cfg.system_prompt
        self.query_prompt_template = prompt_cfg.query_prompt_template

        model_path = str(qwen_cfg.get("model_path", "")).strip()
        if not model_path:
            raise ValueError("Qwen vLLM config missing `qwen_vllm.model_path`.")

        self.video_cfg = dict(qwen_cfg.get("video", {}))
        for key in ("frame_min_token", "frame_max_token", "frame_token_only"):
            if key not in self.video_cfg and qwen_cfg.get(key) is not None:
                self.video_cfg[key] = qwen_cfg.get(key)
        self.gen_cfg = dict(qwen_cfg.get("generation", {}))

        self._prepare_vllm_env()
        from inference.vllm_backend import VLLMInferenceEngine

        self.engine = VLLMInferenceEngine(
            model_path=model_path,
            tensor_parallel_size=int(qwen_cfg.get("tensor_parallel_size", 1)),
            dtype=str(qwen_cfg.get("dtype", "bfloat16")),
            mm_encoder_tp_mode=str(qwen_cfg.get("mm_encoder_tp_mode", "data")),
            enable_expert_parallel=bool(qwen_cfg.get("enable_expert_parallel", False)),
            gpu_memory_utilization=float(qwen_cfg.get("gpu_memory_utilization", 0.88)),
            max_model_len=_none_or_int(qwen_cfg.get("max_model_len")),
            max_num_seqs=_none_or_int(qwen_cfg.get("max_num_seqs")),
            enforce_eager=bool(qwen_cfg.get("enforce_eager", False)),
            system_prompt=self.system_prompt or None,
            include_frame_timestamps=bool(qwen_cfg.get("include_frame_timestamps", False)),
        )
        LOGGER.info("QwenVLLMInferencer initialized: model_path=%s", model_path)

    @staticmethod
    def _prepare_vllm_env() -> None:
        import torch

        torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        if torch_lib_path not in current_ld.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{current_ld}:{torch_lib_path}" if current_ld else torch_lib_path
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    def predict(self, video_path: str, query: str) -> dict[str, Any]:
        prompt = build_grounding_prompt(query, prompt_template=self.query_prompt_template)
        try:
            video_params = {
                "type": "video",
                "video": video_path,
                "total_pixels": int(self.video_cfg.get("total_pixels", 5242880)),
                "min_pixels": int(self.video_cfg.get("min_pixels", 16384)),
                "max_frames": int(self.video_cfg.get("max_frames", 64)),
                "sample_fps": float(self.video_cfg.get("sample_fps", 2.0)),
            }
            if self.video_cfg.get("max_pixels") is not None:
                video_params["max_pixels"] = int(self.video_cfg.get("max_pixels"))
            if self.video_cfg.get("frame_min_token") is not None:
                video_params["frame_min_token"] = int(self.video_cfg.get("frame_min_token"))
            if self.video_cfg.get("frame_max_token") is not None:
                video_params["frame_max_token"] = int(self.video_cfg.get("frame_max_token"))
            if self.video_cfg.get("frame_token_only") is not None:
                video_params["frame_token_only"] = bool(self.video_cfg.get("frame_token_only"))

            output_text = self.engine.inference_video_only(
                video_path=video_path,
                query=prompt,
                video_params=video_params,
                max_new_tokens=int(self.gen_cfg.get("max_new_tokens", 2048)),
                temperature=float(self.gen_cfg.get("temperature", 0.0)),
                top_p=float(self.gen_cfg.get("top_p", 1.0)),
                top_k=int(self.gen_cfg.get("top_k", -1)),
                repetition_penalty=float(self.gen_cfg.get("repetition_penalty", 1.0)),
                use_tqdm=bool(self.gen_cfg.get("use_tqdm", False)),
            )
            parsed = parse_model_output_with_metadata(output_text)
            return {
                "raw_text": output_text,
                "intervals": parsed["intervals"],
                "response_text": parsed["raw_text"],
                "reason": parsed["reason"],
                "event_timeline": parsed["event_timeline"],
                "refined_query": parsed["refined_query"],
                "error": "",
            }
        except Exception as exc:
            return {
                "raw_text": "",
                "intervals": [],
                "response_text": "",
                "reason": "",
                "event_timeline": [],
                "refined_query": "",
                "error": str(exc),
            }


def _none_or_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step2: 配置驱动推理（GeminiKSWrapper + 本地vLLM Qwen3VL-30B）"
    )
    parser.add_argument("--config", required=True, help="YAML config path.")

    # 可选覆盖：优先级高于配置文件
    parser.add_argument("--input", default="", help="Override input JSONL path.")
    parser.add_argument("--output", default="", help="Override output JSONL path.")
    parser.add_argument("--format", choices=("final", "debug"), default="", help="Override output format.")
    parser.add_argument("--models", choices=("gemini", "qwen", "both"), default="", help="Override models.")

    parser.add_argument("--resume", dest="resume", action="store_true", help="Override: resume=True")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Override: resume=False")
    parser.set_defaults(resume=None)

    parser.add_argument(
        "--overwrite-existing",
        dest="overwrite_existing",
        action="store_true",
        help="Override: overwrite existing model fields.",
    )
    parser.add_argument(
        "--no-overwrite-existing",
        dest="overwrite_existing",
        action="store_false",
        help="Override: do not overwrite existing model fields.",
    )
    parser.set_defaults(overwrite_existing=None)

    parser.add_argument(
        "--skip-missing-video",
        dest="skip_missing_video",
        action="store_true",
        help="Override: skip samples when local video file missing.",
    )
    parser.add_argument(
        "--no-skip-missing-video",
        dest="skip_missing_video",
        action="store_false",
        help="Override: do not skip missing videos.",
    )
    parser.set_defaults(skip_missing_video=None)

    parser.add_argument("--max-samples", type=int, default=None, help="Override max samples.")
    parser.add_argument("--sleep-seconds", type=float, default=None, help="Override per-sample sleep.")
    parser.add_argument("--progress-every", type=int, default=None, help="Override progress interval.")
    parser.add_argument("--flush-every", type=int, default=None, help="Override flush interval (rows).")

    parser.add_argument("--dp-enabled", dest="dp_enabled", action="store_true", help="Override distributed enabled.")
    parser.add_argument("--no-dp-enabled", dest="dp_enabled", action="store_false", help="Override distributed off.")
    parser.set_defaults(dp_enabled=None)
    parser.add_argument("--dp-world-size", type=int, default=None, help="Override distributed world size.")
    parser.add_argument("--dp-rank", type=int, default=None, help="Override distributed rank.")
    parser.add_argument("--dp-local-rank", type=int, default=None, help="Override distributed local rank.")
    return parser.parse_args()


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    content = _load_yaml_with_base(config_path)
    if not isinstance(content, dict):
        raise ValueError("Config root must be a YAML object.")
    return content


def _load_yaml_with_base(config_path: Path) -> dict[str, Any]:
    visited: set[Path] = set()

    def _load(path: Path) -> dict[str, Any]:
        resolved = path.resolve()
        if resolved in visited:
            raise ValueError(f"Cyclic __base__ detected at: {resolved}")
        visited.add(resolved)

        with path.open("r", encoding="utf-8") as handle:
            content = yaml.safe_load(handle) or {}
        if not isinstance(content, dict):
            raise ValueError(f"Config must be a dict object: {path}")

        base_value = content.pop("__base__", None)
        if not base_value:
            return content

        base_path = Path(base_value)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        if not base_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_path}")
        base_cfg = _load(base_path)
        return _merge_dict(base_cfg, content)

    return _load(config_path)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_output_path_by_format(output_path: str, output_format: str) -> str:
    path = Path(output_path)
    if output_format == "final":
        if path.suffix.lower() != ".jsonl":
            path = path.with_suffix(".jsonl")
    elif output_format == "debug":
        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")
    return str(path)


def resolve_runtime_config(cfg: dict[str, Any], args: argparse.Namespace) -> RuntimeConfig:
    runtime_cfg = dict(cfg.get("runtime", {}))

    input_jsonl = args.input or runtime_cfg.get("input_jsonl", "")
    output_jsonl = args.output or runtime_cfg.get("output_jsonl", "")
    output_format = str(args.format or runtime_cfg.get("format", "final")).strip().lower()
    models = args.models or runtime_cfg.get("models", "both")

    if not input_jsonl:
        raise ValueError("Missing `runtime.input_jsonl` in config (or --input).")
    if not output_jsonl:
        raise ValueError("Missing `runtime.output_jsonl` in config (or --output).")
    if output_format not in ("final", "debug"):
        raise ValueError(f"Invalid runtime.format={output_format}, expected final/debug.")
    output_jsonl = _normalize_output_path_by_format(str(output_jsonl), output_format)

    resume = bool(runtime_cfg.get("resume", False)) if args.resume is None else bool(args.resume)
    overwrite_existing = (
        bool(runtime_cfg.get("overwrite_existing", False))
        if args.overwrite_existing is None
        else bool(args.overwrite_existing)
    )
    skip_missing_video = (
        bool(runtime_cfg.get("skip_missing_video", False))
        if args.skip_missing_video is None
        else bool(args.skip_missing_video)
    )
    max_samples = int(runtime_cfg.get("max_samples", 0)) if args.max_samples is None else int(args.max_samples)
    sleep_seconds = (
        float(runtime_cfg.get("sleep_seconds", 0.0))
        if args.sleep_seconds is None
        else float(args.sleep_seconds)
    )
    progress_every = (
        int(runtime_cfg.get("progress_every", 20))
        if args.progress_every is None
        else int(args.progress_every)
    )
    flush_every = int(runtime_cfg.get("flush_every", 5)) if args.flush_every is None else int(args.flush_every)
    flush_every = max(1, flush_every)

    return RuntimeConfig(
        input_jsonl=str(input_jsonl),
        output_jsonl=str(output_jsonl),
        output_format=output_format,
        models=str(models),
        resume=resume,
        overwrite_existing=overwrite_existing,
        skip_missing_video=skip_missing_video,
        max_samples=max_samples,
        sleep_seconds=sleep_seconds,
        progress_every=progress_every,
        flush_every=flush_every,
    )


def _first_env_int(keys: list[str], default: int) -> int:
    for key in keys:
        value = os.getenv(key)
        if value is None or not str(value).strip():
            continue
        try:
            return int(str(value).strip())
        except ValueError:
            LOGGER.warning("Ignore non-integer env %s=%s", key, value)
    return default


def resolve_distributed_config(cfg: dict[str, Any], args: argparse.Namespace) -> DistributedConfig:
    runtime_cfg = dict(cfg.get("runtime", {}))
    dist_cfg = dict(runtime_cfg.get("distributed", {}))
    mpi_cfg = dict(dist_cfg.get("mpi", {}))

    rank_env_keys = list(dist_cfg.get("rank_env_keys", ["RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"]))
    world_env_keys = list(
        dist_cfg.get("world_size_env_keys", ["WORLD_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"])
    )
    local_rank_env_keys = list(
        dist_cfg.get("local_rank_env_keys", ["LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"])
    )

    rank_default = int(dist_cfg.get("rank", 0))
    world_default = int(dist_cfg.get("world_size", 1))
    local_rank_default = int(dist_cfg.get("local_rank", 0))

    rank = _first_env_int(rank_env_keys, rank_default)
    world_size = _first_env_int(world_env_keys, world_default)
    local_rank = _first_env_int(local_rank_env_keys, local_rank_default)

    if args.dp_rank is not None:
        rank = int(args.dp_rank)
    if args.dp_world_size is not None:
        world_size = int(args.dp_world_size)
    if args.dp_local_rank is not None:
        local_rank = int(args.dp_local_rank)

    auto_enabled = world_size > 1
    cfg_enabled = bool(dist_cfg.get("enabled", False))
    enabled = cfg_enabled or auto_enabled
    if args.dp_enabled is not None:
        enabled = bool(args.dp_enabled)

    if world_size <= 0:
        raise ValueError(f"Invalid distributed world_size={world_size}, must be >= 1.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid distributed rank={rank} for world_size={world_size}.")

    shard_output = bool(dist_cfg.get("shard_output", True))
    add_global_index_field = bool(dist_cfg.get("add_global_index_field", True))
    global_index_field = str(dist_cfg.get("global_index_field", "_dp_global_index")).strip() or "_dp_global_index"
    hostfile = str(mpi_cfg.get("hostfile", dist_cfg.get("hostfile", ""))).strip()
    mpi_hostfile = str(mpi_cfg.get("mpi_hostfile", dist_cfg.get("mpi_hostfile", ""))).strip()

    if not enabled:
        world_size = 1
        rank = 0
        local_rank = 0

    return DistributedConfig(
        enabled=enabled,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        shard_output=shard_output,
        add_global_index_field=add_global_index_field,
        global_index_field=global_index_field,
        hostfile=hostfile,
        mpi_hostfile=mpi_hostfile,
    )


def resolve_prompt_config(cfg: dict[str, Any]) -> PromptConfig:
    prompt_cfg = dict(cfg.get("prompt", {}))
    enabled = bool(prompt_cfg.get("enable", True))
    default_query_template = GROUNDING_PROMPT_TEMPLATE

    if not enabled:
        return PromptConfig(
            enabled=False,
            system_prompt="",
            query_prompt_template=default_query_template,
        )

    base_system_prompt = str(prompt_cfg.get("system_prompt", "")).strip()
    guidance_prompt = str(prompt_cfg.get("guidance_prompt", "")).strip()
    system_parts = [part for part in (base_system_prompt, guidance_prompt) if part]
    final_system_prompt = "\n\n".join(system_parts).strip()
    query_template = str(prompt_cfg.get("query_template", "")).strip() or default_query_template

    return PromptConfig(
        enabled=True,
        system_prompt=final_system_prompt,
        query_prompt_template=query_template,
    )


def should_run_field(row: dict[str, Any], target_field: str, overwrite: bool) -> bool:
    if overwrite:
        return True
    value = row.get(target_field)
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def apply_gemini_result(row: dict[str, Any], result: dict[str, Any]) -> None:
    row["gemini3pro_raw_output"] = result["raw_text"]
    row["gemini3pro_tgt"] = result["response_text"] or result["raw_text"]
    row["gemini3pro_pred_intervals"] = result["intervals"]
    row["gemini3pro_event_timeline"] = result.get("event_timeline", [])
    row["gemini3pro_refined_query"] = result.get("refined_query", "")
    row["gemini3pro_reason"] = result.get("reason", "")
    row["iou_gemini3pro"] = round(float(temporal_iou(row.get("time_gt"), result["intervals"])), 6)
    if result["error"]:
        row["gemini3pro_error"] = result["error"]


def apply_qwen_result(row: dict[str, Any], result: dict[str, Any]) -> None:
    row["qwen3vl_30b_raw_output"] = result["raw_text"]
    row["qwen3vl_30b_tgt"] = result["response_text"] or result["raw_text"]
    row["qwen3vl_30b_pred_intervals"] = result["intervals"]
    row["qwen3vl_30b_event_timeline"] = result.get("event_timeline", [])
    row["qwen3vl_30b_refined_query"] = result.get("refined_query", "")
    row["qwen3vl_30b_reason"] = result.get("reason", "")
    row["iou_qwen3vl_30b"] = round(float(temporal_iou(row.get("time_gt"), result["intervals"])), 6)
    if result["error"]:
        row["qwen3vl_30b_error"] = result["error"]


def prune_unused_model_fields(row: dict[str, Any], run_gemini: bool, run_qwen: bool) -> None:
    if run_gemini and not run_qwen:
        for key in QWEN_FIELDS:
            row.pop(key, None)
        if "iou_gemini3pro" in row:
            row["iou"] = row["iou_gemini3pro"]
    elif run_qwen and not run_gemini:
        for key in GEMINI_FIELDS:
            row.pop(key, None)
        if "iou_qwen3vl_30b" in row:
            row["iou"] = row["iou_qwen3vl_30b"]


def _is_video_missing(video_path: str, skip_missing_video: bool) -> bool:
    if not skip_missing_video:
        return False
    if video_path.startswith("http://") or video_path.startswith("https://") or video_path.startswith("file://"):
        return False
    return not Path(video_path).exists()


def _build_rank_output_path(base_output: Path, rank: int) -> Path:
    suffix = base_output.suffix
    if suffix:
        return base_output.with_name(f"{base_output.stem}.rank{rank:05d}{suffix}")
    return base_output.with_name(f"{base_output.name}.rank{rank:05d}")


def _indent_block(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" if line else line for line in text.splitlines())


def _load_debug_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Debug file must be JSON array/object: {path}")


class FinalRowWriter:
    def __init__(self, output_path: Path, mode: str) -> None:
        self._handle = output_path.open(mode, encoding="utf-8")

    def write_row(self, row: dict[str, Any]) -> None:
        self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class DebugRowWriter:
    def __init__(self, output_path: Path, existing_rows: list[dict[str, Any]], indent: int = 2) -> None:
        self._handle = output_path.open("w", encoding="utf-8")
        self._indent = indent
        self._first = True
        self._handle.write("[\n")
        for row in existing_rows:
            self.write_row(row)

    def write_row(self, row: dict[str, Any]) -> None:
        serialized = json.dumps(row, ensure_ascii=False, indent=self._indent)
        if not self._first:
            self._handle.write(",\n")
        self._handle.write(_indent_block(serialized, spaces=2))
        self._first = False

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        self._handle.write("\n]\n")
        self._handle.close()


def _build_row_writer(output_path: Path, output_format: str, resume: bool) -> tuple[Any, int]:
    if output_format == "final":
        resume_lines = count_jsonl_lines(output_path) if resume and output_path.exists() else 0
        mode = "a" if resume_lines > 0 else "w"
        return FinalRowWriter(output_path=output_path, mode=mode), resume_lines

    existing_rows: list[dict[str, Any]] = []
    if resume and output_path.exists():
        existing_rows = _load_debug_rows(output_path)
    resume_lines = len(existing_rows)
    return DebugRowWriter(output_path=output_path, existing_rows=existing_rows, indent=2), resume_lines


def _maybe_flush(row_writer: Any, processed: int, flush_every: int) -> None:
    if flush_every > 0 and processed > 0 and processed % flush_every == 0:
        row_writer.flush()


def iter_assigned_rows(
    input_path: Path,
    resume_lines: int,
    max_samples: int,
    distributed: DistributedConfig,
):
    seen_assigned = 0
    yielded = 0
    global_index = -1
    for _, row in read_jsonl(input_path):
        global_index += 1
        if distributed.enabled and distributed.world_size > 1:
            if global_index % distributed.world_size != distributed.rank:
                continue
        seen_assigned += 1
        if seen_assigned <= resume_lines:
            continue
        if max_samples > 0 and yielded >= max_samples:
            break
        yielded += 1
        yield global_index, row


def run_inference_sequential(
    runtime: RuntimeConfig,
    run_gemini: bool,
    run_qwen: bool,
    gemini_engine: GeminiKSInferencer | None,
    qwen_engine: QwenVLLMInferencer | None,
    input_path: Path,
    row_writer: Any,
    resume_lines: int,
    distributed: DistributedConfig,
) -> tuple[int, int]:
    processed = 0
    skipped_missing_video = 0

    for global_index, row in iter_assigned_rows(
        input_path=input_path,
        resume_lines=resume_lines,
        max_samples=runtime.max_samples,
        distributed=distributed,
    ):
        video_path = str(row.get("video_path", "")).strip()
        query = str(row.get("query", "")).strip()
        if not video_path or not query:
            row["pipeline_error"] = "missing video_path or query"
            prune_unused_model_fields(row, run_gemini=run_gemini, run_qwen=run_qwen)
            if distributed.enabled and distributed.add_global_index_field:
                row[distributed.global_index_field] = global_index
            row_writer.write_row(row)
            processed += 1
            _maybe_flush(row_writer, processed, runtime.flush_every)
            continue

        if _is_video_missing(video_path, runtime.skip_missing_video):
            row["pipeline_error"] = f"video not found: {video_path}"
            skipped_missing_video += 1
            prune_unused_model_fields(row, run_gemini=run_gemini, run_qwen=run_qwen)
            if distributed.enabled and distributed.add_global_index_field:
                row[distributed.global_index_field] = global_index
            row_writer.write_row(row)
            processed += 1
            _maybe_flush(row_writer, processed, runtime.flush_every)
            continue

        if run_gemini and gemini_engine and should_run_field(row, "gemini3pro_tgt", runtime.overwrite_existing):
            apply_gemini_result(row, gemini_engine.predict(video_path=video_path, query=query))

        if run_qwen and qwen_engine and should_run_field(row, "qwen3vl_30b_tgt", runtime.overwrite_existing):
            apply_qwen_result(row, qwen_engine.predict(video_path=video_path, query=query))

        prune_unused_model_fields(row, run_gemini=run_gemini, run_qwen=run_qwen)
        if distributed.enabled and distributed.add_global_index_field:
            row[distributed.global_index_field] = global_index
        row_writer.write_row(row)
        processed += 1
        _maybe_flush(row_writer, processed, runtime.flush_every)

        if runtime.sleep_seconds > 0:
            time.sleep(runtime.sleep_seconds)
        if runtime.progress_every > 0 and processed % runtime.progress_every == 0:
            LOGGER.info("processed=%s", processed)
    row_writer.flush()

    return processed, skipped_missing_video


def run_inference_with_gemini_concurrency(
    runtime: RuntimeConfig,
    run_gemini: bool,
    run_qwen: bool,
    gemini_engine: GeminiKSInferencer,
    qwen_engine: QwenVLLMInferencer | None,
    input_path: Path,
    row_writer: Any,
    resume_lines: int,
    distributed: DistributedConfig,
) -> tuple[int, int]:
    processed = 0
    skipped_missing_video = 0
    submitted_rows = 0
    next_seq = 0
    next_write_seq = 0
    max_pending = max(32, gemini_engine.max_workers * 4)

    pending_rows: dict[int, dict[str, Any]] = {}
    pending_futures: dict[int, Future[dict[str, Any]] | None] = {}
    pending_qwen_meta: dict[int, tuple[bool, str, str]] = {}
    pending_global_idx: dict[int, int] = {}

    def flush_ready_rows(block: bool) -> None:
        nonlocal processed, next_write_seq
        while next_write_seq in pending_rows:
            row = pending_rows[next_write_seq]
            gemini_future = pending_futures[next_write_seq]

            if gemini_future is not None:
                if not block and not gemini_future.done():
                    break
                if block and not gemini_future.done():
                    LOGGER.info(
                        "Waiting for Gemini result: seq=%s pending=%s",
                        next_write_seq,
                        len(pending_rows),
                    )
                try:
                    gemini_result = gemini_future.result()
                except Exception as exc:
                    gemini_result = {
                        "raw_text": "",
                        "intervals": [],
                        "response_text": "",
                        "reason": "",
                        "event_timeline": [],
                        "refined_query": "",
                        "error": f"Gemini future error: {exc}",
                    }
                apply_gemini_result(row, gemini_result)

            need_qwen, video_path, query = pending_qwen_meta[next_write_seq]
            if need_qwen and run_qwen and qwen_engine:
                apply_qwen_result(row, qwen_engine.predict(video_path=video_path, query=query))

            prune_unused_model_fields(row, run_gemini=run_gemini, run_qwen=run_qwen)
            if distributed.enabled and distributed.add_global_index_field:
                row[distributed.global_index_field] = pending_global_idx[next_write_seq]
            row_writer.write_row(row)
            processed += 1
            _maybe_flush(row_writer, processed, runtime.flush_every)
            if runtime.sleep_seconds > 0:
                time.sleep(runtime.sleep_seconds)
            if runtime.progress_every > 0 and processed % runtime.progress_every == 0:
                LOGGER.info("processed=%s", processed)

            del pending_rows[next_write_seq]
            del pending_futures[next_write_seq]
            del pending_qwen_meta[next_write_seq]
            del pending_global_idx[next_write_seq]
            next_write_seq += 1

    with ThreadPoolExecutor(max_workers=gemini_engine.max_workers, thread_name_prefix="gemini") as executor:
        for global_index, row in iter_assigned_rows(
            input_path=input_path,
            resume_lines=resume_lines,
            max_samples=runtime.max_samples,
            distributed=distributed,
        ):
            video_path = str(row.get("video_path", "")).strip()
            query = str(row.get("query", "")).strip()

            if not video_path or not query:
                row["pipeline_error"] = "missing video_path or query"
                pending_rows[next_seq] = row
                pending_futures[next_seq] = None
                pending_qwen_meta[next_seq] = (False, "", "")
                pending_global_idx[next_seq] = global_index
                next_seq += 1
                submitted_rows += 1
                flush_ready_rows(block=False)
                continue

            if _is_video_missing(video_path, runtime.skip_missing_video):
                row["pipeline_error"] = f"video not found: {video_path}"
                skipped_missing_video += 1
                pending_rows[next_seq] = row
                pending_futures[next_seq] = None
                pending_qwen_meta[next_seq] = (False, "", "")
                pending_global_idx[next_seq] = global_index
                next_seq += 1
                submitted_rows += 1
                flush_ready_rows(block=False)
                continue

            need_gemini = run_gemini and should_run_field(row, "gemini3pro_tgt", runtime.overwrite_existing)
            need_qwen = run_qwen and should_run_field(row, "qwen3vl_30b_tgt", runtime.overwrite_existing)
            gemini_future = None
            if need_gemini:
                gemini_future = executor.submit(gemini_engine.predict, video_path, query)

            pending_rows[next_seq] = row
            pending_futures[next_seq] = gemini_future
            pending_qwen_meta[next_seq] = (need_qwen, video_path, query)
            pending_global_idx[next_seq] = global_index
            next_seq += 1
            submitted_rows += 1

            flush_ready_rows(block=False)
            while len(pending_rows) >= max_pending:
                flush_ready_rows(block=True)

        flush_ready_rows(block=True)
        row_writer.flush()

    return processed, skipped_missing_video


def run_inference(
    runtime: RuntimeConfig,
    run_gemini: bool,
    run_qwen: bool,
    gemini_engine: GeminiKSInferencer | None,
    qwen_engine: QwenVLLMInferencer | None,
    distributed: DistributedConfig,
) -> None:
    input_path = Path(runtime.input_jsonl)
    output_path = Path(runtime.output_jsonl)
    if distributed.enabled and distributed.world_size > 1 and distributed.shard_output:
        output_path = _build_rank_output_path(output_path, distributed.rank)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_writer, resume_lines = _build_row_writer(
        output_path=output_path,
        output_format=runtime.output_format,
        resume=runtime.resume,
    )
    LOGGER.info(
        "Run inference: input=%s output=%s format=%s models=%s resume_lines=%s",
        input_path,
        output_path,
        runtime.output_format,
        runtime.models,
        resume_lines,
    )
    LOGGER.info(
        "Distributed mode: enabled=%s world_size=%s rank=%s local_rank=%s",
        distributed.enabled,
        distributed.world_size,
        distributed.rank,
        distributed.local_rank,
    )
    if distributed.hostfile or distributed.mpi_hostfile:
        LOGGER.info(
            "MPI hostfiles: hostfile=%s mpi_hostfile=%s",
            distributed.hostfile or "<empty>",
            distributed.mpi_hostfile or "<empty>",
        )

    use_gemini_concurrency = run_gemini and gemini_engine is not None and gemini_engine.max_workers > 1
    try:
        if use_gemini_concurrency:
            LOGGER.info(
                "Gemini concurrency enabled: max_workers=%s pending_window=%s",
                gemini_engine.max_workers,
                max(32, gemini_engine.max_workers * 4),
            )
            processed, skipped_missing_video = run_inference_with_gemini_concurrency(
                runtime=runtime,
                run_gemini=run_gemini,
                run_qwen=run_qwen,
                gemini_engine=gemini_engine,
                qwen_engine=qwen_engine,
                input_path=input_path,
                row_writer=row_writer,
                resume_lines=resume_lines,
                distributed=distributed,
            )
        else:
            processed, skipped_missing_video = run_inference_sequential(
                runtime=runtime,
                run_gemini=run_gemini,
                run_qwen=run_qwen,
                gemini_engine=gemini_engine,
                qwen_engine=qwen_engine,
                input_path=input_path,
                row_writer=row_writer,
                resume_lines=resume_lines,
                distributed=distributed,
            )
    finally:
        row_writer.close()

    print(
        f"[DONE] inference finished: input={input_path} output={output_path} "
        f"processed={processed} resume_lines={resume_lines} skipped_missing_video={skipped_missing_video}"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    cfg = load_yaml_config(args.config)
    runtime = resolve_runtime_config(cfg, args)
    distributed = resolve_distributed_config(cfg, args)
    prompt_cfg = resolve_prompt_config(cfg)
    LOGGER.info(
        "Prompt config resolved: enabled=%s system_prompt_chars=%s query_template_chars=%s",
        prompt_cfg.enabled,
        len(prompt_cfg.system_prompt),
        len(prompt_cfg.query_prompt_template),
    )

    models = runtime.models.lower()
    if models not in ("gemini", "qwen", "both"):
        raise ValueError(f"Invalid runtime.models={runtime.models}, expected one of gemini/qwen/both.")

    gemini_cfg = dict(cfg.get("gemini", {}))
    qwen_cfg = dict(cfg.get("qwen_vllm", {}))

    run_gemini = models in ("gemini", "both") and bool(gemini_cfg.get("enabled", True))
    run_qwen = models in ("qwen", "both") and bool(qwen_cfg.get("enabled", True))
    if not run_gemini and not run_qwen:
        raise ValueError("No active model backend. Check runtime.models and enabled flags.")

    gemini_engine: GeminiKSInferencer | None = None
    qwen_engine: QwenVLLMInferencer | None = None

    if run_gemini:
        gemini_engine = GeminiKSInferencer(gemini_cfg=gemini_cfg, prompt_cfg=prompt_cfg)
    if run_qwen:
        qwen_engine = QwenVLLMInferencer(qwen_cfg=qwen_cfg, prompt_cfg=prompt_cfg)

    run_inference(
        runtime=runtime,
        run_gemini=run_gemini,
        run_qwen=run_qwen,
        gemini_engine=gemini_engine,
        qwen_engine=qwen_engine,
        distributed=distributed,
    )


def parse_merge_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Step2 distributed output shards into one JSONL.")
    parser.add_argument("--shard-pattern", required=True, help="Glob pattern, e.g. outputs/step2_predictions.rank*.jsonl")
    parser.add_argument("--output", required=True, help="Merged output JSONL path.")
    parser.add_argument("--global-index-field", default="_dp_global_index", help="Field used for merge ordering.")
    parser.add_argument(
        "--keep-global-index-field",
        action="store_true",
        help="Keep global index field in merged rows.",
    )
    return parser.parse_args()


def merge_shards_main() -> None:
    args = parse_merge_args()
    shard_paths = sorted(Path(path) for path in glob.glob(args.shard_pattern))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files matched pattern: {args.shard_pattern}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    iterators = [read_jsonl(path) for path in shard_paths]
    heap: list[tuple[int, int, dict[str, Any]]] = []

    def push_next(shard_idx: int) -> None:
        try:
            _, row = next(iterators[shard_idx])
        except StopIteration:
            return
        if args.global_index_field not in row:
            raise KeyError(
                f"Missing `{args.global_index_field}` in shard={shard_paths[shard_idx]}."
                " Enable runtime.distributed.add_global_index_field."
            )
        global_idx = int(row[args.global_index_field])
        heapq.heappush(heap, (global_idx, shard_idx, row))

    for idx in range(len(iterators)):
        push_next(idx)

    merged = 0
    with output_path.open("w", encoding="utf-8") as writer:
        while heap:
            _, shard_idx, row = heapq.heappop(heap)
            if not args.keep_global_index_field:
                row.pop(args.global_index_field, None)
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            merged += 1
            push_next(shard_idx)

    print(f"[DONE] merged shards: count={len(shard_paths)} rows={merged} output={output_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "merge-shards":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        merge_shards_main()
    else:
        main()
