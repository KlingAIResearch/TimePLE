from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark_loader import BenchmarkSample


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelResponse:
    sample_id: str
    raw_text: str
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRunner(ABC):
    def __init__(self, config: dict[str, Any], *, config_path: Path) -> None:
        self.config = config
        self.config_path = config_path
        self.model_cfg = dict(config.get("model", {}))
        self.dataset_cfg = dict(config.get("dataset", {}))
        self.prompt_cfg = dict(config.get("prompt", {}))
        self.inference_cfg = dict(config.get("inference", {}))
        self._validate_prompt_config()

    @abstractmethod
    def predict_batch(self, samples: list[BenchmarkSample]) -> list[ModelResponse]:
        raise NotImplementedError

    def _prompt_context(self) -> str:
        prompt_meta = self.prompt_cfg.get("meta")
        if isinstance(prompt_meta, dict):
            profile_name = prompt_meta.get("profile_name")
            profile_path = prompt_meta.get("profile_path")
            if profile_name or profile_path:
                return f"profile_name={profile_name} profile_path={profile_path}"
        return f"config_path={self.config_path}"

    def _require_prompt_string(self, field_name: str) -> str:
        value = self.prompt_cfg.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Missing non-empty `prompt.{field_name}` for model="
                f"{self.model_cfg.get('name')} dataset={self.dataset_cfg.get('name')} "
                f"({self._prompt_context()})"
            )
        return value

    def _validate_prompt_config(self) -> None:
        self._require_prompt_string("user_prompt")
        if self.prompt_cfg.get("enabled_sys_prompt", False):
            self._require_prompt_string("system_prompt")
        if self.prompt_cfg.get("enabled_format_prompt", False):
            self._require_prompt_string("format_prompt")

    def render_user_prompt(self, sample: BenchmarkSample) -> str:
        template = self._require_prompt_string("user_prompt")

        format_values = dict(sample.raw_sample)
        format_values.update(
            {
                "query": sample.query,
                "sample_id": sample.sample_id,
                "video_path": sample.video_path,
                "gt_timestamps": sample.gt_timestamps,
                "global_index": sample.global_index,
            }
        )
        prompt = template.format(**format_values)

        if self.prompt_cfg.get("enabled_format_prompt", False):
            format_prompt = self._require_prompt_string("format_prompt")
            separator = "" if prompt.endswith("\n") else "\n"
            prompt = f"{prompt}{separator}{format_prompt}"
        return prompt

    def get_system_prompt(self) -> str | None:
        if not self.prompt_cfg.get("enabled_sys_prompt", False):
            return None
        return self._require_prompt_string("system_prompt")


class BaseVLLMRunner(BaseRunner, ABC):
    def __init__(self, config: dict[str, Any], *, config_path: Path) -> None:
        super().__init__(config, config_path=config_path)
        self.engine_cfg = dict(self.inference_cfg.get("engine", {}))
        self.generation_cfg = dict(self.inference_cfg.get("generation", {}))
        batching_cfg = dict(self.inference_cfg.get("batching", {}))
        self.batch_size = max(1, int(batching_cfg.get("batch_size", 1)))
        self._llm = None
        self._llm_init_error: str | None = None

    @abstractmethod
    def build_messages(self, sample: BenchmarkSample) -> list[dict[str, Any]]:
        raise NotImplementedError

    def get_chat_template_kwargs(self) -> dict[str, Any] | None:
        kwargs = self.engine_cfg.get("chat_template_kwargs")
        return dict(kwargs) if isinstance(kwargs, dict) else None

    def get_chat_mm_processor_kwargs(self) -> dict[str, Any] | None:
        kwargs = self.engine_cfg.get("chat_mm_processor_kwargs")
        return dict(kwargs) if isinstance(kwargs, dict) else None

    def get_llm_init_kwargs(self) -> dict[str, Any]:
        model_path = str(self.inference_cfg["model_path"])
        gpu_memory_utilization = float(self.engine_cfg.get("gpu_memory_utilization", 0.9))
        kv_cache_memory_bytes = self.engine_cfg.get("kv_cache_memory_bytes")
        if kv_cache_memory_bytes is not None:
            startup_gpu_memory_utilization = float(
                self.engine_cfg.get("startup_gpu_memory_utilization", 0.1)
            )
            if startup_gpu_memory_utilization <= 0.0 or startup_gpu_memory_utilization > 1.0:
                raise ValueError(
                    "startup_gpu_memory_utilization must be within (0, 1], "
                    f"got {startup_gpu_memory_utilization}."
                )
            if gpu_memory_utilization > startup_gpu_memory_utilization:
                LOGGER.info(
                    "kv_cache_memory_bytes=%s is set for model=%s; "
                    "clamping startup gpu_memory_utilization from %.3f to %.3f "
                    "to avoid vLLM's pre-allocation free-memory gate.",
                    kv_cache_memory_bytes,
                    model_path,
                    gpu_memory_utilization,
                    startup_gpu_memory_utilization,
                )
                gpu_memory_utilization = startup_gpu_memory_utilization

        kwargs: dict[str, Any] = {
            "model": model_path,
            "trust_remote_code": bool(self.inference_cfg.get("trust_remote_code", True)),
            "allowed_local_media_path": str(
                self.inference_cfg.get("allowed_local_media_path", "/")
            ),
            "tensor_parallel_size": int(self.engine_cfg.get("tensor_parallel_size", 1)),
            "dtype": self.engine_cfg.get("dtype", "auto"),
            "gpu_memory_utilization": gpu_memory_utilization,
            "seed": int(self.engine_cfg.get("seed", 0)),
        }

        passthrough_keys = [
            "max_model_len",
            "max_num_seqs",
            "kv_cache_memory_bytes",
            "enforce_eager",
            "mm_encoder_tp_mode",
            "enable_expert_parallel",
            "limit_mm_per_prompt",
            "tokenizer",
            "tokenizer_mode",
            "revision",
            "tokenizer_revision",
            "disable_custom_all_reduce",
            "hf_overrides",
            "media_io_kwargs",
        ]
        for key in passthrough_keys:
            if key in self.engine_cfg and self.engine_cfg[key] is not None:
                kwargs[key] = self.engine_cfg[key]

        mm_processor_kwargs = self.engine_cfg.get("mm_processor_kwargs")
        if isinstance(mm_processor_kwargs, dict) and mm_processor_kwargs:
            kwargs["mm_processor_kwargs"] = dict(mm_processor_kwargs)

        extra_kwargs = self.engine_cfg.get("extra_llm_kwargs")
        if isinstance(extra_kwargs, dict):
            kwargs.update(extra_kwargs)
        return kwargs

    def _validate_model_artifacts(self) -> None:
        model_path = Path(str(self.inference_cfg["model_path"])).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        direct_weight_patterns = (
            "model*.safetensors",
            "pytorch_model*.bin",
            "*.pt",
            "*.pth",
        )
        direct_weight_files: list[Path] = []
        for pattern in direct_weight_patterns:
            direct_weight_files.extend(model_path.glob(pattern))

        index_path = model_path / "model.safetensors.index.json"
        if index_path.exists():
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
            referenced_files = sorted(set(index_payload.get("weight_map", {}).values()))
            missing_files = [
                filename for filename in referenced_files if not (model_path / filename).exists()
            ]
            if missing_files:
                missing_preview = ", ".join(missing_files[:4])
                if len(missing_files) > 4:
                    missing_preview += ", ..."
                raise FileNotFoundError(
                    "Model weight shards referenced by model.safetensors.index.json are missing "
                    f"under {model_path}: {missing_preview}"
                )
            return

        direct_weight_files = [path for path in direct_weight_files if path.name != index_path.name]
        if not direct_weight_files:
            raise FileNotFoundError(
                f"No model weights found under {model_path}. "
                "Expected model.safetensors, model-*.safetensors, or pytorch_model*.bin."
            )

    def _preflight_vllm_runtime(self) -> None:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - runtime import failure
            raise RuntimeError(
                "Failed to import PyTorch while checking the vLLM runtime environment."
            ) from exc

        if torch.cuda.device_count() > 0:
            return

        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        raise RuntimeError(
            "No CUDA devices are visible to the eval_suite vLLM runner. "
            "This host cannot initialize vLLM for model evaluation. "
            f"Current CUDA_VISIBLE_DEVICES={cuda_visible_devices!r}."
        )

    def ensure_llm(self):
        if self._llm_init_error is not None:
            raise RuntimeError(self._llm_init_error)
        if self._llm is None:
            kwargs = self.get_llm_init_kwargs()
            try:
                self._validate_model_artifacts()
                self._preflight_vllm_runtime()
                import_started = time.monotonic()
                LOGGER.info(
                    "Preparing to import vLLM for model=%s family=%s",
                    self.inference_cfg.get("model_path"),
                    self.model_cfg.get("runner", {}).get("family"),
                )
                from vllm import LLM

                LOGGER.info(
                    "Imported vLLM in %.2fs for model=%s",
                    time.monotonic() - import_started,
                    self.inference_cfg.get("model_path"),
                )
                llm_init_started = time.monotonic()
                LOGGER.info(
                    "Initializing vLLM model=%s family=%s tensor_parallel_size=%s",
                    kwargs["model"],
                    self.model_cfg.get("runner", {}).get("family"),
                    kwargs.get("tensor_parallel_size"),
                )
                self._llm = LLM(**kwargs)
            except Exception as exc:
                self._llm_init_error = str(exc)
                LOGGER.error(
                    "Failed to initialize vLLM model=%s family=%s: %s",
                    kwargs["model"],
                    self.model_cfg.get("runner", {}).get("family"),
                    self._llm_init_error,
                )
                raise
            LOGGER.info(
                "Initialized vLLM in %.2fs for model=%s",
                time.monotonic() - llm_init_started,
                kwargs["model"],
            )
        return self._llm

    def build_sampling_params(self):
        from vllm import SamplingParams

        cfg = dict(self.generation_cfg)
        max_tokens = int(cfg.pop("max_tokens", cfg.pop("max_new_tokens", 512)))
        return SamplingParams(max_tokens=max_tokens, **cfg)

    def _extract_output_text(self, output: Any) -> str:
        outputs = getattr(output, "outputs", None)
        if not outputs:
            return ""
        first = outputs[0]
        return str(getattr(first, "text", ""))

    def _predict_batch_once(self, samples: list[BenchmarkSample]) -> list[ModelResponse]:
        llm = self.ensure_llm()
        messages_batch = [self.build_messages(sample) for sample in samples]
        sampling_params = [self.build_sampling_params() for _ in samples]
        outputs = llm.chat(
            messages=messages_batch,
            sampling_params=sampling_params,
            use_tqdm=False,
            chat_template_content_format="openai",
            add_generation_prompt=bool(self.engine_cfg.get("add_generation_prompt", True)),
            chat_template_kwargs=self.get_chat_template_kwargs(),
            mm_processor_kwargs=self.get_chat_mm_processor_kwargs(),
        )

        responses: list[ModelResponse] = []
        for sample, output in zip(samples, outputs):
            responses.append(
                ModelResponse(
                    sample_id=sample.sample_id,
                    raw_text=self._extract_output_text(output),
                    metadata={
                        "finish_reason": (
                            getattr(output.outputs[0], "finish_reason", None)
                            if getattr(output, "outputs", None)
                            else None
                        ),
                    },
                )
            )
        return responses

    def predict_batch(self, samples: list[BenchmarkSample]) -> list[ModelResponse]:
        try:
            return self._predict_batch_once(samples)
        except Exception as exc:
            if self._llm_init_error is not None:
                raise
            if len(samples) == 1:
                return [
                    ModelResponse(
                        sample_id=samples[0].sample_id,
                        raw_text="",
                        error=str(exc),
                    )
                ]

            LOGGER.warning(
                "Batch inference failed for batch_size=%s, falling back to single-sample retries: %s",
                len(samples),
                exc,
            )
            results: list[ModelResponse] = []
            for sample in samples:
                results.extend(self.predict_batch([sample]))
            return results


class BaseTransformersRunner(BaseRunner, ABC):
    def __init__(self, config: dict[str, Any], *, config_path: Path) -> None:
        super().__init__(config, config_path=config_path)
        self.engine_cfg = dict(self.inference_cfg.get("engine", {}))
        self.generation_cfg = dict(self.inference_cfg.get("generation", {}))
        batching_cfg = dict(self.inference_cfg.get("batching", {}))
        self.batch_size = max(1, int(batching_cfg.get("batch_size", 1)))
        self._model_init_error: str | None = None

    def get_device_string(self) -> str:
        explicit_device = self.engine_cfg.get("device")
        if explicit_device:
            return str(explicit_device)

        try:
            import torch
        except Exception:
            return "cpu"

        if not torch.cuda.is_available():
            return "cpu"

        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            visible_count = len(
                [item for item in visible_devices.split(",") if item.strip()]
            )
            if visible_count == 1:
                return "cuda:0"

        local_rank_raw = os.environ.get("LOCAL_RANK", "0")
        try:
            local_rank = int(local_rank_raw)
        except ValueError:
            local_rank = 0
        if local_rank >= torch.cuda.device_count():
            LOGGER.warning(
                "LOCAL_RANK=%s is outside visible CUDA device count=%s; using cuda:0.",
                local_rank,
                torch.cuda.device_count(),
            )
            return "cuda:0"
        return f"cuda:{local_rank}"

    def build_generation_config(self) -> dict[str, Any]:
        cfg = dict(self.generation_cfg)
        max_new_tokens = int(cfg.pop("max_tokens", cfg.pop("max_new_tokens", 512)))
        temperature = cfg.pop("temperature", None)
        top_p = cfg.pop("top_p", None)
        top_k = cfg.pop("top_k", None)
        repetition_penalty = cfg.pop("repetition_penalty", None)

        do_sample = bool(
            (temperature is not None and float(temperature) > 0.0)
            or (top_p is not None and float(top_p) < 1.0)
            or (top_k is not None and int(top_k) not in (0, -1))
        )

        result: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if repetition_penalty is not None:
            result["repetition_penalty"] = float(repetition_penalty)

        if do_sample:
            if temperature is not None:
                result["temperature"] = float(temperature)
            if top_p is not None:
                result["top_p"] = float(top_p)
            if top_k is not None:
                result["top_k"] = int(top_k)

        for key, value in cfg.items():
            if value is not None:
                result[key] = value
        return result

    @abstractmethod
    def _predict_batch_once(self, samples: list[BenchmarkSample]) -> list[ModelResponse]:
        raise NotImplementedError

    def predict_batch(self, samples: list[BenchmarkSample]) -> list[ModelResponse]:
        try:
            return self._predict_batch_once(samples)
        except Exception as exc:
            if len(samples) == 1:
                return [
                    ModelResponse(
                        sample_id=samples[0].sample_id,
                        raw_text="",
                        error=str(exc),
                    )
                ]

            LOGGER.warning(
                "Batch inference failed for batch_size=%s, falling back to single-sample retries: %s",
                len(samples),
                exc,
            )
            results: list[ModelResponse] = []
            for sample in samples:
                results.extend(self.predict_batch([sample]))
            return results
