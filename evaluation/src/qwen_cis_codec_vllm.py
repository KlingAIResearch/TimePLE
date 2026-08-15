from __future__ import annotations

import importlib.util
import json
import logging
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.qwen3_vl import Qwen3VLConfig, Qwen3VLProcessor

from qwen_codec_full_forward import (
    build_full_forward_attention,
    build_full_forward_positions,
    collect_full_mm_embeddings,
    compute_required_block_counts,
    extract_real_step_sampled_token_ids,
)
from vllm.config import CUDAGraphMode
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.qwen3_vl import (
    MULTIMODAL_REGISTRY,
    PromptReplacement,
    PromptUpdateDetails,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
    _cached_tensor,
    _create_qwen2vl_field_factory,
    _merge_multimodal_embeddings,
    compute_retained_tokens_count,
)
from vllm.model_executor.models.registry import ModelRegistry
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalFieldConfig,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_PATCHED = False
_CIS_CODEC_PKG_NAME = "_eval_suite_cis_codec"
_DEBUG_PATH = Path("/tmp/qwen_cis_codec_debug.jsonl")


def _resolve_cis_codec_dir() -> Path:
    return PROJECT_ROOT / "src" / "timeple"


def _append_debug(payload: dict[str, Any]) -> None:
    try:
        with _DEBUG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        LOGGER.exception("Failed to append qwen_cis_codec debug payload.")


def _coerce_eval_suite_video_metadata_items(
    raw_metadata_json: object,
    raw_metadata_legacy: object = None,
) -> list[dict[str, Any] | None] | None:
    raw_metadata = raw_metadata_json
    if raw_metadata is None:
        raw_metadata = raw_metadata_legacy
    if raw_metadata is None:
        return None
    if isinstance(raw_metadata, (bytes, bytearray)):
        raw_metadata = raw_metadata.decode("utf-8")
    if isinstance(raw_metadata, str):
        raw_metadata = json.loads(raw_metadata)
    if isinstance(raw_metadata, Mapping):
        return [dict(raw_metadata)]
    if isinstance(raw_metadata, (list, tuple)):
        if len(raw_metadata) == 1 and isinstance(raw_metadata[0], (list, tuple)):
            raw_metadata = raw_metadata[0]
        items: list[dict[str, Any] | None] = []
        for item in raw_metadata:
            if item is None:
                items.append(None)
            elif isinstance(item, Mapping):
                items.append(dict(item))
            else:
                items.append(dict(item))
        return items
    raise TypeError(
        "eval_suite video metadata must be JSON, a mapping, or a list of mappings."
    )


def _load_local_cis_codec_package():
    cached = sys.modules.get(_CIS_CODEC_PKG_NAME)
    if cached is not None:
        return cached

    cis_codec_dir = _resolve_cis_codec_dir()
    init_path = cis_codec_dir / "__init__.py"
    if not init_path.exists():
        raise FileNotFoundError(f"Missing local cis_codec package: {init_path}")

    spec = importlib.util.spec_from_file_location(
        _CIS_CODEC_PKG_NAME,
        init_path,
        submodule_search_locations=[str(cis_codec_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load local cis_codec package from {init_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[_CIS_CODEC_PKG_NAME] = module
    spec.loader.exec_module(module)
    return module


def _get_cis_codec_classes():
    from timeple import TimePLECodec, TimePLEInterfaceAdapter

    return TimePLEInterfaceAdapter, TimePLECodec


class Qwen3VLCISCodecConfig(Qwen3VLConfig):
    model_type = "qwen3_vl_cis_codec"

    def __init__(
        self,
        use_cis_codec: bool = True,
        timestamp_token_id: int = 151669,
        timespan_token_id: int = 151670,
        cis_codec_config: dict[str, Any] | None = None,
        use_cis_interface_adapter: bool = False,
        cis_interface_adapter: dict[str, Any] | None = None,
        freeze_vision: bool = False,
        freeze_language: bool = False,
        default_video_duration_sec: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        hidden_size = int(self.text_config.hidden_size)
        self.use_cis_codec = use_cis_codec
        self.timestamp_token_id = timestamp_token_id
        self.timespan_token_id = timespan_token_id
        self.use_cis_interface_adapter = bool(use_cis_interface_adapter)
        self.cis_interface_adapter = (
            deepcopy(cis_interface_adapter) if cis_interface_adapter is not None else None
        )
        self.freeze_vision = freeze_vision
        self.freeze_language = freeze_language
        self.default_video_duration_sec = float(default_video_duration_sec)
        self.cis_codec_config = cis_codec_config or {
            "token_dim": hidden_size,
            "transform": {
                "tau": 0.03,
                "eps": 1e-6,
                "point_threshold": 1e-4,
            },
            "grid": {
                "num_u_bins": 32,
                "num_v_bins": 16,
                "span_sigma_u": 0.04,
                "span_sigma_v": 0.06,
                "point_sigma_u": 0.02,
            },
            "encoder": {
                "hidden_dims": [512, 2048],
                "activation": "gelu",
                "dropout": 0.0,
                "use_layer_norm": True,
            },
            "decoder": {
                "trunk_hidden_dims": [2048, 1024],
                "activation": "gelu",
                "dropout": 0.0,
                "use_layer_norm": True,
            },
            "loss": {
                "lambda_type": 1.0,
                "lambda_point": 1.0,
                "lambda_span": 1.0,
                "lambda_interval_l1": 0.5,
                "lambda_interval_giou": 1.0,
                "lambda_point_l1": 1.0,
            },
        }

    def get_cis_codec_output_dim(self) -> int:
        return int(self.cis_codec_config.get("token_dim", self.text_config.hidden_size))


class Qwen3VLCISCodecProcessingInfo(Qwen3VLProcessingInfo):
    @staticmethod
    def _meta_get(metadata: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
        if hasattr(metadata, key):
            return getattr(metadata, key)
        if isinstance(metadata, Mapping):
            try:
                return metadata[key]
            except (KeyError, TypeError, AttributeError):
                return default
        return getattr(metadata, key, default)

    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3VLCISCodecConfig)

    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor:
        return self.ctx.get_hf_processor(
            Qwen3VLProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )

    def _calculate_time_segments(
        self,
        indices: list[int] | torch.Tensor,
        video_fps: float,
        merge_size: int,
    ) -> list[tuple[float, float]]:
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            indices = indices + [indices[-1]] * (merge_size - len(indices) % merge_size)

        timestamps = [idx / video_fps for idx in indices]
        segments: list[tuple[float, float]] = []
        for start_idx in range(0, len(timestamps), merge_size):
            start_time = float(timestamps[start_idx])
            end_time = float(timestamps[min(start_idx + merge_size - 1, len(timestamps) - 1)])
            if abs(end_time - start_time) < 0.01:
                end_time = start_time + 0.5
            segments.append((start_time, end_time))
        return segments

    def _get_video_time_segments_and_duration(
        self,
        metadata: Mapping[str, Any] | Any,
        do_sample_frames: bool | None = None,
        sampled_fps: float | None = None,
        sampled_num_frames: int | None = None,
    ) -> tuple[list[tuple[float, float]], float]:
        video_processor = self.get_video_processor()
        temporal_patch_size = getattr(video_processor, "temporal_patch_size", None)
        merge_size = int(temporal_patch_size or video_processor.merge_size)
        indices = self._meta_get(metadata, "frames_indices")
        if indices is None:
            raise ValueError("Missing `frames_indices` for qwen_cis_codec video input.")

        video_fps = float(self._meta_get(metadata, "fps"))
        if do_sample_frames is None:
            do_sample_frames = bool(self._meta_get(metadata, "do_sample_frames", False))

        if do_sample_frames:
            total_num_frames = int(self._meta_get(metadata, "total_num_frames"))
            if sampled_num_frames is not None:
                num_frames = int(sampled_num_frames)
            else:
                sampled_fps = sampled_fps if sampled_fps else float(video_processor.fps)
                num_frames = int(total_num_frames / video_fps * sampled_fps)

            num_frames = min(
                min(max(num_frames, int(video_processor.min_frames)), int(video_processor.max_frames)),
                total_num_frames,
            )
            indices = (
                torch.linspace(0, total_num_frames - 1, steps=num_frames)
                .round()
                .to(torch.int64)
                .tolist()
            )

        segments = self._calculate_time_segments(indices, video_fps, merge_size)

        duration = self._meta_get(metadata, "duration")
        if duration is None:
            total_num_frames = self._meta_get(metadata, "total_num_frames")
            if total_num_frames is not None:
                duration = float(total_num_frames) / float(video_fps)
        if duration is None:
            num_frames = self._meta_get(metadata, "num_frames")
            if num_frames is not None:
                duration = float(num_frames) / float(video_fps)
        if duration is None and indices:
            duration = (float(max(indices)) + 1.0) / float(video_fps)
        if duration is None and segments:
            duration = segments[-1][1]
        if duration is None or float(duration) <= 0.0:
            duration = 1.0

        return segments, float(duration)


class Qwen3VLCISCodecMultiModalProcessor(Qwen3VLMultiModalProcessor):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ):
        mm_data = dict(mm_data)
        processor_mm_kwargs = dict(mm_kwargs)
        explicit_video_metadata_items = _coerce_eval_suite_video_metadata_items(
            processor_mm_kwargs.pop("eval_suite_video_metadata_json", None),
            processor_mm_kwargs.pop("eval_suite_video_metadata", None),
        )
        processor = self.info.get_hf_processor(**processor_mm_kwargs)

        if videos := mm_data.pop("videos", []):
            video_grid_thw_lst = []
            pixel_values_videos_lst = []
            time_segments_per_video: list[list[tuple[float, float]]] = []
            video_durations_per_video: list[float] = []
            video_metadata_items = mm_data.pop("video_metadata", None)

            for video_idx, item in enumerate(videos):
                explicit_metadata = None
                if explicit_video_metadata_items is not None and video_idx < len(explicit_video_metadata_items):
                    explicit_metadata = explicit_video_metadata_items[video_idx]

                if explicit_metadata is not None:
                    video_array = item[0] if isinstance(item, tuple) and len(item) == 2 else item
                    metadata = explicit_metadata
                elif isinstance(item, tuple) and len(item) == 2:
                    video_array, metadata = item
                else:
                    video_array = item
                    metadata = None
                    if isinstance(video_metadata_items, list) and video_idx < len(video_metadata_items):
                        metadata = video_metadata_items[video_idx]
                    elif video_metadata_items is not None:
                        metadata = video_metadata_items

                video_mm_kwargs = dict(processor_mm_kwargs)

                from transformers.video_utils import VideoMetadata

                do_sample_frames = bool(self.info._meta_get(metadata, "do_sample_frames", False))
                if metadata is None:
                    metadata = VideoMetadata(
                        total_num_frames=len(video_array),
                        fps=None,
                        duration=None,
                        frames_indices=list(range(len(video_array))),
                    )
                elif not isinstance(metadata, VideoMetadata):
                    metadata = VideoMetadata(
                        **{k: metadata[k] for k in metadata if k != "do_sample_frames"}
                    )
                if "do_sample_frames" not in video_mm_kwargs:
                    video_mm_kwargs["do_sample_frames"] = do_sample_frames

                if metadata["fps"] is None and video_mm_kwargs.get("fps") is not None:
                    metadata["fps"] = float(video_mm_kwargs["fps"])
                    if metadata["duration"] is None and metadata["total_num_frames"] is not None:
                        metadata["duration"] = (
                            float(metadata["total_num_frames"]) / metadata["fps"]
                        )

                segments, video_duration = self.info._get_video_time_segments_and_duration(
                    metadata=metadata,
                    do_sample_frames=video_mm_kwargs["do_sample_frames"],
                    sampled_fps=video_mm_kwargs.get("fps"),
                    sampled_num_frames=video_mm_kwargs.get("num_frames"),
                )
                time_segments_per_video.append(segments)
                video_durations_per_video.append(video_duration)

                video_mm_data = {
                    "videos": [(video_array, metadata)],
                }
                if "num_frames" in video_mm_kwargs and "fps" not in video_mm_kwargs:
                    video_mm_kwargs["fps"] = None

                video_outputs = super()._call_hf_processor(
                    prompt="<|vision_start|><|video_pad|><|vision_end|>",
                    mm_data=video_mm_data,
                    mm_kwargs=video_mm_kwargs,
                    tok_kwargs=tok_kwargs,
                )

                merge_size = processor.video_processor.merge_size
                video_grid_thw = video_outputs["video_grid_thw"]
                num_frames = int(video_grid_thw[0, 0])
                tokens_per_frame_base = int(video_grid_thw[0, 1:].prod()) // (merge_size**2)

                video_pruning_rate = self.info.ctx.get_mm_config().video_pruning_rate
                if video_pruning_rate is not None and video_pruning_rate > 0.0:
                    num_tokens = compute_retained_tokens_count(
                        tokens_per_frame=tokens_per_frame_base,
                        num_frames=num_frames,
                        q=video_pruning_rate,
                    )
                    tokens_per_frame = [num_tokens] + [0] * (num_frames - 1)
                    select_token_id = False
                else:
                    tokens_per_frame = [tokens_per_frame_base] * num_frames
                    select_token_id = False

                video_repl = self.get_video_repl(
                    tokens_per_frame=tokens_per_frame,
                    time_segments=segments,
                    timestamp_token_id=self.info.get_hf_config().timestamp_token_id,
                    vision_start_token_id=self.info.get_hf_config().vision_start_token_id,
                    vision_end_token_id=self.info.get_hf_config().vision_end_token_id,
                    video_token_id=self.info.get_hf_config().video_token_id,
                    select_token_id=select_token_id,
                )
                video_outputs.pop("input_ids")
                video_placeholder = processor.tokenizer.batch_decode(
                    [video_repl.full],
                    skip_special_tokens=False,
                )[0]
                prompt = prompt.replace(
                    "<|vision_start|><|video_pad|><|vision_end|>",
                    video_placeholder,
                    1,
                )

                video_grid_thw_lst.append(video_outputs["video_grid_thw"])
                pixel_values_videos_lst.append(video_outputs["pixel_values_videos"])

            from transformers.feature_extraction_utils import BatchFeature

            video_outputs = dict(
                pixel_values_videos=torch.cat(pixel_values_videos_lst),
                video_grid_thw=torch.cat(video_grid_thw_lst),
                timestamps=time_segments_per_video,
                video_durations=video_durations_per_video,
            )
        else:
            video_outputs = dict()

        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=processor_mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        from transformers.feature_extraction_utils import BatchFeature

        return BatchFeature(dict(processed_outputs, **video_outputs))

    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        fields = dict(
            _create_qwen2vl_field_factory(
                self.info.get_hf_config().vision_config.spatial_merge_size
            )(hf_inputs)
        )
        # vLLM drops processor outputs that are not declared here. Keep the
        # source duration on CPU so the full-forward codec can scale its span.
        fields["video_durations"] = MultiModalFieldConfig.batched(
            "video", keep_on_cpu=True
        )
        return fields

    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ):
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        hf_config = self.info.get_hf_config()

        merge_length = image_processor.merge_size**2

        def get_image_replacement_qwen3vl(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            num_tokens = int(grid_thw.prod()) // merge_length
            return [hf_processor.image_token_id] * num_tokens

        def get_video_replacement_qwen3vl(item_idx: int):
            out_item = out_mm_kwargs["video"][item_idx]
            grid_thw = out_item["video_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            time_segments = out_item["timestamps"].data
            assert len(time_segments) == grid_thw[0], (
                f"The time segment length({len(time_segments)}) should be equal "
                f"video length ({grid_thw[0]})."
            )

            num_frames = int(grid_thw[0])
            tokens_per_frame_base = int(grid_thw[1:].prod()) // merge_length
            video_pruning_rate = self.info.ctx.get_mm_config().video_pruning_rate
            if video_pruning_rate is not None and video_pruning_rate > 0.0:
                num_tokens = compute_retained_tokens_count(
                    tokens_per_frame=tokens_per_frame_base,
                    num_frames=num_frames,
                    q=video_pruning_rate,
                )
                tokens_per_frame = [num_tokens] + [0] * (num_frames - 1)
                select_token_id = False
            else:
                tokens_per_frame = [tokens_per_frame_base] * num_frames
                select_token_id = False

            return self.get_video_repl(
                tokens_per_frame=tokens_per_frame,
                time_segments=time_segments,
                timestamp_token_id=hf_config.timestamp_token_id,
                vision_start_token_id=hf_config.vision_start_token_id,
                vision_end_token_id=hf_config.vision_end_token_id,
                video_token_id=hf_config.video_token_id,
                select_token_id=select_token_id,
            )

        return [
            PromptReplacement(
                modality="image",
                target=hf_processor.image_token,
                replacement=get_image_replacement_qwen3vl,
            ),
            PromptReplacement(
                modality="video",
                target="<|vision_start|><|video_pad|><|vision_end|>",
                replacement=get_video_replacement_qwen3vl,
            ),
        ]

    @staticmethod
    def get_video_repl(
        *,
        tokens_per_frame: list[int],
        time_segments: list[tuple[float, float]],
        timestamp_token_id: int,
        vision_start_token_id: int,
        vision_end_token_id: int,
        video_token_id: int,
        select_token_id: bool = False,
    ) -> PromptUpdateDetails[list[int]]:
        assert len(time_segments) == len(tokens_per_frame), (
            "time_segments and tokens_per_frame must have the same length"
        )

        all_token_ids: list[int] = []
        for num_tokens in tokens_per_frame:
            all_token_ids.append(timestamp_token_id)
            all_token_ids.append(vision_start_token_id)
            all_token_ids.extend([video_token_id] * num_tokens)
            all_token_ids.append(vision_end_token_id)

        if select_token_id:
            return PromptUpdateDetails.select_token_id(all_token_ids, video_token_id)
        return PromptUpdateDetails.from_seq(all_token_ids)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLCISCodecMultiModalProcessor,
    info=Qwen3VLCISCodecProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3VLForConditionalGenerationWithCISCodec(Qwen3VLForConditionalGeneration):
    config_class = Qwen3VLCISCodecConfig

    @staticmethod
    def _iter_module_tensors(module: nn.Module | None):
        if module is None:
            return
        yield from module.parameters()
        yield from module.buffers()

    @classmethod
    def _get_module_dtype(
        cls,
        module: nn.Module | None,
        fallback: torch.dtype = torch.float32,
    ) -> torch.dtype:
        if module is None:
            return fallback

        for tensor in cls._iter_module_tensors(module):
            return tensor.dtype
        return fallback

    @classmethod
    def _get_module_compute_dtype(
        cls,
        module: nn.Module | None,
        fallback: torch.dtype = torch.float32,
    ) -> torch.dtype:
        if module is None:
            return fallback

        for min_ndim in (2, 1, 0):
            for tensor in cls._iter_module_tensors(module):
                if not tensor.is_floating_point():
                    continue
                if tensor.ndim >= min_ndim:
                    return tensor.dtype
        return fallback

    @classmethod
    def _describe_module_dtypes(
        cls,
        module: nn.Module | None,
        fallback: torch.dtype = torch.float32,
    ) -> str:
        if module is None:
            return str(fallback)

        summary: list[str] = []
        for tensor in cls._iter_module_tensors(module):
            if not tensor.is_floating_point():
                continue
            key = f"{str(tensor.dtype).replace('torch.', '')}/ndim={tensor.ndim}"
            if key not in summary:
                summary.append(key)
        if not summary:
            return str(fallback)
        return ",".join(summary)

    def __init__(self, *, vllm_config, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        self.use_cis_codec = bool(getattr(config, "use_cis_codec", True))
        self.timestamp_token_id = int(getattr(config, "timestamp_token_id", 151669))
        self.timespan_token_id = int(getattr(config, "timespan_token_id", 151670))
        self.use_cis_interface_adapter = bool(
            getattr(config, "use_cis_interface_adapter", False)
        )
        self.default_video_duration_sec = float(
            getattr(config, "default_video_duration_sec", 1.0)
        )

        if self.use_cis_codec:
            cis_interface_adapter_cls, cis_codec_cls = _get_cis_codec_classes()
            self.cis_codec = cis_codec_cls(config.cis_codec_config)
            if self.use_cis_interface_adapter:
                self.cis_interface_adapter = cis_interface_adapter_cls(
                    int(config.text_config.hidden_size),
                    config=getattr(config, "cis_interface_adapter", None),
                )
            self._cis_runtime_device: torch.device | None = None

    def _ensure_cis_runtime_device(self, device: torch.device | str) -> None:
        if not self.use_cis_codec:
            return

        target_device = torch.device(device)
        if self._cis_runtime_device == target_device:
            return

        self.cis_codec.to(device=target_device)
        if hasattr(self, "cis_interface_adapter"):
            self.cis_interface_adapter.to(device=target_device)
        self._cis_runtime_device = target_device

    def decode_cis_hidden_states(
        self,
        hidden_states: torch.Tensor,
        video_duration_sec: float | list[float] | torch.Tensor,
    ) -> list[tuple[float, float]]:
        if hidden_states.ndim == 1:
            hidden_states = hidden_states.unsqueeze(0)

        self._ensure_cis_runtime_device(hidden_states.device)
        decoded = hidden_states
        if self.use_cis_interface_adapter and hasattr(self, "cis_interface_adapter"):
            adapter_dtype = self._get_module_compute_dtype(
                getattr(self, "cis_interface_adapter", None),
                fallback=decoded.dtype,
            )
            try:
                decoded = self.cis_interface_adapter.forward_output(
                    decoded.to(dtype=adapter_dtype)
                ).adapted
                _append_debug(
                    {
                        "event": "cis_adapter_forward_output",
                        "hidden_shape": tuple(hidden_states.shape),
                        "hidden_dtype": str(hidden_states.dtype),
                        "adapter_compute_dtype": str(adapter_dtype),
                        "adapter_dtype_summary": self._describe_module_dtypes(
                            getattr(self, "cis_interface_adapter", None),
                            fallback=adapter_dtype,
                        ),
                        "decoded_shape": tuple(decoded.shape),
                        "decoded_dtype": str(decoded.dtype),
                    }
                )
            except Exception as exc:
                _append_debug(
                    {
                        "event": "cis_adapter_forward_output_error",
                        "hidden_shape": tuple(hidden_states.shape),
                        "hidden_dtype": str(hidden_states.dtype),
                        "adapter_compute_dtype": str(adapter_dtype),
                        "adapter_dtype_summary": self._describe_module_dtypes(
                            getattr(self, "cis_interface_adapter", None),
                            fallback=adapter_dtype,
                        ),
                        "error": repr(exc),
                    }
                )
                raise

        codec_dtype = self._get_module_compute_dtype(
            getattr(self, "cis_codec", None),
            fallback=decoded.dtype,
        )
        decoded = decoded.to(dtype=codec_dtype)

        start_times, end_times = self.cis_codec.decode(
            decoded,
            video_duration_sec=video_duration_sec,
            hard=True,
        )
        if torch.is_tensor(start_times):
            start_list = start_times.detach().cpu().tolist()
        else:
            start_list = list(start_times)
        if torch.is_tensor(end_times):
            end_list = end_times.detach().cpu().tolist()
        else:
            end_list = list(end_times)
        return [(float(start), float(end)) for start, end in zip(start_list, end_list)]

    def embed_multimodal(self, **kwargs: object):
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return None

        multimodal_embeddings: list[torch.Tensor] = []
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                image_embeddings = self._process_image_input(multimodal_input)
                if self.is_multimodal_pruning_enabled:
                    image_embeddings = self._postprocess_image_embeds_evs(
                        image_embeddings, multimodal_input
                    )
                multimodal_embeddings.extend(image_embeddings)
            elif modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                video_embeddings = self._postprocess_video_embeds_evs(
                    video_embeddings, multimodal_input
                )
                multimodal_embeddings.extend(video_embeddings)

        return tuple(multimodal_embeddings)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if multimodal_embeddings is not None and is_multimodal is not None:
            if torch.is_tensor(multimodal_embeddings):
                actual_tokens = int(multimodal_embeddings.shape[0])
            else:
                actual_tokens = sum(int(emb.shape[0]) for emb in multimodal_embeddings)
            expected_tokens = int(is_multimodal.sum().item())
            if actual_tokens != expected_tokens:
                raise ValueError(
                    "qwen_cis_codec embed_input_ids mismatch: "
                    f"actual_mm_tokens={actual_tokens} "
                    f"expected_mm_slots={expected_tokens} "
                    f"is_multimodal_shape={tuple(is_multimodal.shape)}"
                )
        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def _get_expanded_positions(
        self,
        device,
        seq_len,
        video_grid_thw,
        num_tokens_per_frame,
        timestamps,
        is_video_embed,
        is_vision_start,
        retention_mask,
    ):
        embed_token_id = _cached_tensor(self.config.video_token_id, device=device)
        expanded_positions = torch.zeros(seq_len, 5, device=device, dtype=torch.long)
        _, h, w = video_grid_thw
        merge_size = self.visual.spatial_merge_size
        num_frames = len(num_tokens_per_frame)
        unpruned_token_ids = Qwen3VLCISCodecMultiModalProcessor.get_video_repl(
            tokens_per_frame=[(h // merge_size) * (w // merge_size)] * num_frames,
            time_segments=timestamps,
            timestamp_token_id=self.timestamp_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
            vision_end_token_id=self.config.vision_end_token_id,
            video_token_id=self.config.video_token_id,
        ).full
        unpruned_token_ids_tensor = torch.tensor(unpruned_token_ids, device=device)
        mm_feature = MultiModalFeatureSpec(
            data=MultiModalKwargsItem(
                {
                    "video_grid_thw": MultiModalFieldElem(
                        data=torch.tensor(video_grid_thw),
                        field=None,
                    ),
                }
            ),
            modality="video",
            identifier="DUMMY",
            mm_position=PlaceholderRange(offset=0, length=len(unpruned_token_ids)),
        )
        original_mrope = (
            self.get_mrope_input_positions(
                input_tokens=unpruned_token_ids,
                mm_features=[mm_feature],
            )[0]
            .to(device)
            .permute(1, 0)
        )
        full_is_video_embed = unpruned_token_ids_tensor == embed_token_id
        expanded_positions[is_video_embed, :3] = original_mrope[full_is_video_embed][
            retention_mask
        ]
        expanded_positions[~is_video_embed, :3] = original_mrope[~full_is_video_embed]
        expanded_positions[..., 3] = is_vision_start
        expanded_positions[..., 4] = is_video_embed
        return expanded_positions

    def _create_final_video_embeddings(
        self,
        video_embeddings: torch.Tensor,
        num_tokens_per_frame: list[int],
        timestamps: list[tuple[float, float]],
        video_grid_thw: list[int],
        retention_mask: torch.Tensor | None,
        video_duration_sec: float | None = None,
    ) -> torch.Tensor:
        device = video_embeddings.device
        video_repl = Qwen3VLCISCodecMultiModalProcessor.get_video_repl(
            tokens_per_frame=num_tokens_per_frame,
            time_segments=timestamps,
            timestamp_token_id=self.timestamp_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
            vision_end_token_id=self.config.vision_end_token_id,
            video_token_id=self.config.video_token_id,
            select_token_id=self.is_multimodal_pruning_enabled,
        )

        repl_token_ids = torch.tensor(video_repl.full, device=device)
        embed_token_id = _cached_tensor(self.config.video_token_id, device=device)
        is_video_embed = torch.isin(repl_token_ids, embed_token_id)
        num_video_slots = int(is_video_embed.sum().item())
        if int(video_embeddings.shape[0]) != num_video_slots:
            raise ValueError(
                "qwen_cis_codec video embedding count mismatch: "
                f"visual_tokens={int(video_embeddings.shape[0])} "
                f"video_slots={num_video_slots} "
                f"placeholder_len={int(repl_token_ids.shape[0])} "
                f"num_frames={len(num_tokens_per_frame)} "
                f"tokens_per_frame={num_tokens_per_frame[:4]}"
            )
        text_embeddings = self.get_language_model().embed_input_ids(repl_token_ids)

        timestamp_token_id = _cached_tensor(self.timestamp_token_id, device=device)
        is_timestamp = torch.isin(repl_token_ids, timestamp_token_id)
        if self.use_cis_codec and is_timestamp.any():
            self._ensure_cis_runtime_device(device)
            duration_value = (
                float(video_duration_sec)
                if video_duration_sec is not None
                else max(float(timestamps[-1][1]), self.default_video_duration_sec)
            )
            durations = torch.full(
                (len(timestamps),),
                duration_value,
                device=device,
                dtype=torch.float32,
            )
            start_times = torch.tensor(
                [segment[0] for segment in timestamps],
                device=device,
                dtype=torch.float32,
            )
            end_times = torch.tensor(
                [segment[1] for segment in timestamps],
                device=device,
                dtype=torch.float32,
            )
            timestamp_embeddings = self.cis_codec.encode(
                start_times,
                end_times,
                video_duration_sec=durations,
            )
            if self.use_cis_interface_adapter and hasattr(self, "cis_interface_adapter"):
                adapter_dtype = self._get_module_compute_dtype(
                    getattr(self, "cis_interface_adapter", None),
                    fallback=timestamp_embeddings.dtype,
                )
                anchor_embedding = text_embeddings[is_timestamp].detach().to(
                    dtype=adapter_dtype
                )
                try:
                    timestamp_embeddings = self.cis_interface_adapter.forward_input(
                        timestamp_embeddings.to(dtype=adapter_dtype),
                        anchor_embedding=anchor_embedding,
                    ).adapted
                    _append_debug(
                        {
                            "event": "cis_adapter_forward_input",
                            "timestamp_shape": tuple(timestamp_embeddings.shape),
                            "timestamp_dtype": str(timestamp_embeddings.dtype),
                            "anchor_shape": tuple(anchor_embedding.shape),
                            "anchor_dtype": str(anchor_embedding.dtype),
                            "adapter_compute_dtype": str(adapter_dtype),
                            "adapter_dtype_summary": self._describe_module_dtypes(
                                getattr(self, "cis_interface_adapter", None),
                                fallback=adapter_dtype,
                            ),
                        }
                    )
                except Exception as exc:
                    _append_debug(
                        {
                            "event": "cis_adapter_forward_input_error",
                            "timestamp_shape": tuple(timestamp_embeddings.shape),
                            "timestamp_dtype": str(timestamp_embeddings.dtype),
                            "anchor_shape": tuple(anchor_embedding.shape),
                            "anchor_dtype": str(anchor_embedding.dtype),
                            "adapter_compute_dtype": str(adapter_dtype),
                            "adapter_dtype_summary": self._describe_module_dtypes(
                                getattr(self, "cis_interface_adapter", None),
                                fallback=adapter_dtype,
                            ),
                            "error": repr(exc),
                        }
                    )
                    raise
            timestamp_embeddings = timestamp_embeddings.to(dtype=text_embeddings.dtype)
            text_embeddings = text_embeddings.clone()
            text_embeddings[is_timestamp] = timestamp_embeddings

        if self.use_deepstack:
            deepstack_input_embeds, multimodal_embeddings = self._compute_deepstack_embeds(
                inputs_embeds=text_embeddings,
                multimodal_embeddings=[video_embeddings],
                is_multimodal=is_video_embed,
            )
        else:
            deepstack_input_embeds = None
            multimodal_embeddings = [video_embeddings]

        merged_embeddings = _merge_multimodal_embeddings(
            inputs_embeds=text_embeddings,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_video_embed,
        )

        to_concat = [merged_embeddings]
        if deepstack_input_embeds is not None:
            to_concat.append(
                deepstack_input_embeds.permute(1, 0, 2).reshape(
                    deepstack_input_embeds.shape[1], -1
                )
            )

        if self.is_multimodal_pruning_enabled:
            is_vision_start = repl_token_ids.eq(self.config.vision_start_token_id)
            expanded_positions = self._get_expanded_positions(
                device=merged_embeddings.device,
                seq_len=merged_embeddings.shape[0],
                video_grid_thw=video_grid_thw,
                num_tokens_per_frame=num_tokens_per_frame,
                timestamps=timestamps,
                is_video_embed=is_video_embed,
                is_vision_start=is_vision_start,
                retention_mask=retention_mask,
            )
            to_concat.append(expanded_positions)

        return torch.cat(to_concat, dim=-1)

    def load_weights(self, weights) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        tracked_prefixes = ("cis_codec.", "cis_interface_adapter.")
        expected = {
            name
            for name, _ in self.named_parameters(remove_duplicate=False)
            if name.startswith(tracked_prefixes)
        }
        expected.update(
            name
            for name, _ in self.named_buffers(remove_duplicate=False)
            if name.startswith(tracked_prefixes)
        )
        loaded_tracked = {name for name in loaded if name.startswith(tracked_prefixes)}
        missing = sorted(expected - loaded_tracked)

        LOGGER.info(
            "qwen_cis_codec weight load summary: tracked_loaded=%s tracked_expected=%s "
            "missing=%s cis_codec_dtype=%s cis_codec_dtypes=%s "
            "cis_adapter_dtype=%s cis_adapter_dtypes=%s",
            len(loaded_tracked),
            len(expected),
            len(missing),
            self._get_module_compute_dtype(getattr(self, "cis_codec", None)),
            self._describe_module_dtypes(getattr(self, "cis_codec", None)),
            (
                self._get_module_compute_dtype(getattr(self, "cis_interface_adapter", None))
                if hasattr(self, "cis_interface_adapter")
                else None
            ),
            (
                self._describe_module_dtypes(getattr(self, "cis_interface_adapter", None))
                if hasattr(self, "cis_interface_adapter")
                else None
            ),
        )
        if missing:
            LOGGER.warning("qwen_cis_codec missing tracked weights: %s", missing[:10])
        return loaded


def _flatten_numeric_values(value: Any) -> list[float]:
    if value is None:
        return []
    if torch.is_tensor(value):
        return [float(v) for v in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numeric_values(item))
        return flattened
    return []


def _unwrap_duration_source(value: Any, *, max_depth: int = 8) -> Any:
    current = value
    seen: set[int] = set()
    for _ in range(max_depth):
        if current is None:
            return None
        current_id = id(current)
        if current_id in seen:
            break
        seen.add(current_id)

        nested = getattr(current, "data", None)
        if nested is None or nested is current:
            break
        current = nested
    return current


def _extract_duration_from_mapping(data: Mapping[str, Any]) -> list[float]:
    values: list[float] = []
    for key in ("video_durations", "video_duration", "video_duration_sec", "duration", "durations"):
        if key not in data:
            continue
        values.extend(
            float(v)
            for v in _flatten_numeric_values(_unwrap_duration_source(data.get(key)))
            if v > 0.0
        )
    return values


def _extract_duration_candidates(obj: Any, seen: set[int]) -> list[float]:
    if obj is None:
        return []
    if torch.is_tensor(obj):
        return []
    if isinstance(obj, (int, float)):
        return []
    if isinstance(obj, (str, bytes, bytearray)):
        return []

    obj_id = id(obj)
    if obj_id in seen:
        return []
    seen.add(obj_id)

    if isinstance(obj, Mapping):
        results = _extract_duration_from_mapping(obj)
        for value in obj.values():
            if isinstance(value, Mapping):
                results.extend(_extract_duration_candidates(value, seen))
                continue

            nested = getattr(value, "data", None)
            if nested is not None and nested is not value:
                results.extend(_extract_duration_candidates(nested, seen))
        return results

    data = getattr(obj, "data", None)
    if data is not None and data is not obj:
        results = _extract_duration_candidates(data, seen)
        if results:
            return results

    if isinstance(obj, (list, tuple)):
        results: list[float] = []
        for item in obj:
            if isinstance(item, Mapping):
                results.extend(_extract_duration_candidates(item, seen))
                continue

            nested = getattr(item, "data", None)
            if nested is not None and nested is not item:
                results.extend(_extract_duration_candidates(nested, seen))
        return results

    return []


def _select_required_video_duration(values: list[float]) -> float:
    positives = [float(value) for value in values if float(value) > 0.0]
    if not positives:
        raise ValueError(
            "Missing required positive video_duration for qwen_cis_codec "
            "full-forward decode. Pass real video duration through "
            "video_durations/video_duration/video_duration_sec/duration metadata."
        )

    reference = positives[0]
    tolerance = max(1e-5, abs(reference) * 1e-5)
    for candidate in positives[1:]:
        if abs(candidate - reference) > tolerance:
            raise ValueError(
                "Multiple inconsistent positive video_duration values for "
                "qwen_cis_codec full-forward decode: "
                f"{positives[:8]}"
            )
    return reference


def _infer_request_video_duration(req_state: Any) -> float:
    mm_features = getattr(req_state, "mm_features", None) or []
    seen: set[int] = set()
    values: list[float] = []
    for mm_feature in mm_features:
        values.extend(_extract_duration_candidates(mm_feature, seen))
    return _select_required_video_duration(values)


def _get_qwen_cis_codec_prediction_cache(
    model_runner: GPUModelRunner,
) -> dict[str, list[tuple[float, float]]]:
    cache = getattr(model_runner, "_qwen_cis_codec_prediction_cache", None)
    if cache is None:
        cache = {}
        model_runner._qwen_cis_codec_prediction_cache = cache
    return cache


def _prune_qwen_cis_codec_prediction_cache(model_runner: GPUModelRunner) -> None:
    cache = getattr(model_runner, "_qwen_cis_codec_prediction_cache", None)
    if not cache:
        return
    active_req_ids = set(getattr(model_runner, "requests", {}).keys())
    stale_req_ids = [req_id for req_id in cache if req_id not in active_req_ids]
    for req_id in stale_req_ids:
        cache.pop(req_id, None)


def _get_qwen_cis_codec_pending_jobs(
    model_runner: GPUModelRunner,
) -> dict[str, dict[str, Any]]:
    pending = getattr(model_runner, "_qwen_cis_codec_pending_full_forward", None)
    if pending is None:
        pending = {}
        model_runner._qwen_cis_codec_pending_full_forward = pending
    return pending


def _get_timespan_token_id(model: Any) -> int | None:
    timespan_token_id = getattr(model, "timespan_token_id", None)
    if timespan_token_id is None:
        timespan_token_id = getattr(getattr(model, "config", None), "timespan_token_id", None)
    if timespan_token_id is None:
        return None
    return int(timespan_token_id)


def _ensure_timespan_not_terminal(
    *,
    req_id: str,
    req_state: Any,
    timespan_token_id: int,
) -> None:
    sampling_params = getattr(req_state, "sampling_params", None)
    if sampling_params is None:
        return

    all_stop_token_ids = {
        int(token_id)
        for token_id in (getattr(sampling_params, "all_stop_token_ids", set()) or set())
    }
    stop_token_ids = getattr(sampling_params, "stop_token_ids", None) or []
    all_stop_token_ids.update(int(token_id) for token_id in stop_token_ids)
    if int(timespan_token_id) in all_stop_token_ids:
        raise RuntimeError(
            "qwen_cis_codec strict full-forward mode requires <|TIMESPAN|> not to be a "
            f"terminal token, but request {req_id} treats token_id={timespan_token_id} as stop."
        )


def _register_pending_cis_codec_full_forward(
    *,
    model_runner: GPUModelRunner,
    req_id: str,
    req_state: Any,
    full_token_ids: list[int],
    output_token_ids: list[int],
    timespan_positions: list[int],
    timespan_token_id: int,
    request_video_duration: float,
) -> None:
    _ensure_timespan_not_terminal(
        req_id=req_id,
        req_state=req_state,
        timespan_token_id=timespan_token_id,
    )
    pending_jobs = _get_qwen_cis_codec_pending_jobs(model_runner)
    pending_jobs[req_id] = {
        "full_token_ids": list(full_token_ids),
        "output_token_ids": list(output_token_ids),
        "timespan_positions": list(timespan_positions),
        "required_block_counts": compute_required_block_counts(
            model_runner=model_runner,
            seq_len=len(full_token_ids),
        ),
        "request_video_duration": float(request_video_duration),
    }
    _append_debug(
        {
            "stage": "full_forward_pending_registered",
            "req_id": req_id,
            "full_len": len(full_token_ids),
            "timespan_positions": list(timespan_positions),
            "request_video_duration": float(request_video_duration),
        }
    )


def _validate_pending_cis_codec_job(
    *,
    req_id: str,
    pending_job: Mapping[str, Any],
    current_output_token_ids: list[int],
) -> None:
    snapshot_output_token_ids = [
        int(token_id) for token_id in pending_job.get("output_token_ids", [])
    ]
    current_prefix = current_output_token_ids[: len(snapshot_output_token_ids)]
    if current_prefix != snapshot_output_token_ids:
        raise RuntimeError(
            "qwen_cis_codec pending full-forward snapshot diverged before post-commit flush "
            f"for request {req_id}: expected_prefix={snapshot_output_token_ids[-8:]} "
            f"current_prefix={current_prefix[-8:]}"
        )


def _pending_job_blocks_ready(
    *,
    req_state: Any,
    pending_job: Mapping[str, Any],
) -> bool:
    required_block_counts = dict(pending_job.get("required_block_counts", {}) or {})
    for kv_cache_gid, required_count in required_block_counts.items():
        if len(req_state.block_ids[kv_cache_gid]) < int(required_count):
            return False
    return True


def _run_cis_codec_full_forward(
    *,
    model_runner: GPUModelRunner,
    req_state: Any,
    full_token_ids: list[int],
    timespan_positions: list[int],
    request_video_duration: float,
) -> list[tuple[float, float]]:
    model = model_runner.get_model()
    seq_len = len(full_token_ids)
    if seq_len <= 0:
        return []

    mm_embeds, is_mm_embed = collect_full_mm_embeddings(
        model_runner=model_runner,
        req_state=req_state,
        seq_len=seq_len,
    )
    mm_embeds, positions = build_full_forward_positions(
        model_runner=model_runner,
        model=model,
        req_state=req_state,
        full_token_ids=full_token_ids,
        mm_embeds=mm_embeds,
    )
    attn_metadata, slot_mapping = build_full_forward_attention(
        model_runner=model_runner,
        req_state=req_state,
        seq_len=seq_len,
    )

    input_ids = torch.tensor(
        full_token_ids,
        dtype=torch.int32,
        device=model_runner.device,
    )
    inputs_embeds = model.embed_input_ids(
        input_ids=input_ids,
        multimodal_embeddings=mm_embeds,
        is_multimodal=is_mm_embed,
    )
    model_input_ids = input_ids if getattr(model, "requires_raw_input_tokens", False) else None

    with torch.inference_mode(), set_forward_context(
        attn_metadata,
        model_runner.vllm_config,
        num_tokens=seq_len,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        slot_mapping=slot_mapping,
    ):
        hidden_states = model(
            input_ids=model_input_ids,
            positions=positions,
            intermediate_tensors=None,
            inputs_embeds=inputs_embeds,
        )

    time_hidden = hidden_states[timespan_positions]
    durations = torch.full(
        (len(timespan_positions),),
        float(request_video_duration),
        device=time_hidden.device,
        dtype=torch.float32,
    )
    return model.decode_cis_hidden_states(time_hidden, video_duration_sec=durations)


def _debug_decode_from_current_step_hidden(
    *,
    model_runner: GPUModelRunner,
    req_ids: list[str],
    hidden_states: Any,
    real_output_token_ids_by_req: Mapping[str, list[int]] | None,
) -> None:
    if hidden_states is None:
        return
    model = model_runner.get_model()
    if not hasattr(model, "decode_cis_hidden_states"):
        return

    timespan_token_id = _get_timespan_token_id(model)
    if timespan_token_id is None:
        return
    if not torch.is_tensor(hidden_states):
        return
    if hidden_states.ndim != 2:
        return
    if hidden_states.shape[0] != len(req_ids):
        return

    for req_idx, req_id in enumerate(req_ids):
        output_ids = list((real_output_token_ids_by_req or {}).get(req_id, []))
        if not output_ids or output_ids[-1] != int(timespan_token_id):
            continue

        req_state = model_runner.requests.get(req_id)
        if req_state is None:
            continue
        request_video_duration = _infer_request_video_duration(req_state)
        try:
            predictions = model.decode_cis_hidden_states(
                hidden_states[req_idx],
                video_duration_sec=float(request_video_duration),
            )
        except Exception as exc:
            _append_debug(
                {
                    "stage": "decode_step_hidden_exception",
                    "req_id": req_id,
                    "request_video_duration": float(request_video_duration),
                    "error": repr(exc),
                }
            )
            continue
        _append_debug(
            {
                "stage": "decode_step_hidden_prediction",
                "req_id": req_id,
                "request_video_duration": float(request_video_duration),
                "predictions": predictions,
            }
        )


def _update_real_output_token_cache(
    *,
    model_runner: GPUModelRunner,
    req_ids: list[str],
    step_sampled_token_ids: list[list[int]] | None,
    invalid_req_indices: set[int] | None = None,
) -> dict[str, list[int]]:
    invalid_req_indices = invalid_req_indices or set()
    cache = getattr(model_runner, "_qwen_cis_codec_real_output_ids", None)
    if cache is None:
        cache = {}
        model_runner._qwen_cis_codec_real_output_ids = cache

    active_req_ids = set(getattr(model_runner, "requests", {}).keys())
    stale_req_ids = [req_id for req_id in cache if req_id not in active_req_ids]
    for req_id in stale_req_ids:
        cache.pop(req_id, None)

    for req_idx, req_id in enumerate(req_ids):
        req_state = model_runner.requests.get(req_id)
        if req_state is None:
            continue

        real_output_ids = cache.setdefault(
            req_id,
            [int(token_id) for token_id in req_state.output_token_ids if int(token_id) >= 0],
        )

        target_len = len(req_state.output_token_ids)
        if len(real_output_ids) > target_len:
            del real_output_ids[target_len:]

        if req_idx in invalid_req_indices:
            continue
        if not step_sampled_token_ids or req_idx >= len(step_sampled_token_ids):
            continue

        missing = target_len - len(real_output_ids)
        if missing <= 0:
            continue

        step_ids = step_sampled_token_ids[req_idx][:missing]
        if not step_ids:
            continue
        real_output_ids.extend(int(token_id) for token_id in step_ids)

    return cache


def _attach_cis_codec_step_outputs(
    output: Any,
    *,
    req_ids: list[str],
    model_runner: GPUModelRunner,
    real_output_token_ids_by_req: Mapping[str, list[int]] | None = None,
    invalid_req_indices: set[int] | None = None,
) -> None:
    if output is None:
        return
    model = model_runner.get_model()
    if not hasattr(model, "decode_cis_hidden_states"):
        return

    timespan_token_id = _get_timespan_token_id(model)
    if timespan_token_id is None:
        return
    if get_pp_group().world_size > 1:
        LOGGER.warning(
            "qwen_cis_codec strict full-forward sidecar currently supports PP=1 only."
        )
        return

    invalid_req_indices = invalid_req_indices or set()
    prediction_cache = _get_qwen_cis_codec_prediction_cache(model_runner)
    pending_jobs = _get_qwen_cis_codec_pending_jobs(model_runner)
    _prune_qwen_cis_codec_prediction_cache(model_runner)

    step_payloads: dict[str, dict[str, Any]] = {}
    pending_req_ids: set[str] = set()
    for req_idx, req_id in enumerate(req_ids):
        if req_idx in invalid_req_indices:
            continue

        req_state = model_runner.requests.get(req_id)
        if req_state is None or req_state.prompt_token_ids is None:
            continue

        cached_output_token_ids = None
        if real_output_token_ids_by_req is not None:
            cached_output_token_ids = real_output_token_ids_by_req.get(req_id)
        if cached_output_token_ids is None:
            cached_output_token_ids = [
                int(token_id) for token_id in req_state.output_token_ids if int(token_id) >= 0
            ]

        cached_predictions = prediction_cache.get(req_id)
        if cached_predictions:
            step_payloads[req_id] = {
                "predictions": cached_predictions,
                "token_ids": [int(timespan_token_id)] * len(cached_predictions),
                "mode": "full_forward_cached",
            }
            continue

        prompt_token_ids = list(req_state.prompt_token_ids)
        full_token_ids = prompt_token_ids + list(cached_output_token_ids)
        # Prompt templates may mention <|TIMESPAN|> as an instruction. Only
        # generated occurrences represent predictions that should be decoded.
        timespan_positions = [
            len(prompt_token_ids) + output_pos
            for output_pos, token_id in enumerate(cached_output_token_ids)
            if token_id == int(timespan_token_id)
        ]
        if not timespan_positions:
            _append_debug(
                {
                    "stage": "timespan_not_found",
                    "req_id": req_id,
                    "prompt_len": len(req_state.prompt_token_ids),
                    "output_len": len(cached_output_token_ids),
                    "output_tail": list(cached_output_token_ids[-8:]),
                    "req_state_output_tail": list(req_state.output_token_ids[-8:]),
                    "timespan_token_id": int(timespan_token_id),
                }
            )
            continue

        if req_id in pending_jobs:
            _validate_pending_cis_codec_job(
                req_id=req_id,
                pending_job=pending_jobs[req_id],
                current_output_token_ids=cached_output_token_ids,
            )
        else:
            request_video_duration = _infer_request_video_duration(req_state)
            _register_pending_cis_codec_full_forward(
                model_runner=model_runner,
                req_id=req_id,
                req_state=req_state,
                full_token_ids=full_token_ids,
                output_token_ids=cached_output_token_ids,
                timespan_positions=timespan_positions,
                timespan_token_id=int(timespan_token_id),
                request_video_duration=request_video_duration,
            )
        pending_req_ids.add(req_id)

    target = getattr(output, "_model_runner_output", output)
    if step_payloads:
        target._qwen_cis_codec_step_payloads = step_payloads
        _append_debug(
            {
                "stage": "attach_payloads",
                "req_ids": list(step_payloads.keys()),
            }
        )
    if pending_req_ids:
        target._qwen_cis_codec_pending_req_ids = sorted(pending_req_ids)
        _append_debug(
            {
                "stage": "attach_pending_reqs",
                "req_ids": sorted(pending_req_ids),
            }
        )
    if not step_payloads and not pending_req_ids:
        _append_debug(
            {
                "stage": "no_step_payloads",
                "req_ids": req_ids,
            }
        )


def _flush_pending_cis_codec_full_forwards(model_runner: GPUModelRunner) -> None:
    pending_jobs = _get_qwen_cis_codec_pending_jobs(model_runner)
    if not pending_jobs:
        return

    prediction_cache = _get_qwen_cis_codec_prediction_cache(model_runner)
    real_output_token_ids_by_req = getattr(model_runner, "_qwen_cis_codec_real_output_ids", {}) or {}
    for req_id in list(pending_jobs.keys()):
        req_state = model_runner.requests.get(req_id)
        if req_state is None or req_state.prompt_token_ids is None:
            raise RuntimeError(
                f"Pending qwen_cis_codec full-forward request disappeared before flush: {req_id}"
            )

        pending_job = pending_jobs[req_id]
        snapshot_output_token_ids = [
            int(token_id) for token_id in pending_job.get("output_token_ids", [])
        ]
        current_output_token_ids = [
            int(token_id)
            for token_id in real_output_token_ids_by_req.get(req_id, req_state.output_token_ids)
            if int(token_id) >= 0
        ]
        if len(current_output_token_ids) < len(snapshot_output_token_ids):
            raise RuntimeError(
                "qwen_cis_codec pending full-forward output length regressed before flush "
                f"for request {req_id}: current={len(current_output_token_ids)} "
                f"snapshot={len(snapshot_output_token_ids)}"
            )
        _validate_pending_cis_codec_job(
            req_id=req_id,
            pending_job=pending_job,
            current_output_token_ids=current_output_token_ids,
        )
        if not _pending_job_blocks_ready(req_state=req_state, pending_job=pending_job):
            continue

        full_token_ids = [int(token_id) for token_id in pending_job["full_token_ids"]]
        timespan_positions = [int(pos) for pos in pending_job["timespan_positions"]]
        request_video_duration = float(pending_job["request_video_duration"])
        predictions = _run_cis_codec_full_forward(
            model_runner=model_runner,
            req_state=req_state,
            full_token_ids=full_token_ids,
            timespan_positions=timespan_positions,
            request_video_duration=request_video_duration,
        )
        if not predictions:
            raise RuntimeError(
                f"qwen_cis_codec strict full-forward produced empty predictions for request {req_id}"
            )

        prediction_cache[req_id] = predictions
        pending_jobs.pop(req_id, None)
        _append_debug(
            {
                "stage": "full_forward_success",
                "req_id": req_id,
                "full_len": len(full_token_ids),
                "timespan_positions": timespan_positions,
                "request_video_duration": request_video_duration,
                "predictions": predictions,
            }
        )


def _patch_gpu_model_runner_sample_tokens() -> None:
    original_sample_tokens = GPUModelRunner.sample_tokens
    if getattr(original_sample_tokens, "_qwen_cis_codec_patched", False):
        return

    def wrapped_sample_tokens(self, grammar_output):
        pre_req_ids = list(getattr(self.input_batch, "req_ids", []) or [])
        pre_hidden_states = None
        pre_real_output_token_ids_by_req = dict(
            getattr(self, "_qwen_cis_codec_real_output_ids", {}) or {}
        )
        execute_model_state = getattr(self, "execute_model_state", None)
        if execute_model_state is not None and len(execute_model_state) >= 5:
            pre_hidden_states = execute_model_state[4]

        output = original_sample_tokens(self, grammar_output)
        if not getattr(self, "requests", None):
            return output

        invalid_req_indices = set(getattr(output, "_invalid_req_indices", []) or [])
        step_sampled_token_ids = extract_real_step_sampled_token_ids(output)
        real_output_token_ids_by_req = _update_real_output_token_cache(
            model_runner=self,
            req_ids=list(self.input_batch.req_ids),
            step_sampled_token_ids=step_sampled_token_ids,
            invalid_req_indices=invalid_req_indices,
        )
        _append_debug(
            {
                "stage": "real_step_ids",
                "req_ids": list(self.input_batch.req_ids),
                "step_sampled_token_ids": step_sampled_token_ids,
                "cache_lens": {
                    req_id: len(real_output_token_ids_by_req.get(req_id, []))
                    for req_id in self.input_batch.req_ids
                },
            }
        )
        _debug_decode_from_current_step_hidden(
            model_runner=self,
            req_ids=pre_req_ids,
            hidden_states=pre_hidden_states,
            real_output_token_ids_by_req=pre_real_output_token_ids_by_req,
        )
        _attach_cis_codec_step_outputs(
            output,
            req_ids=list(self.input_batch.req_ids),
            model_runner=self,
            real_output_token_ids_by_req=real_output_token_ids_by_req,
            invalid_req_indices=invalid_req_indices,
        )
        return output

    wrapped_sample_tokens._qwen_cis_codec_patched = True  # type: ignore[attr-defined]
    GPUModelRunner.sample_tokens = wrapped_sample_tokens


def _patch_gpu_model_runner_update_states() -> None:
    original_update_states = GPUModelRunner._update_states
    if getattr(original_update_states, "_qwen_cis_codec_patched", False):
        return

    def wrapped_update_states(self, scheduler_output):
        deferred_fn = original_update_states(self, scheduler_output)
        pending_jobs = getattr(self, "_qwen_cis_codec_pending_full_forward", None) or {}
        if not pending_jobs:
            return deferred_fn

        def run_deferred_and_flush():
            if deferred_fn is not None:
                deferred_fn()
            _flush_pending_cis_codec_full_forwards(self)

        return run_deferred_and_flush

    wrapped_update_states._qwen_cis_codec_patched = True  # type: ignore[attr-defined]
    GPUModelRunner._update_states = wrapped_update_states


def _patch_scheduler_update_from_output() -> None:
    original_update_from_output = Scheduler.update_from_output
    if getattr(original_update_from_output, "_qwen_cis_codec_patched", False):
        return

    def wrapped_update_from_output(self, scheduler_output, model_runner_output):
        payload_source = getattr(model_runner_output, "_model_runner_output", model_runner_output)
        accumulator = getattr(self, "_qwen_cis_codec_accumulator", None)
        if accumulator is None:
            accumulator = {}
            self._qwen_cis_codec_accumulator = accumulator

        step_payloads = getattr(payload_source, "_qwen_cis_codec_step_payloads", None) or {}
        pending_req_ids = set(getattr(payload_source, "_qwen_cis_codec_pending_req_ids", None) or [])
        for req_id, payload in step_payloads.items():
            accumulator[req_id] = payload
        _append_debug(
            {
                "stage": "scheduler_seen_payloads",
                "req_ids": list(step_payloads.keys()),
            }
        )
        if pending_req_ids:
            _append_debug(
                {
                    "stage": "scheduler_seen_pending_reqs",
                    "req_ids": sorted(pending_req_ids),
                }
            )

        engine_outputs_by_client = original_update_from_output(
            self,
            scheduler_output,
            model_runner_output,
        )

        for engine_outputs in engine_outputs_by_client.values():
            for engine_output in engine_outputs.outputs:
                if engine_output.finish_reason is None:
                    continue
                req_id = engine_output.request_id
                payload = accumulator.pop(req_id, None)
                if payload is None:
                    if req_id in pending_req_ids:
                        raise RuntimeError(
                            "qwen_cis_codec request finished before strict post-commit "
                            "full-forward sidecar resolved. This indicates the request "
                            f"terminated while request_id={req_id} still had an unresolved "
                            "<|TIMESPAN|> pending job."
                        )
                    _append_debug(
                        {
                            "stage": "scheduler_missing_payload_for_finished_req",
                            "req_id": req_id,
                        }
                    )
                    continue
                merged_kv_params = dict(engine_output.kv_transfer_params or {})
                merged_kv_params["qwen_cis_codec"] = payload
                engine_output.kv_transfer_params = merged_kv_params
                _append_debug(
                    {
                        "stage": "scheduler_attached_payload",
                        "req_id": req_id,
                        "predictions": payload.get("predictions"),
                    }
                )

        return engine_outputs_by_client

    wrapped_update_from_output._qwen_cis_codec_patched = True  # type: ignore[attr-defined]
    Scheduler.update_from_output = wrapped_update_from_output


def apply_qwen_cis_codec_vllm_runtime_patch() -> None:
    global _RUNTIME_PATCHED
    if _RUNTIME_PATCHED:
        return

    _CONFIG_REGISTRY["qwen3_vl_cis_codec"] = Qwen3VLCISCodecConfig
    AutoConfig.register("qwen3_vl_cis_codec", Qwen3VLCISCodecConfig, exist_ok=True)
    ModelRegistry.register_model(
        "Qwen3VLForConditionalGenerationWithCISCodec",
        Qwen3VLForConditionalGenerationWithCISCodec,
    )
    _patch_gpu_model_runner_sample_tokens()
    _patch_gpu_model_runner_update_states()
    _patch_scheduler_update_from_output()
    _RUNTIME_PATCHED = True
    LOGGER.info(
        "Registered qwen_cis_codec vLLM runtime patch using local CIS codec package at %s",
        _resolve_cis_codec_dir(),
    )
