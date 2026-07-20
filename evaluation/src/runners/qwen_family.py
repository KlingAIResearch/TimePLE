from __future__ import annotations

import base64
import importlib.util
import json
import logging
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from benchmark_loader import BenchmarkSample
from qwen_video_preprocess import (
    prepare_qwen_video_input,
    prepare_qwen_video_input_with_qwen_vl_utils,
)
from runners.base import BaseTransformersRunner, BaseVLLMRunner, ModelResponse
from videochat_r1_vllm_patch import apply_videochat_r1_qwen25_processor_runtime_patch
from vllm_video_metadata_patch import apply_vllm_video_metadata_runtime_patch


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _to_json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _to_json_compatible(value.item())
        except Exception:
            return value
    return value


def _encode_eval_suite_video_metadata(metadata_items: list[dict[str, Any] | None]) -> str:
    normalized_items = [
        _to_json_compatible(item) if item is not None else None for item in metadata_items
    ]
    return json.dumps(
        normalized_items,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _ensure_qwen_vl_utils_importable() -> None:
    if importlib.util.find_spec("qwen_vl_utils") is not None:
        return

    candidate = PROJECT_ROOT / "qwen-vl-utils" / "src"
    if candidate.exists():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

    if importlib.util.find_spec("qwen_vl_utils") is None:
        raise ModuleNotFoundError(
            "qwen_vl_utils is required for the official VideoChat-R1 transformers runner."
        )


class SingleVideoMessageBuilderMixin:
    def _video_file_uri(self, sample: BenchmarkSample) -> str:
        return Path(sample.video_path).resolve().as_uri()

    def _load_qwen_vision_factors(self) -> tuple[int, int]:
        model_path = Path(str(self.inference_cfg["model_path"])).resolve()
        preprocessor_path = model_path / "preprocessor_config.json"
        config_path = model_path / "config.json"

        if not preprocessor_path.exists():
            raise ValueError(
                f"Missing preprocessor_config.json for qwen-family model at {model_path}"
            )
        if not config_path.exists():
            raise ValueError(f"Missing config.json for qwen-family model at {model_path}")

        preprocessor_cfg = json.loads(preprocessor_path.read_text(encoding="utf-8"))
        model_cfg = json.loads(config_path.read_text(encoding="utf-8"))
        vision_cfg = dict(model_cfg.get("vision_config", {}))

        patch_size = preprocessor_cfg.get("patch_size")
        spatial_merge_size = vision_cfg.get("spatial_merge_size")
        temporal_patch_size = vision_cfg.get("temporal_patch_size")

        if patch_size is None or spatial_merge_size is None or temporal_patch_size is None:
            raise ValueError(
                "Missing qwen vision factors in model configs: "
                f"patch_size={patch_size} spatial_merge_size={spatial_merge_size} "
                f"temporal_patch_size={temporal_patch_size}"
            )

        image_factor = int(patch_size) * int(spatial_merge_size)
        frame_factor = int(temporal_patch_size)
        return image_factor, frame_factor

    def _translate_video_message_kwargs(self) -> dict[str, Any]:
        video_cfg = dict(self.inference_cfg.get("video", {}))
        message_kwargs = dict(video_cfg.get("message_kwargs", {}))
        if not message_kwargs:
            return {}

        translated: dict[str, Any] = {}
        passthrough_keys = (
            "fps",
            "nframes",
            "num_frames",
            "max_frames",
            "min_frames",
            "do_sample_frames",
            "min_pixels",
            "max_pixels",
            "total_pixels",
        )
        for key in passthrough_keys:
            value = message_kwargs.get(key)
            if value is not None:
                translated[key] = value

        frame_token_only = bool(message_kwargs.get("frame_token_only", False))
        frame_min_token = message_kwargs.get("frame_min_token")
        frame_max_token = message_kwargs.get("frame_max_token")

        if frame_token_only and (frame_min_token is not None or frame_max_token is not None):
            image_factor, frame_factor = self._load_qwen_vision_factors()
            token_pixels_factor = image_factor * image_factor * frame_factor

            if "min_pixels" in translated and frame_min_token is not None:
                raise ValueError(
                    "Conflicting qwen video settings: both min_pixels and "
                    "frame_min_token are set"
                )
            if "max_pixels" in translated and frame_max_token is not None:
                raise ValueError(
                    "Conflicting qwen video settings: both max_pixels and "
                    "frame_max_token are set"
                )

            if frame_min_token is not None:
                translated["min_pixels"] = int(frame_min_token) * token_pixels_factor
            if frame_max_token is not None:
                translated["max_pixels"] = int(frame_max_token) * token_pixels_factor

        return translated

    def build_video_item(self, sample: BenchmarkSample) -> dict[str, Any]:
        return {
            "type": "video_url",
            "video_url": {"url": self._video_file_uri(sample)},
        }

    def build_messages(self, sample: BenchmarkSample) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        system_prompt = self.get_system_prompt()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_prompt = self.render_user_prompt(sample)
        text_item = {"type": "text", "text": user_prompt}
        video_item = self.build_video_item(sample)

        content_order = str(
            self.inference_cfg.get("video", {}).get("content_order", "media_first")
        )
        if content_order == "text_first":
            content = [text_item, video_item]
        else:
            content = [video_item, text_item]

        messages.append({"role": "user", "content": content})
        return messages


class BaseSingleVideoRunner(SingleVideoMessageBuilderMixin, BaseVLLMRunner):
    def get_chat_mm_processor_kwargs(self) -> dict[str, Any] | None:
        kwargs = super().get_chat_mm_processor_kwargs() or {}
        kwargs.update(self._translate_video_message_kwargs())
        return kwargs or None


class BaseQwenOfficialVideoRunner(BaseSingleVideoRunner):
    def __init__(self, config: dict[str, Any], *, config_path: Path) -> None:
        super().__init__(config, config_path=config_path)
        self._processor = None

    def build_video_item(self, sample: BenchmarkSample) -> dict[str, Any]:
        return {
            "type": "video",
            "video": str(Path(sample.video_path).resolve()),
        }

    def _get_processor_init_kwargs(self) -> dict[str, Any]:
        processor_kwargs = dict(self.inference_cfg.get("processor_kwargs", {}))
        return processor_kwargs

    def ensure_processor(self):
        if self._processor is None:
            from transformers import AutoProcessor

            processor_kwargs = self._get_processor_init_kwargs()
            self._processor = AutoProcessor.from_pretrained(
                str(self.inference_cfg["model_path"]),
                trust_remote_code=bool(self.inference_cfg.get("trust_remote_code", True)),
                **processor_kwargs,
            )
        return self._processor

    def ensure_llm(self):
        apply_vllm_video_metadata_runtime_patch()
        return super().ensure_llm()

    def should_include_video_metadata(self) -> bool:
        return True

    def use_qwen_vl_utils_video_preprocess(self) -> bool:
        return False

    def _build_generate_input(
        self,
        sample: BenchmarkSample,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        processor = self.ensure_processor()
        prompt_messages = self.build_messages(sample)
        prompt_text = processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=bool(self.engine_cfg.get("add_generation_prompt", True)),
        )

        video_kwargs = self._translate_video_message_kwargs()
        if self.use_qwen_vl_utils_video_preprocess():
            image_patch_size = getattr(getattr(processor, "image_processor", None), "patch_size", None)
            if image_patch_size is None:
                raise ValueError(
                    "Qwen official video preprocessing requires processor.image_processor.patch_size."
                )
            prepared_video = prepare_qwen_video_input_with_qwen_vl_utils(
                sample.video_path,
                image_patch_size=int(image_patch_size),
                video_kwargs=video_kwargs,
                include_video_metadata=self.should_include_video_metadata(),
            )
        else:
            image_factor, frame_factor = self._load_qwen_vision_factors()
            prepared_video = prepare_qwen_video_input(
                sample.video_path,
                image_factor=image_factor,
                frame_factor=frame_factor,
                video_kwargs=video_kwargs,
            )
        mm_processor_kwargs = dict(prepared_video.processor_video_kwargs)
        video_item: Any = prepared_video.video
        if self.should_include_video_metadata():
            video_item = (prepared_video.video, prepared_video.metadata)
            mm_processor_kwargs["eval_suite_video_metadata_json"] = (
                _encode_eval_suite_video_metadata([dict(prepared_video.metadata)])
            )

        generate_input = {
            "prompt": prompt_text,
            "multi_modal_data": {
                "video": [video_item],
            },
            "mm_processor_kwargs": mm_processor_kwargs,
        }
        response_meta = {
            "frame_timestamps": prepared_video.frame_timestamps,
            "sampled_num_frames": prepared_video.sampled_num_frames,
            "resized_height": prepared_video.resized_height,
            "resized_width": prepared_video.resized_width,
            "video_metadata": prepared_video.metadata,
            "video_kwargs": video_kwargs,
            "mm_processor_kwargs": mm_processor_kwargs,
            "video_metadata_injected_to_processor": self.should_include_video_metadata(),
        }
        return generate_input, response_meta

    def _predict_batch_once(self, samples: list[BenchmarkSample]) -> list[ModelResponse]:
        llm = self.ensure_llm()
        sampling_params = self.build_sampling_params()

        vllm_inputs: list[dict[str, Any]] = []
        metas: list[dict[str, Any]] = []
        for sample in samples:
            generate_input, meta = self._build_generate_input(sample)
            vllm_inputs.append(generate_input)
            metas.append(meta)

        outputs = llm.generate(
            vllm_inputs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        responses: list[ModelResponse] = []
        for sample, output, meta in zip(samples, outputs, metas):
            responses.append(
                ModelResponse(
                    sample_id=sample.sample_id,
                    raw_text=self._extract_output_text(output),
                    metadata=meta,
                )
            )
        return responses


class Qwen3VLRunner(BaseQwenOfficialVideoRunner):
    def use_qwen_vl_utils_video_preprocess(self) -> bool:
        return True


class TimeLensRunner(BaseQwenOfficialVideoRunner):
    def use_qwen_vl_utils_video_preprocess(self) -> bool:
        return True


class Qwen25VLLMRunner(BaseQwenOfficialVideoRunner):
    def use_qwen_vl_utils_video_preprocess(self) -> bool:
        return True


class NumProFrameSamplingMixin:
    def _numpro_video_cfg(self) -> dict[str, Any]:
        message_kwargs = dict(self.inference_cfg.get("video", {}).get("message_kwargs", {}))
        if not message_kwargs:
            return {}

        translated: dict[str, Any] = {}
        passthrough_keys = (
            "fps",
            "nframes",
            "num_frames",
            "max_frames",
            "min_frames",
            "do_sample_frames",
            "min_pixels",
            "max_pixels",
            "total_pixels",
        )
        for key in passthrough_keys:
            value = message_kwargs.get(key)
            if value is not None:
                translated[key] = value

        frame_token_only = bool(message_kwargs.get("frame_token_only", False))
        frame_min_token = message_kwargs.get("frame_min_token")
        frame_max_token = message_kwargs.get("frame_max_token")

        if frame_token_only and (frame_min_token is not None or frame_max_token is not None):
            image_factor, frame_factor = self._load_qwen_vision_factors()
            token_pixels_factor = image_factor * image_factor * frame_factor

            if "min_pixels" in translated and frame_min_token is not None:
                raise ValueError(
                    "Conflicting NumPro qwen video settings: both min_pixels and "
                    "frame_min_token are set"
                )
            if "max_pixels" in translated and frame_max_token is not None:
                raise ValueError(
                    "Conflicting NumPro qwen video settings: both max_pixels and "
                    "frame_max_token are set"
                )

            if frame_min_token is not None:
                translated["min_pixels"] = int(frame_min_token) * token_pixels_factor
            if frame_max_token is not None:
                translated["max_pixels"] = int(frame_max_token) * token_pixels_factor

        return translated

    def _numpro_sampling_cfg(self) -> dict[str, Any]:
        return dict(self.inference_cfg.get("video", {}).get("numpro_sampling", {}))

    def _load_qwen_vision_factors(self) -> tuple[int, int]:
        model_path = Path(str(self.inference_cfg["model_path"])).resolve()
        preprocessor_path = model_path / "preprocessor_config.json"
        config_path = model_path / "config.json"

        if not preprocessor_path.exists():
            raise ValueError(
                f"Missing preprocessor_config.json for qwen-family model at {model_path}"
            )
        if not config_path.exists():
            raise ValueError(f"Missing config.json for qwen-family model at {model_path}")

        preprocessor_cfg = json.loads(preprocessor_path.read_text(encoding="utf-8"))
        model_cfg = json.loads(config_path.read_text(encoding="utf-8"))
        vision_cfg = dict(model_cfg.get("vision_config", {}))

        patch_size = preprocessor_cfg.get("patch_size")
        spatial_merge_size = vision_cfg.get("spatial_merge_size")
        temporal_patch_size = vision_cfg.get("temporal_patch_size")

        if patch_size is None or spatial_merge_size is None or temporal_patch_size is None:
            raise ValueError(
                "Missing qwen vision factors in model configs: "
                f"patch_size={patch_size} spatial_merge_size={spatial_merge_size} "
                f"temporal_patch_size={temporal_patch_size}"
            )

        image_factor = int(patch_size) * int(spatial_merge_size)
        frame_factor = int(temporal_patch_size)
        return image_factor, frame_factor

    def _sample_numpro_frames(
        self, sample: BenchmarkSample
    ) -> tuple[list[Any], list[float], dict[str, Any]]:
        try:
            import numpy as np
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as exc:
            raise RuntimeError(
                f"{type(self).__name__} requires numpy and Pillow in the eval environment."
            ) from exc

        sampling_cfg = self._numpro_sampling_cfg()
        marker_fill = tuple(sampling_cfg.get("marker_fill", [255, 255, 255]))
        marker_outline = tuple(sampling_cfg.get("marker_outline", [220, 20, 60]))

        image_factor, _ = self._load_qwen_vision_factors()
        model_path = Path(str(self.inference_cfg["model_path"])).resolve()
        preprocessor_path = model_path / "preprocessor_config.json"
        preprocessor_cfg = json.loads(preprocessor_path.read_text(encoding="utf-8"))
        patch_size = int(preprocessor_cfg.get("patch_size", image_factor))
        prepared = prepare_qwen_video_input_with_qwen_vl_utils(
            sample.video_path,
            image_patch_size=patch_size,
            video_kwargs=self._numpro_video_cfg(),
            include_video_metadata=True,
        )

        if hasattr(prepared.video, "permute"):
            frames_np = prepared.video.permute(0, 2, 3, 1).cpu().numpy()
        else:
            frames_np = np.asarray(prepared.video)
            if frames_np.ndim != 4:
                raise ValueError(
                    "Prepared NumPro video must be a 4D tensor/array in TCHW or THWC format."
                )
            if frames_np.shape[-1] != 3:
                frames_np = np.transpose(frames_np, (0, 2, 3, 1))

        timestamps = list(prepared.frame_timestamps)
        if not timestamps:
            timestamps = [float(idx / 2.0) for idx in range(frames_np.shape[0])]
        duration = float(prepared.metadata.get("duration") or (timestamps[-1] if timestamps else 0))

        font = ImageFont.load_default()
        numbered_frames: list[Any] = []
        for idx, frame in enumerate(frames_np, start=1):
            frame_array = np.asarray(frame)
            if frame_array.dtype != np.uint8:
                if frame_array.max(initial=0) <= 1.0:
                    frame_array = frame_array * 255
                frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)
            image = Image.fromarray(frame_array).convert("RGB")
            draw = ImageDraw.Draw(image)
            label = str(idx)
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            pad = max(6, min(image.size) // 80)
            radius = max(text_w, text_h) // 2 + pad
            cx = pad + radius
            cy = pad + radius
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                fill=marker_outline,
            )
            draw.text(
                (cx - text_w / 2, cy - text_h / 2),
                label,
                fill=marker_fill,
                font=font,
            )
            numbered_frames.append(image)

        return numbered_frames, timestamps, {
            "fps": prepared.metadata.get("fps"),
            "frame_indices": prepared.metadata.get("frames_indices"),
            "frame_timestamps": timestamps,
            "video_duration": duration,
            "sampled_num_frames": prepared.sampled_num_frames,
            "resized_height": prepared.resized_height,
            "resized_width": prepared.resized_width,
            "video_kwargs": self._numpro_video_cfg(),
            "processor_video_kwargs": dict(prepared.processor_video_kwargs),
            "video_metadata": prepared.metadata,
        }

    def _build_numpro_prompt_text(
        self,
        sample: BenchmarkSample,
        timestamps: list[float],
        *,
        duration: float,
    ) -> str:
        lines = [
            f"Frame {frame_idx}, timestamp {timestamp:.2f}s:"
            for frame_idx, timestamp in enumerate(timestamps, start=1)
        ]
        lines.append(
            (
                f"The video duration is {duration:.2f} seconds. "
                "Use the numbered frames and their timestamps to answer the temporal "
                "grounding request below.\n"
                f"{self.render_user_prompt(sample)}"
            )
        )
        return "\n".join(lines)

    def _encode_numbered_frames_as_video_jpeg_uri(self, frames: list[Any]) -> str:
        from PIL import Image

        encoded_frames: list[str] = []
        for frame in frames:
            image = frame if hasattr(frame, "save") else Image.fromarray(frame).convert("RGB")
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=95)
            encoded_frames.append(base64.b64encode(buffer.getvalue()).decode("ascii"))

        return "data:video/jpeg;base64," + ",".join(encoded_frames)

    def _build_numpro_image_messages(
        self,
        sample: BenchmarkSample,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        frames, timestamps, meta = self._sample_numpro_frames(sample)

        content: list[dict[str, Any]] = []
        for frame_idx, (frame, timestamp) in enumerate(zip(frames, timestamps), start=1):
            content.append(
                {
                    "type": "text",
                    "text": f"Frame {frame_idx}, timestamp {timestamp:.2f}s:",
                }
            )
            content.append({"type": "image_pil", "image_pil": frame})

        duration = float(meta["video_duration"])
        content.append(
            {
                "type": "text",
                "text": (
                    f"The video duration is {duration:.2f} seconds. "
                    f"Query: {sample.query}\n"
                    "Using the numbered frames and their timestamps, locate all matching "
                    "time intervals. Return only [[start_time, end_time], ...] in seconds."
                ),
            }
        )

        messages: list[dict[str, Any]] = []
        system_prompt = self.get_system_prompt()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        return messages, {"numpro": meta}

    def _build_numpro_video_messages(
        self,
        sample: BenchmarkSample,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        frames, timestamps, meta = self._sample_numpro_frames(sample)
        duration = float(meta["video_duration"])
        prompt_text = self._build_numpro_prompt_text(
            sample,
            timestamps,
            duration=duration,
        )

        video_item = {
            "type": "video_url",
            "video_url": {"url": self._encode_numbered_frames_as_video_jpeg_uri(frames)},
        }
        text_item = {"type": "text", "text": prompt_text}

        content_order = str(
            self.inference_cfg.get("video", {}).get("content_order", "media_first")
        )
        if content_order == "text_first":
            content = [text_item, video_item]
        else:
            content = [video_item, text_item]

        messages: list[dict[str, Any]] = []
        system_prompt = self.get_system_prompt()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        response_meta = dict(meta)
        response_meta["content_order"] = content_order
        response_meta["encoded_video_format"] = "video/jpeg"
        return messages, {"numpro": response_meta}


class NumProQwen25VLLMRunner(NumProFrameSamplingMixin, BaseVLLMRunner):
    def build_messages(self, sample: BenchmarkSample) -> list[dict[str, Any]]:
        messages, _ = self._build_numpro_video_messages(sample)
        return messages

    def get_chat_mm_processor_kwargs(self) -> dict[str, Any] | None:
        kwargs = super().get_chat_mm_processor_kwargs() or {}
        kwargs.update(dict(self._numpro_video_cfg()))
        kwargs["do_sample_frames"] = False
        kwargs["do_resize"] = False
        return kwargs or None

    def _predict_batch_once(self, samples: list[BenchmarkSample]) -> list[ModelResponse]:
        llm = self.ensure_llm()
        messages_batch: list[list[dict[str, Any]]] = []
        metas: list[dict[str, Any]] = []
        for sample in samples:
            messages, meta = self._build_numpro_video_messages(sample)
            messages_batch.append(messages)
            metas.append(meta)

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
        for sample, output, meta in zip(samples, outputs, metas):
            response_meta = dict(meta)
            response_meta["finish_reason"] = (
                getattr(output.outputs[0], "finish_reason", None)
                if getattr(output, "outputs", None)
                else None
            )
            responses.append(
                ModelResponse(
                    sample_id=sample.sample_id,
                    raw_text=self._extract_output_text(output),
                    metadata=response_meta,
                )
            )
        return responses


class QwenTimeCodecRunner(BaseQwenOfficialVideoRunner):
    def use_qwen_vl_utils_video_preprocess(self) -> bool:
        return True

    def ensure_llm(self):
        from qwen_tcodec_vllm import apply_qwen_tcodec_vllm_runtime_patch

        apply_qwen_tcodec_vllm_runtime_patch()
        return super().ensure_llm()

    def _predict_batch_once(self, samples: list[BenchmarkSample]) -> list[ModelResponse]:
        llm = self.ensure_llm()
        sampling_params = self.build_sampling_params()

        vllm_inputs: list[dict[str, Any]] = []
        metas: list[dict[str, Any]] = []
        for sample in samples:
            generate_input, meta = self._build_generate_input(sample)
            vllm_inputs.append(generate_input)
            metas.append(meta)

        outputs = llm.generate(
            vllm_inputs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        responses: list[ModelResponse] = []
        for sample, output, meta in zip(samples, outputs, metas):
            kv_payload = dict(getattr(output, "kv_transfer_params", {}) or {})
            time_codec_payload = dict(kv_payload.get("qwen_tcodec", {}) or {})
            response_meta = dict(meta)
            if time_codec_payload:
                response_meta["time_codec"] = time_codec_payload
                response_meta["time_codec_predictions"] = time_codec_payload.get("predictions")
            responses.append(
                ModelResponse(
                    sample_id=sample.sample_id,
                    raw_text=self._extract_output_text(output),
                    metadata=response_meta,
                )
            )
        return responses


class QwenCISCodecRunner(BaseQwenOfficialVideoRunner):
    def ensure_llm(self):
        from qwen_cis_codec_vllm import apply_qwen_cis_codec_vllm_runtime_patch

        apply_qwen_cis_codec_vllm_runtime_patch()
        return super().ensure_llm()

    def _predict_batch_once(self, samples: list[BenchmarkSample]) -> list[ModelResponse]:
        llm = self.ensure_llm()
        sampling_params = self.build_sampling_params()

        vllm_inputs: list[dict[str, Any]] = []
        metas: list[dict[str, Any]] = []
        for sample in samples:
            generate_input, meta = self._build_generate_input(sample)
            vllm_inputs.append(generate_input)
            metas.append(meta)

        outputs = llm.generate(
            vllm_inputs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        responses: list[ModelResponse] = []
        for sample, output, meta in zip(samples, outputs, metas):
            kv_payload = dict(getattr(output, "kv_transfer_params", {}) or {})
            cis_codec_payload = dict(kv_payload.get("qwen_cis_codec", {}) or {})
            response_meta = dict(meta)
            if cis_codec_payload:
                response_meta["cis_codec"] = cis_codec_payload
                response_meta["cis_codec_predictions"] = cis_codec_payload.get("predictions")
            responses.append(
                ModelResponse(
                    sample_id=sample.sample_id,
                    raw_text=self._extract_output_text(output),
                    metadata=response_meta,
                )
            )
        return responses


class QwenCISSpanCodecRunner(BaseQwenOfficialVideoRunner):
    def ensure_llm(self):
        from qwen_cis_span_codec_vllm import apply_qwen_cis_span_codec_vllm_runtime_patch

        apply_qwen_cis_span_codec_vllm_runtime_patch()
        return super().ensure_llm()

    def _predict_batch_once(self, samples: list[BenchmarkSample]) -> list[ModelResponse]:
        llm = self.ensure_llm()
        sampling_params = self.build_sampling_params()

        vllm_inputs: list[dict[str, Any]] = []
        metas: list[dict[str, Any]] = []
        for sample in samples:
            generate_input, meta = self._build_generate_input(sample)
            vllm_inputs.append(generate_input)
            metas.append(meta)

        outputs = llm.generate(
            vllm_inputs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        responses: list[ModelResponse] = []
        for sample, output, meta in zip(samples, outputs, metas):
            kv_payload = dict(getattr(output, "kv_transfer_params", {}) or {})
            cis_span_payload = dict(kv_payload.get("qwen_cis_codec", {}) or {})
            response_meta = dict(meta)
            if cis_span_payload:
                response_meta["cis_span_codec"] = cis_span_payload
                response_meta["cis_span_codec_predictions"] = cis_span_payload.get("predictions")
                response_meta["codec_predictions"] = cis_span_payload.get("predictions")
            responses.append(
                ModelResponse(
                    sample_id=sample.sample_id,
                    raw_text=self._extract_output_text(output),
                    metadata=response_meta,
                )
            )
        return responses


class QwenCISSpanDurationAdaptiveCodecRunner(BaseQwenOfficialVideoRunner):
    def ensure_llm(self):
        from qwen_cis_span_duration_adaptive_codec_vllm import (
            apply_qwen_cis_span_duration_adaptive_codec_vllm_runtime_patch,
        )

        apply_qwen_cis_span_duration_adaptive_codec_vllm_runtime_patch()
        return super().ensure_llm()

    def _predict_batch_once(self, samples: list[BenchmarkSample]) -> list[ModelResponse]:
        llm = self.ensure_llm()
        sampling_params = self.build_sampling_params()

        vllm_inputs: list[dict[str, Any]] = []
        metas: list[dict[str, Any]] = []
        for sample in samples:
            generate_input, meta = self._build_generate_input(sample)
            vllm_inputs.append(generate_input)
            metas.append(meta)

        outputs = llm.generate(
            vllm_inputs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        responses: list[ModelResponse] = []
        for sample, output, meta in zip(samples, outputs, metas):
            kv_payload = dict(getattr(output, "kv_transfer_params", {}) or {})
            duration_adaptive_payload = dict(kv_payload.get("qwen_cis_codec", {}) or {})
            response_meta = dict(meta)
            if duration_adaptive_payload:
                response_meta["cis_span_duration_adaptive_codec"] = (
                    duration_adaptive_payload
                )
                response_meta["cis_span_duration_adaptive_codec_predictions"] = (
                    duration_adaptive_payload.get("predictions")
                )
                response_meta["codec_predictions"] = duration_adaptive_payload.get(
                    "predictions"
                )
            responses.append(
                ModelResponse(
                    sample_id=sample.sample_id,
                    raw_text=self._extract_output_text(output),
                    metadata=response_meta,
                )
            )
        return responses


class QwenTimePLERunner(QwenCISSpanDurationAdaptiveCodecRunner):
    """Public TimePLE runner using the vLLM hidden-state side channel."""

    def ensure_llm(self):
        from timeple_vllm import apply_timeple_vllm_runtime_patch

        apply_timeple_vllm_runtime_patch()
        return BaseQwenOfficialVideoRunner.ensure_llm(self)


class QwenTimeEDRunner(BaseQwenOfficialVideoRunner):
    def __init__(self, config: dict[str, Any], *, config_path: Path) -> None:
        super().__init__(config, config_path=config_path)
        self._timeed_full_forward_decoder = None

    def use_qwen_vl_utils_video_preprocess(self) -> bool:
        return True

    def ensure_processor(self):
        if self._processor is None:
            from transformers.models.qwen3_vl import Qwen3VLProcessor

            processor_kwargs = self._get_processor_init_kwargs()
            self._processor = Qwen3VLProcessor.from_pretrained(
                str(self.inference_cfg["model_path"]),
                trust_remote_code=bool(self.inference_cfg.get("trust_remote_code", True)),
                **processor_kwargs,
            )
        return self._processor

    def ensure_llm(self):
        from qwen_timeed_vllm import apply_qwen_timeed_vllm_runtime_patch

        apply_qwen_timeed_vllm_runtime_patch()
        return super().ensure_llm()

    def _timeed_full_forward_fallback_cfg(self) -> dict[str, Any]:
        cfg = self.inference_cfg.get("timeed_full_forward_fallback", {})
        return dict(cfg) if isinstance(cfg, dict) else {}

    def _timeed_full_forward_fallback_enabled(self) -> bool:
        return bool(self._timeed_full_forward_fallback_cfg().get("enabled", False))

    def _replace_response_text_with_timeed_fallback(self) -> bool:
        cfg = self._timeed_full_forward_fallback_cfg()
        return bool(cfg.get("replace_response_text", True))

    def ensure_timeed_full_forward_decoder(self):
        if self._timeed_full_forward_decoder is None:
            from qwen_timeed_full_forward import TimeEDFullForwardFallbackDecoder

            cfg = self._timeed_full_forward_fallback_cfg()
            dtype = str(cfg.get("dtype") or self.engine_cfg.get("dtype") or "bfloat16")
            if dtype == "auto":
                dtype = "bfloat16"
            self._timeed_full_forward_decoder = TimeEDFullForwardFallbackDecoder(
                model_path=str(self.inference_cfg["model_path"]),
                timeed_project_root=str(
                    cfg.get("timeed_project_root")
                    or str(PROJECT_ROOT)
                ),
                device=str(cfg.get("device") or "cuda"),
                dtype=dtype,
                attn_implementation=str(cfg.get("attn_implementation") or "sdpa"),
                insert_space_before_eos=bool(cfg.get("insert_space_before_eos", True)),
            )
        return self._timeed_full_forward_decoder

    def _extract_output_text(self, output: Any) -> str:
        outputs = getattr(output, "outputs", None)
        if not outputs:
            return ""

        first = outputs[0]
        token_ids = getattr(first, "token_ids", None)
        if token_ids is None:
            return super()._extract_output_text(output)

        if hasattr(token_ids, "tolist") and callable(token_ids.tolist):
            token_ids = token_ids.tolist()
        token_ids = [int(token_id) for token_id in token_ids]

        tokenizer = getattr(self.ensure_processor(), "tokenizer", None)
        if tokenizer is None:
            return super()._extract_output_text(output)

        try:
            return tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return tokenizer.decode(token_ids, skip_special_tokens=False)

    def _predict_batch_once(self, samples: list[BenchmarkSample]) -> list[ModelResponse]:
        llm = self.ensure_llm()
        sampling_params = self.build_sampling_params()

        vllm_inputs: list[dict[str, Any]] = []
        metas: list[dict[str, Any]] = []
        for sample in samples:
            generate_input, meta = self._build_generate_input(sample)
            vllm_inputs.append(generate_input)
            metas.append(meta)

        outputs = llm.generate(
            vllm_inputs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        responses: list[ModelResponse] = []
        for sample, output, meta in zip(samples, outputs, metas):
            kv_payload = dict(getattr(output, "kv_transfer_params", {}) or {})
            timeed_payload = dict(kv_payload.get("qwen_timeed", {}) or {})
            raw_text = self._extract_output_text(output)
            response_meta = dict(meta)
            if timeed_payload:
                response_meta["timeed"] = timeed_payload
                response_meta["timeed_predictions"] = timeed_payload.get("predictions")
                response_meta["codec_predictions"] = timeed_payload.get("predictions")
            else:
                from qwen_timeed_full_forward import should_run_full_forward_fallback

                if should_run_full_forward_fallback(
                    enabled=self._timeed_full_forward_fallback_enabled(),
                    response_text=raw_text,
                    response_meta=response_meta,
                ):
                    try:
                        cfg = self._timeed_full_forward_fallback_cfg()
                        fallback_result = self.ensure_timeed_full_forward_decoder().decode(
                            video_path=sample.video_path,
                            prompt_user_text=self.render_user_prompt(sample),
                            response_text=raw_text,
                            video_kwargs=dict(meta.get("video_kwargs") or {}),
                            force_insert=bool(cfg.get("force_insert", False)),
                        )
                        fallback_payload = dict(fallback_result.payload)
                        response_meta["timeed"] = fallback_payload
                        response_meta["timeed_predictions"] = fallback_payload.get("predictions")
                        response_meta["codec_predictions"] = fallback_payload.get("predictions")
                        response_meta["timeed_full_forward_fallback"] = {
                            "enabled": True,
                            "original_response_text": raw_text,
                            "patched_response_text": fallback_result.patched_response_text,
                            "mode": fallback_payload.get("mode"),
                            "completion_patch": fallback_payload.get("completion_patch"),
                        }
                        if self._replace_response_text_with_timeed_fallback():
                            raw_text = fallback_result.patched_response_text
                    except Exception as exc:
                        LOGGER.exception(
                            "TimeED full-forward fallback failed for sample_id=%s",
                            sample.sample_id,
                        )
                        response_meta["timeed_full_forward_fallback_error"] = repr(exc)
            responses.append(
                ModelResponse(
                    sample_id=sample.sample_id,
                    raw_text=raw_text,
                    metadata=response_meta,
                )
            )
        return responses


class VideoChatR1Runner(BaseQwenOfficialVideoRunner):
    def use_qwen_vl_utils_video_preprocess(self) -> bool:
        return True

    def ensure_llm(self):
        # The current official transformers reference for VideoChat-R1 is still
        # based on the older Qwen2.5-VL processor semantics where
        # `second_per_grid_ts` comes from the requested `fps`, not from
        # metadata.sampled_fps. Patch only this runner to preserve that
        # temporal scale under the eval-suite vLLM environment.
        apply_videochat_r1_qwen25_processor_runtime_patch()
        return super().ensure_llm()

    def should_include_video_metadata(self) -> bool:
        # Keep the current prompt/data contract identical to the official
        # transformers comparison path: qwen_vl_utils pre-samples the tensor,
        # then the processor receives the sampled video without explicit
        # metadata. VideoChat-R1-specific temporal-scale compatibility is
        # handled in `ensure_llm()` above.
        return False


class BaseQwenOfficialTransformersRunner(SingleVideoMessageBuilderMixin, BaseTransformersRunner):
    def __init__(self, config: dict[str, Any], *, config_path: Path) -> None:
        super().__init__(config, config_path=config_path)
        self._model = None
        self._processor = None
        self._device = None
        self._process_vision_info = None

    def build_video_item(self, sample: BenchmarkSample) -> dict[str, Any]:
        return {
            "type": "video",
            "video": str(Path(sample.video_path).resolve()),
        }

    def _get_processor_init_kwargs(self) -> dict[str, Any]:
        return dict(self.inference_cfg.get("processor_kwargs", {}))

    def _resolve_torch_dtype(self, raw_dtype: Any) -> Any:
        import torch

        if not isinstance(raw_dtype, str):
            return raw_dtype

        lowered = raw_dtype.lower()
        if lowered in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if lowered in {"fp16", "float16", "half"}:
            return torch.float16
        if lowered == "float32":
            return torch.float32
        return "auto"

    def ensure_model_components(self):
        if self._model_init_error is not None:
            raise RuntimeError(self._model_init_error)
        if (
            self._model is not None
            and self._processor is not None
            and self._device is not None
            and self._process_vision_info is not None
        ):
            return self._model, self._processor, self._device, self._process_vision_info

        try:
            _ensure_qwen_vl_utils_importable()

            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            device = torch.device(self.get_device_string())
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    f"Requested CUDA device {device}, but torch.cuda.is_available() is False."
                )

            official_cfg = dict(self.model_cfg.get("official", {}))
            load_cfg = dict(official_cfg.get("load", {}))
            trust_remote_code = bool(
                load_cfg.get(
                    "trust_remote_code",
                    self.inference_cfg.get("trust_remote_code", True),
                )
            )
            requested_device_map = load_cfg.get("device_map", "auto")
            if requested_device_map is not None and importlib.util.find_spec("accelerate") is None:
                LOGGER.warning(
                    "accelerate is unavailable; falling back from device_map=%r to explicit model.to(%s).",
                    requested_device_map,
                    device,
                )
                requested_device_map = None

            attn_implementation = load_cfg.get("attn_implementation")
            if (
                attn_implementation == "flash_attention_2"
                and importlib.util.find_spec("flash_attn") is None
            ):
                LOGGER.warning(
                    "flash_attn is unavailable; falling back from attn_implementation=%r.",
                    attn_implementation,
                )
                attn_implementation = None

            model_kwargs: dict[str, Any] = {
                "torch_dtype": self._resolve_torch_dtype(load_cfg.get("dtype", "auto")),
                "trust_remote_code": trust_remote_code,
            }
            if requested_device_map is not None:
                model_kwargs["device_map"] = requested_device_map
            if attn_implementation is not None:
                model_kwargs["attn_implementation"] = attn_implementation

            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(self.inference_cfg["model_path"]),
                **model_kwargs,
            ).eval()
            if requested_device_map is None:
                model = model.to(device)

            processor = AutoProcessor.from_pretrained(
                str(self.inference_cfg["model_path"]),
                trust_remote_code=trust_remote_code,
                **self._get_processor_init_kwargs(),
            )

            self._model = model
            self._processor = processor
            self._device = device
            self._process_vision_info = process_vision_info
        except Exception as exc:
            self._model_init_error = str(exc)
            LOGGER.error(
                "Failed to initialize official transformers model=%s family=%s: %s",
                self.inference_cfg.get("model_path"),
                self.model_cfg.get("runner", {}).get("family"),
                self._model_init_error,
            )
            raise

        return self._model, self._processor, self._device, self._process_vision_info

    def _build_official_messages(
        self,
        sample: BenchmarkSample,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        messages = [dict(message) for message in self.build_messages(sample)]
        video_message_kwargs = self._translate_video_message_kwargs()
        resolved_video_path = str(Path(sample.video_path).resolve())

        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue

            converted_content: list[dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    converted_content.append(item)
                    continue

                item_type = item.get("type")
                if item_type in {"video", "video_url"} or "video_url" in item:
                    converted_item = {"type": "video", "video": resolved_video_path}
                    converted_item.update(video_message_kwargs)
                    converted_content.append(converted_item)
                    continue

                converted_content.append(dict(item))
            message["content"] = converted_content

        return messages, video_message_kwargs

    def _prepare_inputs(
        self,
        messages: list[dict[str, Any]],
        *,
        processor: Any,
        device: Any,
        process_vision_info: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=bool(self.engine_cfg.get("add_generation_prompt", True)),
        )

        image_processor = getattr(processor, "image_processor", None)
        patch_size = int(getattr(image_processor, "patch_size", 14))
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            image_patch_size=patch_size,
        )
        if isinstance(video_kwargs, dict):
            normalized_video_kwargs: dict[str, Any] = {}
            for key, value in video_kwargs.items():
                if isinstance(value, list) and len(value) == 1:
                    normalized_video_kwargs[key] = value[0]
                else:
                    normalized_video_kwargs[key] = value
            video_kwargs = normalized_video_kwargs

        processor_kwargs: dict[str, Any] = {
            "text": [prompt_text],
            "images": image_inputs,
            "videos": video_inputs,
            "padding": True,
            "return_tensors": "pt",
        }
        if video_kwargs:
            processor_kwargs.update(video_kwargs)

        inputs = processor(**processor_kwargs)
        inputs = inputs.to(device)
        input_dict = {key: value for key, value in inputs.items()}
        if "input_ids" in input_dict and "inputs_embeds" in input_dict:
            input_dict.pop("inputs_embeds", None)

        response_meta = {
            "prompt_text": prompt_text,
            "processor_video_kwargs": dict(video_kwargs or {}),
            "input_keys": sorted(input_dict.keys()),
        }
        return input_dict, response_meta

    def _build_generate_inputs(self, inputs: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise ValueError("Processor outputs missing input_ids for generation.")

        generate_inputs: dict[str, Any] = {}
        for key in (
            "input_ids",
            "attention_mask",
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "timestamp_labels",
            "timestamp_positions",
            "timespan_labels",
            "timespan_positions",
        ):
            if key in inputs:
                generate_inputs[key] = inputs[key]

        return generate_inputs, input_ids

    def _predict_batch_once(self, samples: list[BenchmarkSample]) -> list[ModelResponse]:
        import torch

        model, processor, device, process_vision_info = self.ensure_model_components()
        generation_config = self.build_generation_config()
        responses: list[ModelResponse] = []

        for sample in samples:
            messages, input_video_kwargs = self._build_official_messages(sample)
            inputs, response_meta = self._prepare_inputs(
                messages,
                processor=processor,
                device=device,
                process_vision_info=process_vision_info,
            )
            generate_inputs, input_ids = self._build_generate_inputs(inputs)

            started = time.monotonic()
            with torch.inference_mode():
                output_ids = model.generate(
                    **generate_inputs,
                    **generation_config,
                )
            elapsed_seconds = round(time.monotonic() - started, 3)

            generated_ids = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, output_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            response_meta.update(
                {
                    "backend": "transformers",
                    "elapsed_seconds": elapsed_seconds,
                    "input_video_kwargs": input_video_kwargs,
                }
            )
            responses.append(
                ModelResponse(
                    sample_id=sample.sample_id,
                    raw_text=str(output_text),
                    metadata=response_meta,
                )
            )

        return responses


class VideoChatR1TransformersRunner(BaseQwenOfficialTransformersRunner):
    pass


class Qwen25VLTransformersRunner(BaseQwenOfficialTransformersRunner):
    pass


class NumProQwen25VLTransformersRunner(NumProFrameSamplingMixin, BaseQwenOfficialTransformersRunner):
    def _build_official_messages(
        self,
        sample: BenchmarkSample,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        messages, meta = self._build_numpro_image_messages(sample)
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_pil":
                    item["type"] = "image"
                    item["image"] = item.pop("image_pil")
        return messages, meta
