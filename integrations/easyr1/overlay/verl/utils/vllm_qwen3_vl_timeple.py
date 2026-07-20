from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional

import torch
from transformers import BatchFeature
from transformers.video_utils import VideoMetadata
from vllm.logger import init_logger
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration as VLLMQwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)

logger = init_logger(__name__)

_TIMEPLE_INTERFACE_GATE_KEYS = {
    "timeple_interface_adapter.input_adapter.gate",
    "timeple_interface_adapter.output_adapter.gate",
}


def _metadata_get(metadata: Any, key: str, default: Any = None) -> Any:
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    return getattr(metadata, key, default)


def _normalize_video_metadata(metadata: Any, num_frames: int, fallback_fps: Optional[float]) -> dict[str, Any]:
    fps = _metadata_get(metadata, "fps", fallback_fps)
    if fps is None:
        fps = 2.0

    frames_indices = _metadata_get(metadata, "frames_indices", None)
    if frames_indices is None:
        frames_indices = list(range(num_frames))

    total_num_frames = _metadata_get(metadata, "total_num_frames", len(frames_indices))
    duration = _metadata_get(metadata, "duration", float(total_num_frames) / float(fps) if fps else 0.0)
    video_backend = _metadata_get(metadata, "video_backend", "easyr1")
    do_sample_frames = bool(_metadata_get(metadata, "do_sample_frames", False))

    return {
        "fps": float(fps),
        "frames_indices": list(frames_indices),
        "total_num_frames": int(total_num_frames),
        "duration": float(duration),
        "video_backend": video_backend,
        "do_sample_frames": do_sample_frames,
    }


class Qwen3VLTimePLEProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        from transformers.models.qwen3_vl.configuration_qwen3_vl_timeple import Qwen3VLTimePLEConfig

        return self.ctx.get_hf_config(Qwen3VLTimePLEConfig)

    def get_hf_processor(self, **kwargs: object):
        from transformers.models.qwen3_vl.processing_qwen3_vl_timeple import Qwen3VLProcessorWithTimePLECodec

        return self.ctx.get_hf_processor(
            Qwen3VLProcessorWithTimePLECodec,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )


class Qwen3VLTimePLEMultiModalProcessor(Qwen3VLMultiModalProcessor):
    def _build_video_placeholder_from_grid(self, processor: Any, *, grid_thw: torch.Tensor) -> str:
        merge_length = processor.video_processor.merge_size**2
        frame_seqlen = int(grid_thw[1:].prod().item() // merge_length)

        placeholder_parts: list[str] = []
        for _ in range(int(grid_thw[0].item())):
            placeholder_parts.append(processor.timestamp_token)
            placeholder_parts.append(processor.vision_start_token)
            placeholder_parts.append(processor.video_token * frame_seqlen)
            placeholder_parts.append(processor.vision_end_token)

        return "".join(placeholder_parts)

    def _normalize_sequence_value(self, value: Any, *, item_idx: int, grid_t: int) -> Optional[list[float]]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.tolist()
        elif hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
            value = value.tolist()
        elif isinstance(value, tuple):
            value = list(value)

        if not isinstance(value, (list, tuple)):
            normalized = [float(value)]
        elif len(value) == 0:
            normalized = []
        else:
            normalized_items = []
            for item in value:
                if isinstance(item, torch.Tensor):
                    normalized_items.append(item.tolist())
                elif hasattr(item, "tolist") and not isinstance(item, (list, tuple)):
                    normalized_items.append(item.tolist())
                elif isinstance(item, tuple):
                    normalized_items.append(list(item))
                else:
                    normalized_items.append(item)
            value = normalized_items

            first = value[0]
            if isinstance(first, (list, tuple)):
                if item_idx >= len(value):
                    return None
                normalized = [float(v) for v in value[item_idx]]
            else:
                normalized = [float(v) for v in value]

        if len(normalized) == 0:
            return None
        if len(normalized) < grid_t:
            normalized = normalized + [normalized[-1]] * (grid_t - len(normalized))
        return normalized[:grid_t]

    def _normalize_provided_time_labels(
        self,
        starts_value: Any,
        ends_value: Any,
        durations_value: Any,
        *,
        item_idx: int,
        grid_t: int,
    ) -> Optional[tuple[list[float], list[float], list[float]]]:
        starts = self._normalize_sequence_value(starts_value, item_idx=item_idx, grid_t=grid_t)
        ends = self._normalize_sequence_value(ends_value, item_idx=item_idx, grid_t=grid_t)
        durations = self._normalize_sequence_value(durations_value, item_idx=item_idx, grid_t=grid_t)
        if starts is None or ends is None:
            return None
        if len(starts) != len(ends):
            raise ValueError("Provided rollout timestamp labels have inconsistent start/end lengths.")
        if durations is None:
            durations = [max(float(max(ends)), 1.0)] * len(starts)
        elif len(durations) != len(starts):
            raise ValueError("Provided rollout timestamp labels have inconsistent duration lengths.")
        return starts, ends, durations

    def _build_video_placeholder(
        self,
        processor: Any,
        *,
        metadata: dict[str, Any],
        grid_thw: torch.Tensor,
    ) -> tuple[list[float], list[float], list[float], str]:
        temporal_patch_size = getattr(processor.video_processor, "temporal_patch_size", None)
        if temporal_patch_size is None:
            temporal_patch_size = 1

        time_segments = processor._calculate_time_segments(
            metadata["frames_indices"],
            metadata["fps"],
            int(temporal_patch_size),
        )

        grid_t = int(grid_thw[0].item())
        if len(time_segments) < grid_t and len(time_segments) > 0:
            time_segments = time_segments + [time_segments[-1]] * (grid_t - len(time_segments))
        time_segments = time_segments[:grid_t]

        merge_length = processor.video_processor.merge_size**2
        frame_seqlen = int(grid_thw[1:].prod().item() // merge_length)
        video_duration = float(metadata["duration"]) if float(metadata["duration"]) > 0 else 1.0

        starts: list[float] = []
        ends: list[float] = []
        durations: list[float] = []
        placeholder_parts: list[str] = []

        for start_time, end_time in time_segments:
            starts.append(float(start_time))
            ends.append(float(end_time))
            durations.append(video_duration)
            placeholder_parts.append(processor.timestamp_token)
            placeholder_parts.append(processor.vision_start_token)
            placeholder_parts.append(processor.video_token * frame_seqlen)
            placeholder_parts.append(processor.vision_end_token)

        return starts, ends, durations, "".join(placeholder_parts)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        mm_kwargs = dict(mm_kwargs)
        provided_timestamp_labels_start = mm_kwargs.pop("timestamp_labels_start", None)
        provided_timestamp_labels_end = mm_kwargs.pop("timestamp_labels_end", None)
        provided_timestamp_video_durations = mm_kwargs.pop("timestamp_video_durations", None)
        processor = self.info.get_hf_processor(**mm_kwargs)

        if ("videos" in mm_data and isinstance(mm_data["videos"], list) and len(mm_data["videos"]) > 0):
            video_grid_thw_lst = []
            pixel_values_videos_lst = []
            timestamp_labels_start = []
            timestamp_labels_end = []
            timestamp_video_durations = []

            for item_idx, item in enumerate(mm_data.pop("videos", [])):
                if isinstance(item, tuple) and len(item) == 2:
                    video_array, metadata = item
                else:
                    video_array, metadata = item, {}

                normalized_metadata = _normalize_video_metadata(
                    metadata,
                    num_frames=len(video_array),
                    fallback_fps=getattr(processor.video_processor, "fps", None),
                )

                video_mm_kwargs = dict(**mm_kwargs)
                if "do_sample_frames" not in video_mm_kwargs:
                    video_mm_kwargs["do_sample_frames"] = normalized_metadata["do_sample_frames"]

                video_metadata = VideoMetadata(
                    **{k: v for k, v in normalized_metadata.items() if k != "do_sample_frames"}
                )
                video_mm_data = {
                    "videos": [[video_array]],
                    "video_metadata": [[video_metadata]],
                }

                video_outputs = BaseMultiModalProcessor._call_hf_processor(
                    self,
                    prompt="<|vision_start|><|video_pad|><|vision_end|>",
                    mm_data=video_mm_data,
                    mm_kwargs=video_mm_kwargs,
                    tok_kwargs=tok_kwargs,
                )

                video_grid_thw = video_outputs["video_grid_thw"]
                current_grid = video_grid_thw[0]
                provided_labels = self._normalize_provided_time_labels(
                    provided_timestamp_labels_start,
                    provided_timestamp_labels_end,
                    provided_timestamp_video_durations,
                    item_idx=item_idx,
                    grid_t=int(current_grid[0].item()),
                )
                if provided_labels is not None:
                    starts, ends, durations = provided_labels
                    video_placeholder = self._build_video_placeholder_from_grid(processor, grid_thw=current_grid)
                else:
                    starts, ends, durations, video_placeholder = self._build_video_placeholder(
                        processor,
                        metadata=normalized_metadata,
                        grid_thw=current_grid,
                    )
                prompt = prompt.replace("<|vision_start|><|video_pad|><|vision_end|>", video_placeholder, 1)

                video_grid_thw_lst.append(video_grid_thw)
                pixel_values_videos_lst.append(video_outputs["pixel_values_videos"])
                timestamp_labels_start.append(torch.tensor(starts, dtype=torch.float32))
                timestamp_labels_end.append(torch.tensor(ends, dtype=torch.float32))
                timestamp_video_durations.append(torch.tensor(durations, dtype=torch.float32))

            video_outputs = dict(
                pixel_values_videos=torch.cat(pixel_values_videos_lst),
                video_grid_thw=torch.cat(video_grid_thw_lst),
                timestamp_labels_start=timestamp_labels_start,
                timestamp_labels_end=timestamp_labels_end,
                timestamp_video_durations=timestamp_video_durations,
            )
        else:
            video_outputs = dict()

        processed_outputs = BaseMultiModalProcessor._call_hf_processor(
            self,
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        return BatchFeature(dict(processed_outputs, **video_outputs))

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        config = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))

        if "timestamp_labels_start" in hf_inputs:
            config["timestamp_labels_start"] = MultiModalFieldConfig.batched("video")
        if "timestamp_labels_end" in hf_inputs:
            config["timestamp_labels_end"] = MultiModalFieldConfig.batched("video")
        if "timestamp_video_durations" in hf_inputs:
            config["timestamp_video_durations"] = MultiModalFieldConfig.batched("video")

        return config

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor_mm_kwargs = dict(hf_processor_mm_kwargs)
        hf_processor_mm_kwargs.pop("timestamp_labels_start", None)
        hf_processor_mm_kwargs.pop("timestamp_labels_end", None)
        hf_processor_mm_kwargs.pop("timestamp_video_durations", None)
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        hf_config = self.info.get_hf_config()

        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        vision_end_token_id = hf_config.vision_end_token_id
        timestamp_token_id = hf_processor.timestamp_token_id

        merge_length = image_processor.merge_size**2

        def get_image_replacement(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            num_tokens = int(grid_thw.prod()) // merge_length
            return [hf_processor.image_token_id] * num_tokens

        def get_video_replacement(item_idx: int):
            out_item = out_mm_kwargs["video"][item_idx]
            grid_thw = out_item["video_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            num_tokens_per_frame = int(grid_thw[1:].prod()) // merge_length
            placeholder = []
            for _ in range(int(grid_thw[0])):
                placeholder.append(timestamp_token_id)
                placeholder.extend([vision_start_token_id] + [video_token_id] * num_tokens_per_frame + [vision_end_token_id])

            return PromptUpdateDetails.select_token_id(placeholder, video_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=hf_processor.image_token,
                replacement=get_image_replacement,
            ),
            PromptReplacement(
                modality="video",
                target="<|vision_start|><|video_pad|><|vision_end|>",
                replacement=get_video_replacement,
            ),
        ]


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLTimePLEMultiModalProcessor,
    info=Qwen3VLTimePLEProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3VLForConditionalGenerationWithTimePLECodec(VLLMQwen3VLForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config

        self.use_timeple_codec = getattr(config, "use_timeple_codec", False)
        self.use_timeple_interface_adapter = bool(getattr(config, "use_timeple_interface_adapter", False))
        self.timestamp_token_id = getattr(config, "timestamp_token_id", None)
        self.timespan_token_id = getattr(config, "timespan_token_id", None)
        self.default_video_duration_sec = float(getattr(config, "default_video_duration_sec", 1.0))
        self._pending_timestamp_starts: deque[torch.Tensor] = deque()
        self._pending_timestamp_ends: deque[torch.Tensor] = deque()
        self._pending_timestamp_durations: deque[torch.Tensor] = deque()
        self._pending_timestamp_label_count = 0

        if not self.use_timeple_codec:
            return

        from timeple.models import TimePLEInterfaceAdapter, TimePLECodec

        self.timeple_codec = TimePLECodec(config.timeple_codec_config)
        if self.use_timeple_interface_adapter:
            self.timeple_interface_adapter = TimePLEInterfaceAdapter(
                int(config.text_config.hidden_size),
                config=getattr(config, "timeple_interface_adapter", None),
            )

        codec_dim = config.get_timeple_codec_output_dim()
        hidden_size = config.text_config.hidden_size
        if codec_dim != hidden_size:
            raise ValueError(
                f"TimePLE output dimension ({codec_dim}) must match model hidden size ({hidden_size})"
            )

    def _flatten_time_label_batch(self, value: object) -> list[torch.Tensor]:
        if value is None:
            return []

        if isinstance(value, torch.Tensor):
            if value.ndim <= 1:
                return [value.reshape(-1)]
            return [row.reshape(-1) for row in value]

        if isinstance(value, (list, tuple)):
            flattened: list[torch.Tensor] = []
            for item in value:
                if item is None:
                    continue
                if isinstance(item, torch.Tensor):
                    if item.ndim <= 1:
                        flattened.append(item.reshape(-1))
                    else:
                        flattened.extend([row.reshape(-1) for row in item])
                else:
                    flattened.append(torch.as_tensor(item, dtype=torch.float32).reshape(-1))
            return flattened

        return [torch.as_tensor(value, dtype=torch.float32).reshape(-1)]

    def _enqueue_timestamp_labels(self, starts_value: object, ends_value: object, durations_value: object) -> None:
        starts = self._flatten_time_label_batch(starts_value)
        ends = self._flatten_time_label_batch(ends_value)
        durations = self._flatten_time_label_batch(durations_value)
        if len(starts) != len(ends):
            raise ValueError("timestamp_labels_start and timestamp_labels_end have inconsistent batch sizes.")

        if len(durations) == 0:
            durations = [torch.full_like(start, self.default_video_duration_sec) for start in starts]
        elif len(durations) != len(starts):
            raise ValueError("timestamp_video_durations has inconsistent batch size with timestamp labels.")

        for start, end, duration in zip(starts, ends, durations):
            start = start.reshape(-1).detach().to(device="cpu", dtype=torch.float32)
            end = end.reshape(-1).detach().to(device="cpu", dtype=torch.float32)
            duration = duration.reshape(-1).detach().to(device="cpu", dtype=torch.float32)

            if start.numel() != end.numel():
                raise ValueError("Per-item rollout timestamp labels have inconsistent start/end lengths.")
            if duration.numel() == 0:
                duration = torch.full_like(start, self.default_video_duration_sec)
            elif duration.numel() == 1 and start.numel() > 1:
                duration = duration.repeat(start.numel())
            elif duration.numel() != start.numel():
                raise ValueError("Per-item rollout timestamp labels have inconsistent duration lengths.")
            if start.numel() == 0:
                continue

            self._pending_timestamp_starts.append(start)
            self._pending_timestamp_ends.append(end)
            self._pending_timestamp_durations.append(duration.clamp_min(1e-6))
            self._pending_timestamp_label_count += int(start.numel())

    def _consume_pending_timestamp_labels(
        self,
        count: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}.")
        if count == 0:
            empty = torch.empty(0, device=device, dtype=torch.float32)
            return empty, empty, empty
        if count > self._pending_timestamp_label_count:
            raise ValueError(
                "Not enough pending timeple labels for the TIMESTAMP tokens scheduled by vLLM. "
                f"token_count={count}, pending_label_count={self._pending_timestamp_label_count}."
            )

        remaining = count
        start_chunks: list[torch.Tensor] = []
        end_chunks: list[torch.Tensor] = []
        duration_chunks: list[torch.Tensor] = []

        while remaining > 0:
            current_start = self._pending_timestamp_starts.popleft()
            current_end = self._pending_timestamp_ends.popleft()
            current_duration = self._pending_timestamp_durations.popleft()
            take = min(remaining, int(current_start.numel()))

            start_chunks.append(current_start[:take])
            end_chunks.append(current_end[:take])
            duration_chunks.append(current_duration[:take])

            if take < current_start.numel():
                self._pending_timestamp_starts.appendleft(current_start[take:])
                self._pending_timestamp_ends.appendleft(current_end[take:])
                self._pending_timestamp_durations.appendleft(current_duration[take:])

            remaining -= take
            self._pending_timestamp_label_count -= take

        return (
            torch.cat(start_chunks, dim=0).to(device=device, dtype=torch.float32),
            torch.cat(end_chunks, dim=0).to(device=device, dtype=torch.float32),
            torch.cat(duration_chunks, dim=0).to(device=device, dtype=torch.float32),
        )

    def get_multimodal_embeddings(self, **kwargs: object):
        timestamp_labels_start = kwargs.get("timestamp_labels_start")
        timestamp_labels_end = kwargs.get("timestamp_labels_end")
        timestamp_video_durations = kwargs.get("timestamp_video_durations")

        outputs = super().get_multimodal_embeddings(**kwargs)

        if self.use_timeple_codec and timestamp_labels_start is not None and timestamp_labels_end is not None:
            self._enqueue_timestamp_labels(
                timestamp_labels_start,
                timestamp_labels_end,
                timestamp_video_durations,
            )

        return outputs

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
    ) -> torch.Tensor:
        inputs_embeds = super().get_input_embeddings(
            input_ids=input_ids,
            multimodal_embeddings=multimodal_embeddings,
        )

        if not self.use_timeple_codec or self._pending_timestamp_label_count == 0:
            return inputs_embeds

        timestamp_mask = input_ids == self.timestamp_token_id
        timestamp_token_count = int(timestamp_mask.sum().item())
        if timestamp_token_count == 0:
            return inputs_embeds

        start_times, end_times, durations = self._consume_pending_timestamp_labels(
            timestamp_token_count,
            device=inputs_embeds.device,
        )

        time_embeddings = self.timeple_codec.encode(start_times, end_times, durations).to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        if self.use_timeple_interface_adapter:
            anchor_embedding = inputs_embeds[timestamp_mask].detach()
            adapter_output = self.timeple_interface_adapter.forward_input(
                time_embeddings,
                anchor_embedding=anchor_embedding,
            )
            time_embeddings = adapter_output.adapted.to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )

        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[timestamp_mask] = time_embeddings
        return inputs_embeds

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        named_buffers = dict(self.named_buffers())
        named_parameters = dict(self.named_parameters())
        remaining_weights: list[tuple[str, torch.Tensor]] = []
        loaded_buffer_names: set[str] = set()

        for name, tensor in weights:
            if name in _TIMEPLE_INTERFACE_GATE_KEYS and name in named_parameters and tensor.ndim == 0:
                tensor = tensor.reshape(named_parameters[name].shape)

            buffer = named_buffers.get(name)
            if buffer is None:
                remaining_weights.append((name, tensor))
                continue

            source = tensor
            if hasattr(source, "full_tensor"):
                source = source.full_tensor()
            source = source.detach().to(device=buffer.device, dtype=buffer.dtype)
            if tuple(source.shape) != tuple(buffer.shape):
                raise ValueError(
                    f"Buffer shape mismatch for {name}: expected {tuple(buffer.shape)}, got {tuple(source.shape)}"
                )
            with torch.no_grad():
                buffer.copy_(source)
            loaded_buffer_names.add(name)

        loaded = super().load_weights(remaining_weights)
        if isinstance(loaded, set):
            return loaded | loaded_buffer_names
        return loaded


def register_qwen3_vl_timeple_model() -> None:
    ModelRegistry.register_model(
        "Qwen3VLForConditionalGenerationWithTimePLECodec",
        Qwen3VLForConditionalGenerationWithTimePLECodec,
    )
