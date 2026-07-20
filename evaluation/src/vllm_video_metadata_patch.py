from __future__ import annotations

from collections.abc import Mapping
import json
import logging
from typing import Any


LOGGER = logging.getLogger(__name__)

_RUNTIME_PATCHED = False


def _build_default_video_metadata(video: Any) -> dict[str, Any]:
    num_frames = len(video)
    return {
        "total_num_frames": int(num_frames),
        "fps": None,
        "duration": None,
        "frames_indices": list(range(int(num_frames))),
    }


def _meta_get(metadata: object, key: str, default: Any = None) -> Any:
    if metadata is None:
        return default

    if hasattr(metadata, key):
        return getattr(metadata, key)

    if isinstance(metadata, Mapping):
        try:
            return metadata[key]
        except (KeyError, TypeError, AttributeError):
            return default

    return default


def _metadata_to_dict(metadata: object, *, drop_do_sample_frames: bool = False) -> dict[str, Any]:
    if metadata is None:
        return {}

    if isinstance(metadata, Mapping):
        keys = list(metadata)
        result = {
            str(key): metadata[key]
            for key in keys
            if not (drop_do_sample_frames and key == "do_sample_frames")
        }
        return result

    raise TypeError(f"Unsupported video metadata payload: {type(metadata)!r}")


def _coerce_explicit_video_metadata_items(
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

        coerced_items: list[dict[str, Any] | None] = []
        for item in raw_metadata:
            if item is None:
                coerced_items.append(None)
            elif isinstance(item, Mapping):
                coerced_items.append(dict(item))
            else:
                try:
                    coerced_items.append(dict(item))
                except Exception as exc:  # pragma: no cover - defensive fallback
                    raise TypeError(
                        "eval_suite video metadata must contain mapping items."
                    ) from exc
        return coerced_items

    raise TypeError(
        "eval_suite video metadata must be JSON, a mapping, or a list of mappings."
    )


def _coerce_qwen25_timestamp_segments(
    raw_segments: object,
) -> list[tuple[float, float]] | None:
    if raw_segments is None:
        return None

    if isinstance(raw_segments, (bytes, bytearray)):
        raw_segments = raw_segments.decode("utf-8")

    if isinstance(raw_segments, str):
        raw_segments = json.loads(raw_segments)

    if not isinstance(raw_segments, (list, tuple)):
        raise TypeError("Qwen2.5 timestamp segments must be a list.")

    segments: list[tuple[float, float]] = []
    for item in raw_segments:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise TypeError(
                "Qwen2.5 timestamp segments must contain [start, end] pairs."
            )
        start, end = item
        segments.append((float(start), float(end)))
    return segments


def _extract_qwen25_interleave_kwargs(
    mm_kwargs: Mapping[str, object],
) -> tuple[bool, list[tuple[float, float]] | None, dict[str, object]]:
    processor_mm_kwargs = dict(mm_kwargs)
    text_enabled = bool(
        processor_mm_kwargs.pop("eval_suite_qwen25_timestamp_text_interleave", False)
    )
    cis_enabled = bool(
        processor_mm_kwargs.pop("eval_suite_qwen25_cis_timestamp_interleave", False)
    )
    raw_segments = processor_mm_kwargs.pop(
        "eval_suite_qwen25_timestamp_segments_json", None
    )
    if raw_segments is None:
        raw_segments = processor_mm_kwargs.pop(
            "eval_suite_qwen25_timestamp_segments", None
        )
    processor_mm_kwargs.pop("eval_suite_video_metadata_json", None)
    processor_mm_kwargs.pop("eval_suite_video_metadata", None)
    segments = _coerce_qwen25_timestamp_segments(raw_segments)
    return text_enabled or cis_enabled, segments, processor_mm_kwargs


def _to_plain_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return [value]
    return value


def _get_interleaved_video_token_lengths(
    video_grid_thw: Any,
    merge_length: int,
) -> list[int]:
    grids = _to_plain_list(video_grid_thw)
    if grids and not isinstance(grids[0], (list, tuple)):
        grids = [grids]

    lengths: list[int] = []
    for grid in grids:
        grid_t = int(grid[0])
        tokens_per_grid = int(grid[1]) * int(grid[2]) // int(merge_length)
        lengths.extend([tokens_per_grid] * grid_t)
    return lengths


def _expand_interleaved_video_grid_thw(video_grid_thw: Any) -> Any:
    if video_grid_thw is None:
        return None

    grids = video_grid_thw.tolist() if hasattr(video_grid_thw, "tolist") else video_grid_thw
    expanded: list[list[int]] = []
    for grid in grids:
        for _ in range(int(grid[0])):
            expanded.append([1, int(grid[1]), int(grid[2])])

    if hasattr(video_grid_thw, "new_tensor"):
        return video_grid_thw.new_tensor(expanded)
    return expanded


def _expand_interleaved_second_per_grid_ts(
    second_per_grid_ts: Any,
    video_grid_thw: Any,
) -> Any:
    if second_per_grid_ts is None or video_grid_thw is None:
        return second_per_grid_ts

    values = (
        second_per_grid_ts.tolist()
        if hasattr(second_per_grid_ts, "tolist")
        else list(second_per_grid_ts)
    )
    grids = video_grid_thw.tolist() if hasattr(video_grid_thw, "tolist") else video_grid_thw
    expanded: list[float] = []
    for value, grid in zip(values, grids):
        expanded.extend([float(value)] * int(grid[0]))

    if hasattr(second_per_grid_ts, "new_tensor"):
        return second_per_grid_ts.new_tensor(expanded)
    return expanded


def _make_interleaved_video_grid_row_counts(video_grid_thw: Any) -> Any:
    if video_grid_thw is None:
        return None

    row_count = int(video_grid_thw.shape[0]) if hasattr(video_grid_thw, "shape") else len(video_grid_thw)
    if hasattr(video_grid_thw, "new_tensor"):
        return video_grid_thw.new_tensor([row_count])
    return [row_count]


def _build_qwen25_text_interleave_target(
    segments: list[tuple[float, float]],
) -> str:
    return "".join(
        f"<{0.5 * (start + end):.1f} seconds>"
        "<|vision_start|><|video_pad|><|vision_end|>"
        for start, end in segments
    )


def _patch_qwen2_video_data_parser() -> None:
    from vllm.model_executor.models.qwen2_vl import (
        Qwen2VLProcessingInfo,
        Qwen2VLMultiModalDataParser,
    )

    original_get_data_parser = getattr(Qwen2VLProcessingInfo, "get_data_parser", None)
    if original_get_data_parser is None:
        LOGGER.warning("Qwen2VLProcessingInfo.get_data_parser is unavailable; skip patch.")
        return

    if getattr(original_get_data_parser, "_eval_suite_video_metadata_patched", False):
        return

    def patched_get_data_parser(self):
        return Qwen2VLMultiModalDataParser(
            self.get_hf_config().vision_config.spatial_merge_size,
            video_needs_metadata=True,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    patched_get_data_parser._eval_suite_video_metadata_patched = True  # type: ignore[attr-defined]
    Qwen2VLProcessingInfo.get_data_parser = patched_get_data_parser


def _patch_multimodal_video_parse() -> None:
    import numpy as np
    import torch
    from vllm.multimodal.parse import (
        MultiModalDataParser,
        VideoEmbeddingItems,
        VideoProcessorItems,
    )
    from vllm.utils.collection_utils import is_list_of

    import PIL.Image as PILImage

    if getattr(MultiModalDataParser._parse_video_data, "_eval_suite_video_metadata_patched", False):
        return

    def patched_parse_video_data(
        self,
        data,
    ):
        if data is None:
            return None

        if self.is_embeddings(data):
            return VideoEmbeddingItems(data, self.expected_hidden_size)

        data_items: list[Any]
        if (is_list_of(data, PILImage.Image) and len(data) > 0) or (
            isinstance(data, (np.ndarray, torch.Tensor)) and data.ndim == 4
        ):
            data_items = [data]
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            data_items = [elem for elem in data]
        elif isinstance(data, tuple) and len(data) == 2:
            data_items = [data]
        else:
            data_items = data

        new_videos: list[Any] = []
        metadata_lst: list[dict[str, Any] | None] = []
        for data_item in data_items:
            video, metadata = self._get_video_with_metadata(data_item)
            if self.video_needs_metadata:
                if metadata is None:
                    metadata = _build_default_video_metadata(video)
                metadata_lst.append(metadata)
            new_videos.append(video)

        metadata = metadata_lst if self.video_needs_metadata else None
        return VideoProcessorItems(new_videos, metadata=metadata)

    patched_parse_video_data._eval_suite_video_metadata_patched = True  # type: ignore[attr-defined]
    MultiModalDataParser._parse_video_data = patched_parse_video_data


def _patch_video_processor_items() -> None:
    from vllm.multimodal.parse import VideoProcessorItems

    if getattr(VideoProcessorItems.get_processor_data, "_eval_suite_video_metadata_patched", False):
        return

    def patched_get_processor_data(self) -> dict[str, object]:
        videos: list[Any] = []
        for idx in range(self.get_count()):
            item = self.get(idx)
            if isinstance(item, tuple) and len(item) == 2:
                videos.append(item[0])
            else:
                videos.append(item)

        payload: dict[str, object] = {"videos": videos}
        if self.metadata is not None:
            payload["video_metadata"] = self.metadata
        return payload

    patched_get_processor_data._eval_suite_video_metadata_patched = True  # type: ignore[attr-defined]
    VideoProcessorItems.get_processor_data = patched_get_processor_data


def _patch_qwen25_multimodal_processor() -> None:
    try:
        import torch
        from vllm.model_executor.models.qwen2_5_vl import (
            Qwen2_5_VLForConditionalGeneration,
            Qwen2_5_VLMultiModalProcessor,
            compute_retained_tokens_count,
        )
        from vllm.multimodal.inputs import MultiModalFieldConfig
        from vllm.multimodal.processing import PromptReplacement, PromptUpdateDetails
    except ImportError:
        LOGGER.warning("Qwen2.5-VL vLLM processor is unavailable; skip patch.")
        return

    original_call_hf_processor = getattr(
        Qwen2_5_VLMultiModalProcessor,
        "_call_hf_processor",
        None,
    )
    original_get_mm_fields_config = getattr(
        Qwen2_5_VLMultiModalProcessor,
        "_get_mm_fields_config",
        None,
    )
    original_get_prompt_updates = getattr(
        Qwen2_5_VLMultiModalProcessor,
        "_get_prompt_updates",
        None,
    )
    original_hf_processor_applies_updates = getattr(
        Qwen2_5_VLMultiModalProcessor,
        "_hf_processor_applies_updates",
        None,
    )
    original_iter_mm_grid_thw = getattr(
        Qwen2_5_VLForConditionalGeneration,
        "iter_mm_grid_thw",
        None,
    )
    original_embed_multimodal = getattr(
        Qwen2_5_VLForConditionalGeneration,
        "embed_multimodal",
        None,
    )
    if (
        original_call_hf_processor is None
        or original_get_mm_fields_config is None
        or original_get_prompt_updates is None
        or original_hf_processor_applies_updates is None
        or original_iter_mm_grid_thw is None
        or original_embed_multimodal is None
    ):
        LOGGER.warning("Qwen2.5-VL vLLM processor hooks are unavailable; skip patch.")
        return

    if getattr(original_call_hf_processor, "_eval_suite_qwen25_interleave_patched", False):
        return

    def _unwrap_mm_field(value: Any) -> Any:
        return getattr(value, "data", value)

    def _to_int_list(value: Any) -> list[int]:
        result: list[int] = []

        def _append(item: Any) -> None:
            item = _unwrap_mm_field(item)
            if item is None:
                return
            if hasattr(item, "tolist"):
                item = item.tolist()
            if isinstance(item, (list, tuple)):
                for nested_item in item:
                    _append(nested_item)
                return
            result.append(int(item))

        _append(value)
        return result

    def _get_qwen25_interleaved_video_row_counts(video_kwargs: Mapping[str, Any]) -> list[int]:
        return _to_int_list(
            video_kwargs.get("eval_suite_qwen25_interleaved_video_grid_row_counts")
        )

    def _is_qwen25_interleaved_video_kwargs(video_kwargs: Mapping[str, Any]) -> bool:
        row_counts = _get_qwen25_interleaved_video_row_counts(video_kwargs)
        return bool(row_counts and any(count > 1 for count in row_counts))

    def _eval_suite_merge_interleaved_video_outputs(outputs: Any, row_counts: list[int]) -> Any:
        if (
            not row_counts
            or not isinstance(outputs, (list, tuple))
            or not all(isinstance(item, torch.Tensor) and item.ndim == 2 for item in outputs)
        ):
            return outputs

        if sum(row_counts) != len(outputs):
            raise ValueError(
                "Qwen2.5 timestamp text interleave encoder output mismatch: "
                f"row_counts={row_counts} outputs={len(outputs)}."
            )

        merged: list[torch.Tensor] = []
        cursor = 0
        for row_count in row_counts:
            next_cursor = cursor + int(row_count)
            merged.append(torch.cat(list(outputs[cursor:next_cursor]), dim=0))
            cursor = next_cursor
        return type(outputs)(merged) if isinstance(outputs, list) else tuple(merged)

    def _is_qwen25_interleaved_video_feature_data(video_kwargs: Mapping[str, Any]) -> bool:
        row_counts = _get_qwen25_interleaved_video_row_counts(video_kwargs)
        if not row_counts or len(row_counts) != 1 or row_counts[0] <= 1:
            return False
        video_grid_thw = _unwrap_mm_field(video_kwargs.get("video_grid_thw"))
        if video_grid_thw is None:
            return False
        return int(video_grid_thw.shape[0]) == row_counts[0]

    def patched_embed_multimodal(self, **kwargs: object):
        outputs = original_embed_multimodal(self, **kwargs)
        if _is_qwen25_interleaved_video_kwargs(kwargs):
            row_counts = _get_qwen25_interleaved_video_row_counts(kwargs)
            return _eval_suite_merge_interleaved_video_outputs(outputs, row_counts)
        return outputs

    def _iter_interleaved_video_grid_thw(self, mm_feature):
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        tokens_per_second = getattr(self.config.vision_config, "tokens_per_second", 1.0)
        offset = mm_feature.mm_position.offset
        grid_thw = _unwrap_mm_field(mm_feature.data["video_grid_thw"])
        second_per_grid_ts = mm_feature.data.get("second_per_grid_ts", None)
        grid_rows = grid_thw.tolist()
        if second_per_grid_ts is None:
            second_values = [1.0] * len(grid_rows)
        else:
            second_values = _unwrap_mm_field(second_per_grid_ts).tolist()
            if not isinstance(second_values, list):
                second_values = [second_values]

        embed_mask = mm_feature.mm_position.is_embed
        if embed_mask is not None:
            starts = torch.nonzero(
                torch.diff(embed_mask.int(), prepend=embed_mask.new_zeros(1)) == 1
            ).flatten().tolist()
        else:
            starts = []
            cursor = 0
            for grid in grid_rows:
                starts.append(cursor)
                cursor += int(grid[0]) * int(grid[1]) * int(grid[2]) // (
                    spatial_merge_size**2
                )
        if len(starts) != len(grid_rows):
            raise ValueError(
                "Qwen2.5 timestamp text interleave M-RoPE mismatch: "
                f"video_grid_rows={len(grid_rows)} embed_spans={len(starts)}."
            )

        for start_offset, grid, second_value in zip(starts, grid_rows, second_values):
            t, h, w = [int(v) for v in grid]
            t_factor = int(float(second_value)) * tokens_per_second
            yield (
                offset + int(start_offset),
                t,
                h // spatial_merge_size,
                w // spatial_merge_size,
                t_factor,
            )

    def patched_iter_mm_grid_thw(self, mm_features):
        for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
            if (
                mm_feature.modality == "video"
                and mm_feature.data is not None
                and _is_qwen25_interleaved_video_feature_data(mm_feature.data)
            ):
                yield from _iter_interleaved_video_grid_thw(self, mm_feature)
            else:
                yield from original_iter_mm_grid_thw(self, [mm_feature])

    def patched_call_hf_processor(
        self,
        prompt: str,
        mm_data,
        mm_kwargs,
        tok_kwargs,
    ):
        interleave_enabled, _segments, processor_mm_kwargs = (
            _extract_qwen25_interleave_kwargs(mm_kwargs)
        )
        if not interleave_enabled:
            return original_call_hf_processor(
                self,
                prompt=prompt,
                mm_data=mm_data,
                mm_kwargs=processor_mm_kwargs,
                tok_kwargs=tok_kwargs,
            )

        mm_data = dict(mm_data)
        videos = mm_data.pop("videos", None)
        video_metadata = mm_data.pop("video_metadata", None)
        if not videos:
            return original_call_hf_processor(
                self,
                prompt=prompt,
                mm_data=mm_data,
                mm_kwargs=processor_mm_kwargs,
                tok_kwargs=tok_kwargs,
            )

        video_mm_data: dict[str, object] = {"videos": videos}
        if video_metadata is not None:
            video_mm_data["video_metadata"] = video_metadata

        video_outputs = original_call_hf_processor(
            self,
            prompt="<|vision_start|><|video_pad|><|vision_end|>",
            mm_data=video_mm_data,
            mm_kwargs=processor_mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        video_outputs.pop("input_ids", None)

        original_video_grid_thw = video_outputs.get("video_grid_thw")
        if original_video_grid_thw is not None:
            video_outputs["video_grid_thw"] = _expand_interleaved_video_grid_thw(
                original_video_grid_thw
            )
            video_outputs["eval_suite_qwen25_interleaved_video_grid_row_counts"] = (
                _make_interleaved_video_grid_row_counts(video_outputs["video_grid_thw"])
            )
        if "second_per_grid_ts" in video_outputs:
            video_outputs["second_per_grid_ts"] = _expand_interleaved_second_per_grid_ts(
                video_outputs["second_per_grid_ts"],
                original_video_grid_thw,
            )

        text_outputs = original_call_hf_processor(
            self,
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=processor_mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        return type(text_outputs)(dict(text_outputs, **video_outputs))

    def patched_hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items,
        hf_processor_mm_kwargs,
        tokenization_kwargs,
    ) -> bool:
        interleave_enabled, _segments, _processor_mm_kwargs = (
            _extract_qwen25_interleave_kwargs(hf_processor_mm_kwargs)
        )
        if interleave_enabled:
            return False
        return original_hf_processor_applies_updates(
            self,
            prompt_text=prompt_text,
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
        )

    def patched_get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs,
    ):
        interleave_enabled, _segments, processor_mm_kwargs = (
            _extract_qwen25_interleave_kwargs(hf_processor_mm_kwargs)
        )
        fields = dict(
            original_get_mm_fields_config(
                self,
                hf_inputs,
                processor_mm_kwargs,
            )
        )
        if not interleave_enabled or "video_grid_thw" not in hf_inputs:
            return fields

        video_grid_thw = hf_inputs["video_grid_thw"]
        if len(video_grid_thw) == 0:
            return fields

        row_count = int(video_grid_thw.shape[0])
        video_grid_sizes = video_grid_thw.prod(-1)
        pixel_size = torch.tensor(
            [int(video_grid_sizes.sum().item())],
            dtype=video_grid_sizes.dtype,
            device=video_grid_sizes.device,
        )
        embed_size = torch.tensor(
            [
                int(
                    (
                        video_grid_sizes
                        // (self.info.get_hf_config().vision_config.spatial_merge_size**2)
                    )
                    .sum()
                    .item()
                )
            ],
            dtype=video_grid_sizes.dtype,
            device=video_grid_sizes.device,
        )
        one_video_slice = [slice(0, row_count)]
        fields["pixel_values_videos"] = MultiModalFieldConfig.flat_from_sizes(
            "video",
            pixel_size,
        )
        fields["video_embeds"] = MultiModalFieldConfig.flat_from_sizes(
            "video",
            embed_size,
        )
        fields["video_grid_thw"] = MultiModalFieldConfig.flat(
            "video",
            one_video_slice,
        )
        if "second_per_grid_ts" in hf_inputs:
            fields["second_per_grid_ts"] = MultiModalFieldConfig.flat(
                "video",
                one_video_slice,
            )
        if "eval_suite_qwen25_interleaved_video_grid_row_counts" in hf_inputs:
            fields["eval_suite_qwen25_interleaved_video_grid_row_counts"] = (
                MultiModalFieldConfig.batched("video", keep_on_cpu=True)
            )
        return fields

    def patched_get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs,
        out_mm_kwargs,
    ):
        interleave_enabled, segments, processor_mm_kwargs = (
            _extract_qwen25_interleave_kwargs(hf_processor_mm_kwargs)
        )
        if not interleave_enabled:
            return original_get_prompt_updates(
                self,
                mm_items,
                processor_mm_kwargs,
                out_mm_kwargs,
            )
        if not segments:
            raise ValueError("Qwen2.5 timestamp text interleave requires timestamp segments.")

        hf_processor = self.info.get_hf_processor(**processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        hf_config = self.info.get_hf_config()
        vocab = tokenizer.get_vocab()
        placeholder = {
            "image": vocab[hf_processor.image_token],
            "video": vocab[hf_processor.video_token],
        }
        merge_length = int(image_processor.merge_size) ** 2

        def get_image_replacement(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            num_tokens = int(grid_thw.prod()) // merge_length
            return [placeholder["image"]] * num_tokens

        def get_video_replacement(item_idx: int):
            out_item = out_mm_kwargs["video"][item_idx]
            grid_thw = out_item["video_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            token_lengths = _get_interleaved_video_token_lengths(
                grid_thw,
                merge_length,
            )
            if len(token_lengths) != len(segments):
                raise ValueError(
                    "Qwen2.5 timestamp text interleave segment count must match "
                    f"video grids: segments={len(segments)} grids={len(token_lengths)}."
                )

            full: list[int] = []
            for grid_idx, ((start, end), token_length) in enumerate(
                zip(segments, token_lengths)
            ):
                full.extend(
                    tokenizer.encode(
                        f"<{0.5 * (start + end):.1f} seconds>",
                        add_special_tokens=False,
                    )
                )
                full.append(int(hf_config.vision_start_token_id))
                video_pruning_rate = self.info.ctx.get_mm_config().video_pruning_rate
                if video_pruning_rate is not None and video_pruning_rate > 0.0:
                    grid = grid_thw[grid_idx]
                    _grid_t, grid_h, grid_w = [int(v) for v in grid]
                    tokens_per_frame = (
                        grid_h // int(image_processor.merge_size)
                    ) * (grid_w // int(image_processor.merge_size))
                    token_length = compute_retained_tokens_count(
                        tokens_per_frame=tokens_per_frame,
                        num_frames=_grid_t,
                        q=float(video_pruning_rate),
                    )
                full.extend([int(hf_config.video_token_id)] * int(token_length))
                full.append(int(hf_config.vision_end_token_id))
            return PromptUpdateDetails.select_token_id(
                full,
                int(hf_config.video_token_id),
            )

        return [
            PromptReplacement(
                modality="image",
                target=[placeholder["image"]],
                replacement=get_image_replacement,
            ),
            PromptReplacement(
                modality="video",
                target=_build_qwen25_text_interleave_target(segments),
                replacement=get_video_replacement,
            ),
        ]

    patched_call_hf_processor._eval_suite_qwen25_interleave_patched = True  # type: ignore[attr-defined]
    Qwen2_5_VLMultiModalProcessor._call_hf_processor = patched_call_hf_processor
    Qwen2_5_VLMultiModalProcessor._hf_processor_applies_updates = (
        patched_hf_processor_applies_updates
    )
    Qwen2_5_VLMultiModalProcessor._get_mm_fields_config = (
        patched_get_mm_fields_config
    )
    Qwen2_5_VLMultiModalProcessor._get_prompt_updates = patched_get_prompt_updates
    Qwen2_5_VLForConditionalGeneration.embed_multimodal = patched_embed_multimodal
    patched_iter_mm_grid_thw._eval_suite_qwen25_iter_mm_grid_thw_patched = True  # type: ignore[attr-defined]
    Qwen2_5_VLForConditionalGeneration.iter_mm_grid_thw = patched_iter_mm_grid_thw


def _patch_qwen3_multimodal_processor() -> None:
    try:
        import torch
        from transformers.feature_extraction_utils import BatchFeature
        from transformers.video_utils import VideoMetadata
        from vllm.model_executor.models.qwen3_vl import (
            Qwen3VLMultiModalProcessor,
            compute_retained_tokens_count,
        )
        from vllm.multimodal.processing import BaseMultiModalProcessor
    except ImportError:
        LOGGER.warning("Qwen3VLMultiModalProcessor is unavailable; skip qwen3 video patch.")
        return

    original_call_hf_processor = getattr(Qwen3VLMultiModalProcessor, "_call_hf_processor", None)
    if original_call_hf_processor is None:
        LOGGER.warning("Qwen3VLMultiModalProcessor._call_hf_processor is unavailable; skip patch.")
        return

    if getattr(original_call_hf_processor, "_eval_suite_video_metadata_patched", False):
        return

    def patched_call_hf_processor(
        self,
        prompt: str,
        mm_data,
        mm_kwargs,
        tok_kwargs,
    ):
        mm_data = dict(mm_data)
        processor_mm_kwargs = dict(mm_kwargs)
        explicit_video_metadata_items = _coerce_explicit_video_metadata_items(
            processor_mm_kwargs.pop("eval_suite_video_metadata_json", None),
            processor_mm_kwargs.pop("eval_suite_video_metadata", None),
        )
        processor = self.info.get_hf_processor(**processor_mm_kwargs)

        if videos := mm_data.pop("videos", []):
            video_metadata_items = mm_data.pop("video_metadata", None)
            video_grid_thw_lst = []
            pixel_values_videos_lst = []
            timestamps_per_video = []

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
                do_sample_frames = bool(_meta_get(metadata, "do_sample_frames", False))
                if metadata is None:
                    metadata = VideoMetadata(**_build_default_video_metadata(video_array))
                elif not isinstance(metadata, VideoMetadata):
                    metadata = VideoMetadata(
                        **_metadata_to_dict(metadata, drop_do_sample_frames=True)
                    )

                if "do_sample_frames" not in video_mm_kwargs:
                    video_mm_kwargs["do_sample_frames"] = do_sample_frames

                if metadata["fps"] is None and video_mm_kwargs.get("fps") is not None:
                    metadata["fps"] = float(video_mm_kwargs["fps"])
                    if metadata["duration"] is None and metadata["total_num_frames"] is not None:
                        metadata["duration"] = float(metadata["total_num_frames"]) / metadata["fps"]

                timestamps = self.info._get_video_second_idx(
                    metadata=metadata,
                    do_sample_frames=video_mm_kwargs["do_sample_frames"],
                    sampled_fps=video_mm_kwargs.get("fps"),
                    sampled_num_frames=video_mm_kwargs.get("num_frames"),
                )
                timestamps_per_video.append(timestamps)

                video_mm_data = {
                    "videos": [[video_array]],
                    "video_metadata": [[metadata]],
                }
                if "num_frames" in video_mm_kwargs and "fps" not in video_mm_kwargs:
                    video_mm_kwargs["fps"] = None

                video_outputs = BaseMultiModalProcessor._call_hf_processor(
                    self,
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
                    select_token_id = True

                video_repl = Qwen3VLMultiModalProcessor.get_video_repl(
                    tokens_per_frame=tokens_per_frame,
                    timestamps=timestamps,
                    tokenizer=self.info.get_tokenizer(),
                    vision_start_token_id=self.info.get_hf_config().vision_start_token_id,
                    vision_end_token_id=self.info.get_hf_config().vision_end_token_id,
                    video_token_id=self.info.get_hf_config().video_token_id,
                    select_token_id=select_token_id,
                )
                video_outputs.pop("input_ids", None)
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

            video_outputs = dict(
                pixel_values_videos=torch.cat(pixel_values_videos_lst),
                video_grid_thw=torch.cat(video_grid_thw_lst),
                timestamps=timestamps_per_video,
            )
        else:
            video_outputs = dict()

        processed_outputs = BaseMultiModalProcessor._call_hf_processor(
            self,
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=processor_mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        return BatchFeature(dict(processed_outputs, **video_outputs))

    patched_call_hf_processor._eval_suite_video_metadata_patched = True  # type: ignore[attr-defined]
    Qwen3VLMultiModalProcessor._call_hf_processor = patched_call_hf_processor


def apply_vllm_video_metadata_runtime_patch() -> None:
    global _RUNTIME_PATCHED
    if _RUNTIME_PATCHED:
        return

    _patch_qwen2_video_data_parser()
    _patch_qwen25_multimodal_processor()
    _patch_qwen3_multimodal_processor()

    _RUNTIME_PATCHED = True
    LOGGER.info("Applied eval_suite vLLM video runtime patches.")
