"""
Qwen3-VL Processor with TimePLE integration.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging
from transformers.video_utils import VideoInput

from .processing_qwen3_vl import Qwen3VLProcessor, Qwen3VLProcessorKwargs

logger = logging.get_logger(__name__)


class Qwen3VLProcessorWithTimePLECodec(Qwen3VLProcessor):
    """
    Adds dual time-token processing for TimePLE.

    Compared with the TPFD processor, this version additionally keeps track of
    the per-token video duration so the codec can work in relative time.
    """

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        use_timeple_codec: bool = True,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template, **kwargs)
        self.use_timeple_codec = use_timeple_codec
        self.timestamp_token = "<|TIMESTAMP|>"
        self.timespan_token = "<|TIMESPAN|>"
        self.timestamp_token_id = tokenizer.convert_tokens_to_ids(self.timestamp_token)
        self.timespan_token_id = tokenizer.convert_tokens_to_ids(self.timespan_token)

    def _calculate_time_segments(
        self,
        indices: Union[List[int], np.ndarray],
        video_fps: float,
        temporal_patch_size: int = 2,
    ) -> List[Tuple[float, float]]:
        if not isinstance(indices, list):
            indices = indices.tolist()

        if temporal_patch_size <= 0:
            raise ValueError(f"temporal_patch_size must be > 0, got {temporal_patch_size}")
        if not indices:
            return []
        if len(indices) % temporal_patch_size != 0:
            indices.extend(indices[-1] for _ in range(temporal_patch_size - len(indices) % temporal_patch_size))

        timestamps = [idx / video_fps for idx in indices]
        time_segments = []
        for i in range(0, len(timestamps), temporal_patch_size):
            start_time = timestamps[i]
            end_time = timestamps[min(i + temporal_patch_size - 1, len(timestamps) - 1)]
            if abs(end_time - start_time) < 0.01:
                end_time = start_time + 0.5
            time_segments.append((start_time, end_time))
        return time_segments

    @staticmethod
    def _meta_get(metadata, key, default=None):
        if isinstance(metadata, dict):
            return metadata.get(key, default)
        return getattr(metadata, key, default)

    def _resolve_video_duration(
        self,
        metadata,
        fps: float,
        frame_indices: Optional[Union[List[int], np.ndarray]],
        time_segments: List[Tuple[float, float]],
    ) -> float:
        duration = self._meta_get(metadata, "duration")
        if duration is None:
            total_num_frames = self._meta_get(metadata, "total_num_frames")
            if total_num_frames is not None and fps:
                duration = float(total_num_frames) / float(fps)
        if duration is None:
            num_frames = self._meta_get(metadata, "num_frames")
            if num_frames is not None and fps:
                duration = float(num_frames) / float(fps)
        if duration is None and frame_indices is not None and len(frame_indices) > 0 and fps:
            duration = (float(max(frame_indices)) + 1.0) / float(fps)
        if duration is None and time_segments:
            duration = time_segments[-1][1]
        if duration is None or duration <= 0:
            duration = 1.0
        return float(duration)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        return_time_labels: bool = True,
        **kwargs,
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            Qwen3VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        return_video_metadata = bool(kwargs.get("return_metadata", False))

        if videos is not None:
            is_preprocessed = (
                isinstance(videos, (list, tuple))
                and len(videos) > 0
                and isinstance(videos[0], torch.Tensor)
                and videos[0].ndim == 5
            )
            if is_preprocessed:
                videos_inputs = {
                    "pixel_values_videos": videos[0] if len(videos) == 1 else torch.stack(videos),
                }
                video_grid_thw = output_kwargs["videos_kwargs"].get("video_grid_thw")
                if video_grid_thw is None:
                    raise ValueError(
                        "When passing preprocessed video tensors, `video_grid_thw` must be provided in videos_kwargs."
                    )
                video_metadata = output_kwargs["videos_kwargs"].get("video_metadata", [])
            else:
                videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
                video_grid_thw = videos_inputs["video_grid_thw"]
                video_metadata = videos_inputs.pop("video_metadata", [])
        else:
            videos_inputs = {}
            video_grid_thw = None
            video_metadata = []

        if not isinstance(text, list):
            text = [text]
        text = text.copy()

        all_time_labels_start = []
        all_time_labels_end = []
        all_time_token_positions = []
        all_time_video_durations = []

        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            merge_length = self.video_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                sample_time_starts = []
                sample_time_ends = []
                sample_time_durations = []

                while self.video_token in text[i]:
                    metadata = video_metadata[index]
                    fps = self._meta_get(metadata, "fps")
                    raw_temporal_patch_size = getattr(self.video_processor, "temporal_patch_size", None)
                    temporal_patch_size = int(raw_temporal_patch_size) if raw_temporal_patch_size is not None else 1
                    if fps is None:
                        logger.warning_once(
                            "Qwen3VL-TimePLECodec requires video fps but couldn't infer it. Defaulting to fps=24."
                        )
                        fps = 24.0
                        if isinstance(metadata, dict):
                            metadata["fps"] = fps
                        else:
                            metadata.fps = fps

                    frames_indices = self._meta_get(metadata, "frames_indices")
                    if frames_indices is None:
                        logger.warning_once(
                            "Qwen3VL-TimePLECodec couldn't find `frames_indices`; falling back to a synthetic sequential index."
                        )
                        frames_indices = list(range(int(video_grid_thw[index][0]) * temporal_patch_size))
                    time_segments = self._calculate_time_segments(
                        frames_indices,
                        float(fps),
                        temporal_patch_size,
                    )
                    expected_time_tokens = int(video_grid_thw[index][0])
                    if len(time_segments) != expected_time_tokens:
                        raise ValueError(
                            "Timestamp segment count mismatch with video_grid_thw: "
                            f"segments={len(time_segments)} vs grid_t={expected_time_tokens}. "
                            f"fps={fps}, temporal_patch_size={temporal_patch_size}, index={index}"
                        )
                    video_duration = self._resolve_video_duration(
                        metadata=metadata,
                        fps=float(fps),
                        frame_indices=frames_indices,
                        time_segments=time_segments,
                    )

                    video_placeholder = ""
                    frame_seqlen = video_grid_thw[index][1:].prod() // merge_length
                    for frame_idx in range(video_grid_thw[index][0]):
                        start_time, end_time = time_segments[frame_idx]
                        sample_time_starts.append(start_time)
                        sample_time_ends.append(end_time)
                        sample_time_durations.append(video_duration)

                        if self.use_timeple_codec:
                            video_placeholder += self.timestamp_token
                        else:
                            curr_time = 0.5 * (start_time + end_time)
                            video_placeholder += f"<{curr_time:.1f} seconds>"
                        video_placeholder += (
                            self.vision_start_token +
                            "<|placeholder|>" * frame_seqlen +
                            self.vision_end_token
                        )

                    wrapped_video_token = f"{self.vision_start_token}{self.video_token}{self.vision_end_token}"
                    if wrapped_video_token in text[i]:
                        text[i] = text[i].replace(wrapped_video_token, video_placeholder, 1)
                    else:
                        text[i] = text[i].replace(self.video_token, video_placeholder, 1)
                    index += 1

                text[i] = text[i].replace("<|placeholder|>", self.video_token)
                if sample_time_starts:
                    all_time_labels_start.append(sample_time_starts)
                    all_time_labels_end.append(sample_time_ends)
                    all_time_video_durations.append(sample_time_durations)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if self.use_timeple_codec and return_time_labels:
            input_ids = text_inputs["input_ids"]
            for ids in input_ids:
                timestamp_positions = []
                for pos, token_id in enumerate(ids):
                    if token_id == self.timestamp_token_id:
                        timestamp_positions.append(pos)
                all_time_token_positions.append(timestamp_positions)

        if return_mm_token_type_ids:
            processor_mm_ids = (
                list(getattr(self, "image_ids", []) or [])
                + list(getattr(self, "video_ids", []) or [])
                + list(getattr(self, "audio_ids", []) or [])
            )
            if hasattr(self, "create_mm_token_type_ids") and any(token_id is not None for token_id in processor_mm_ids):
                text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])
            else:
                array_ids = np.array(text_inputs["input_ids"])
                mm_token_type_ids = np.zeros_like(array_ids)
                mm_token_type_ids[array_ids == self.image_token_id] = 1
                mm_token_type_ids[array_ids == self.video_token_id] = 2
                text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        result = BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)
        if return_video_metadata:
            result["video_metadata"] = video_metadata
        if self.use_timeple_codec and return_time_labels and all_time_labels_start:
            result["timestamp_labels"] = {
                "start": all_time_labels_start,
                "end": all_time_labels_end,
                "video_duration": all_time_video_durations,
            }
            result["timestamp_positions"] = all_time_token_positions
            result["timestamp_video_durations"] = all_time_video_durations
        return result

    def post_process_image_text_to_text(
        self,
        generated_outputs,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
        **kwargs,
    ):
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )


__all__ = ["Qwen3VLProcessorWithTimePLECodec"]
