# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
from collections import defaultdict
from io import BytesIO
from numbers import Real
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils import vision_process as qwen_vl_vision_process
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def _ensure_frame_list(frames: Any) -> list[Any]:
    def _is_video_metadata_dict(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        metadata_keys = {"fps", "video_fps", "sample_fps", "avg_fps", "average_fps", "frames_indices", "total_num_frames", "duration", "video_backend", "do_sample_frames"}
        frame_keys = {"bytes", "path", "image", "frames", "video", "images", "pixel_values_videos"}
        return len(set(value.keys()) & metadata_keys) > 0 and len(set(value.keys()) & frame_keys) == 0

    if isinstance(frames, torch.Tensor):
        return [frame for frame in frames]
    if isinstance(frames, np.ndarray):
        if frames.ndim >= 4:
            return [frame for frame in frames]
        return list(frames)
    if isinstance(frames, dict):
        if any(key in frames for key in ("bytes", "path", "image")):
            return [frames]
        if "frames" in frames:
            return _ensure_frame_list(frames["frames"])
        if "video" in frames:
            return _ensure_frame_list(frames["video"])
        if "images" in frames:
            return _ensure_frame_list(frames["images"])
        raise TypeError(f"Unsupported video container dict keys: {sorted(frames.keys())}")
    if isinstance(frames, (list, tuple)):
        normalized_frames: list[Any] = []
        for item in frames:
            if _is_video_metadata_dict(item):
                continue
            if isinstance(item, torch.Tensor) and item.ndim >= 4:
                normalized_frames.extend([frame for frame in item])
            elif isinstance(item, np.ndarray) and item.ndim >= 4:
                normalized_frames.extend([frame for frame in item])
            elif isinstance(item, (list, tuple)):
                normalized_frames.extend(_ensure_frame_list(item))
            else:
                normalized_frames.append(item)
        return normalized_frames
    return list(frames)


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return float(value.item())
    if isinstance(value, np.ndarray) and value.size == 1:
        return float(value.reshape(-1)[0])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_duration_list(value: Any, count: int) -> list[Optional[float]]:
    if count <= 0:
        return []

    scalar_value = _coerce_optional_float(value)
    if scalar_value is not None:
        return [scalar_value] * count

    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif hasattr(value, "tolist") and not isinstance(value, (list, tuple, dict)):
        value = value.tolist()

    if not isinstance(value, (list, tuple)):
        return [None] * count

    normalized = [_coerce_optional_float(item) for item in value]
    if len(normalized) == 1:
        return normalized * count
    if len(normalized) >= count:
        return normalized[:count]
    return normalized + [None] * (count - len(normalized))


def _video_durations_from_example(example: dict[str, Any], count: int) -> list[Optional[float]]:
    for key in ("video_duration_sec", "video_duration", "duration"):
        if key in example:
            durations = _normalize_duration_list(example.get(key), count)
            if any(duration is not None for duration in durations):
                return durations
    return [None] * count


def _metadata_to_dict(metadata: Any) -> Optional[dict[str, Any]]:
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        return dict(metadata)

    normalized: dict[str, Any] = {}
    for key in ("fps", "video_fps", "sample_fps", "avg_fps", "average_fps", "frames_indices", "total_num_frames", "duration", "video_backend"):
        if hasattr(metadata, key):
            normalized[key] = getattr(metadata, key)
    return normalized or None


def _metadata_get(metadata: Optional[dict[str, Any]], key: str, default: Any = None) -> Any:
    if not metadata:
        return default
    return metadata.get(key, default)


def _normalize_frame_indices(frame_indices: Any) -> Optional[list[int]]:
    if frame_indices is None:
        return None
    if isinstance(frame_indices, torch.Tensor):
        frame_indices = frame_indices.detach().cpu().reshape(-1).tolist()
    elif isinstance(frame_indices, np.ndarray):
        frame_indices = frame_indices.reshape(-1).tolist()
    elif hasattr(frame_indices, "tolist") and not isinstance(frame_indices, (list, tuple)):
        frame_indices = frame_indices.tolist()
    if not isinstance(frame_indices, (list, tuple)):
        return None

    normalized: list[int] = []
    for index in frame_indices:
        scalar = _coerce_optional_float(index)
        if scalar is None:
            return None
        normalized.append(int(round(scalar)))
    return normalized


def _slice_video_metadata(metadata: Optional[dict[str, Any]], selected_indices: np.ndarray) -> Optional[dict[str, Any]]:
    if metadata is None:
        return None

    sliced = dict(metadata)
    frame_indices = _normalize_frame_indices(sliced.get("frames_indices"))
    max_selected_index = int(selected_indices.max()) if selected_indices.size > 0 else -1
    if frame_indices is not None and len(frame_indices) >= max_selected_index + 1:
        sliced["frames_indices"] = [frame_indices[int(index)] for index in selected_indices]
    else:
        sliced["frames_indices"] = [int(index) for index in selected_indices]
    return sliced


def _finalize_video_metadata(
    metadata: Optional[dict[str, Any]],
    *,
    frame_count: int,
    fallback_fps: Optional[float],
    video_duration_sec: Optional[float],
) -> dict[str, Any]:
    normalized = dict(metadata or {})

    fps = _coerce_optional_float(
        normalized.get(
            "fps",
            normalized.get(
                "video_fps",
                normalized.get("sample_fps", normalized.get("avg_fps", normalized.get("average_fps"))),
            ),
        )
    )
    if fps is None or fps <= 0:
        fps = fallback_fps if fallback_fps is not None and fallback_fps > 0 else 1.0
    normalized["fps"] = float(fps)

    frame_indices = _normalize_frame_indices(normalized.get("frames_indices"))
    if frame_indices is None or len(frame_indices) != frame_count:
        frame_indices = list(range(frame_count))
    normalized["frames_indices"] = frame_indices

    total_num_frames = _coerce_optional_float(normalized.get("total_num_frames"))
    if total_num_frames is None or total_num_frames <= 0:
        if video_duration_sec is not None and video_duration_sec > 0:
            total_num_frames = video_duration_sec * float(fps)
        else:
            total_num_frames = max(frame_count, 1)
    normalized["total_num_frames"] = total_num_frames

    if video_duration_sec is not None and video_duration_sec > 0:
        normalized["duration"] = float(video_duration_sec)

    return normalized


def _parse_ground_truth_timespans(value: Any) -> Optional[dict[str, list[float]]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None

    if isinstance(value, dict):
        value = value.get("time_gt", value.get("segments"))

    if not isinstance(value, list):
        return None

    starts: list[float] = []
    ends: list[float] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        start = float(item[0])
        end = float(item[1])
        if end < start:
            start, end = end, start
        starts.append(start)
        ends.append(end)

    if len(starts) == 0:
        return None

    return {"start": starts, "end": ends}


def _shift_token_positions_after_postprocess(
    positions: Any,
    *,
    original_length: int,
    processed_length: int,
    truncation: str,
    left_pad: bool,
) -> list[int]:
    if positions is None:
        return []
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().tolist()
    elif isinstance(positions, np.ndarray):
        positions = positions.tolist()
    if isinstance(positions, tuple):
        positions = list(positions)
    if not isinstance(positions, list):
        positions = [positions]

    removed_left = 0
    if original_length > processed_length and truncation == "left":
        removed_left = original_length - processed_length
    pad_left = processed_length - original_length if left_pad and original_length < processed_length else 0

    shifted_positions: list[int] = []
    for position in positions:
        if isinstance(position, (list, tuple)):
            if len(position) != 1:
                continue
            position = position[0]
        try:
            shifted_position = int(position) - removed_left + pad_left
        except Exception:
            continue
        if 0 <= shifted_position < processed_length:
            shifted_positions.append(shifted_position)
    return shifted_positions


def _to_pil_video_frame(frame: Any) -> ImageObject:
    if isinstance(frame, ImageObject):
        if frame.mode != "RGB":
            frame = frame.convert("RGB")
        return frame

    if isinstance(frame, str):
        image = Image.open(frame)
        image.load()
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    if isinstance(frame, dict):
        if "image" in frame:
            return _to_pil_video_frame(frame["image"])
        if "bytes" in frame and frame["bytes"] is not None:
            image = Image.open(BytesIO(frame["bytes"]))
            image.load()
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image
        if "path" in frame and frame["path"]:
            image = Image.open(frame["path"])
            image.load()
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image
        raise TypeError(f"Unsupported video frame dict keys: {sorted(frame.keys())}")

    if isinstance(frame, (bytes, bytearray)):
        image = Image.open(BytesIO(frame))
        image.load()
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu()
        while frame.ndim > 3 and frame.shape[0] == 1:
            frame = frame.squeeze(0)
        while frame.ndim > 3 and frame.shape[-1] == 1:
            frame = frame.squeeze(-1)
        if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
            frame = frame.permute(1, 2, 0)
        frame = frame.numpy()

    if isinstance(frame, np.ndarray):
        while frame.ndim > 3 and frame.shape[0] == 1:
            frame = np.squeeze(frame, axis=0)
        while frame.ndim > 3 and frame.shape[-1] == 1:
            frame = np.squeeze(frame, axis=-1)
        if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
            frame = np.transpose(frame, (1, 2, 0))
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[..., 0]
        if frame.ndim not in (2, 3):
            raise TypeError(f"Unsupported normalized video frame shape: {frame.shape}")
        if frame.dtype != np.uint8:
            frame = frame.astype(np.float32)
            if frame.size > 0 and frame.min() >= 0.0 and frame.max() <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
        image = Image.fromarray(frame)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    raise TypeError(f"Unsupported video frame type: {type(frame)!r}")


def _fetch_video_without_resize(
    video: Union[str, list[Any], tuple[Any, ...]],
    video_fps: float,
    fps_max_frames: Optional[int],
    return_fps: bool = False,
    return_metadata: bool = False,
) -> Union[Any, tuple[Any, Optional[float]], tuple[Any, Optional[dict[str, Any]]], tuple[Any, Optional[float], Optional[dict[str, Any]]]]:
    def _is_video_metadata(value: Any) -> bool:
        if isinstance(value, dict):
            return any(
                key in value
                for key in ("fps", "video_fps", "sample_fps", "avg_fps", "average_fps", "frames_indices", "total_num_frames", "duration", "video_backend", "do_sample_frames")
            )
        return any(hasattr(value, key) for key in ("fps", "video_fps", "sample_fps", "avg_fps", "average_fps", "frames_indices", "total_num_frames", "duration", "video_backend", "do_sample_frames"))

    def _extract_fps_from_metadata(value: Any) -> Optional[float]:
        if isinstance(value, dict):
            for key in ("fps", "video_fps", "sample_fps", "avg_fps", "average_fps"):
                if key in value:
                    return _coerce_scalar_fps(value[key])
            return None
        for key in ("fps", "video_fps", "sample_fps", "avg_fps", "average_fps"):
            if hasattr(value, key):
                return _coerce_scalar_fps(getattr(value, key))
        return None

    def _coerce_scalar_fps(value: Any) -> Optional[float]:
        if isinstance(value, Real):
            return float(value)
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return float(value.item())
        if isinstance(value, np.ndarray) and value.size == 1:
            return float(value.reshape(-1)[0])
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _normalize_video_inputs(
        video_inputs: Any,
        fallback_fps: Optional[float],
    ) -> tuple[Any, Optional[float], Optional[dict[str, Any]]]:
        if isinstance(video_inputs, (list, tuple)) and len(video_inputs) == 3:
            maybe_fps = _coerce_scalar_fps(video_inputs[2])
            if maybe_fps is not None and _is_video_metadata(video_inputs[1]):
                return video_inputs[0], maybe_fps, _metadata_to_dict(video_inputs[1])
        if isinstance(video_inputs, (list, tuple)) and len(video_inputs) == 2:
            maybe_fps = _coerce_scalar_fps(video_inputs[1])
            if maybe_fps is not None:
                frames = video_inputs[0]
                metadata = None
                if isinstance(frames, (list, tuple)) and len(frames) == 2 and _is_video_metadata(frames[1]):
                    frames, metadata = frames[0], frames[1]
                return frames, maybe_fps, _metadata_to_dict(metadata)
            maybe_fps = _extract_fps_from_metadata(video_inputs[1])
            if maybe_fps is not None:
                return video_inputs[0], maybe_fps, _metadata_to_dict(video_inputs[1])
            if _is_video_metadata(video_inputs[1]):
                return video_inputs[0], fallback_fps, _metadata_to_dict(video_inputs[1])
        if isinstance(video_inputs, dict):
            maybe_fps = _extract_fps_from_metadata(video_inputs)
            for key in ("frames", "video", "images", "pixel_values_videos"):
                if key in video_inputs:
                    return video_inputs[key], maybe_fps if maybe_fps is not None else fallback_fps, _metadata_to_dict(video_inputs)
        return video_inputs, fallback_fps, None

    if not isinstance(video, str):
        if return_fps and return_metadata:
            return video, video_fps, None
        if return_metadata:
            return video, None
        if return_fps:
            return video, video_fps
        return video

    vision_info = {"video": video, "fps": video_fps}
    if fps_max_frames is not None:
        vision_info["max_frames"] = fps_max_frames

    reader_backends = getattr(qwen_vl_vision_process, "VIDEO_READER_BACKENDS", None)
    get_reader_backend = getattr(qwen_vl_vision_process, "get_video_reader_backend", None)
    if not isinstance(reader_backends, dict) or get_reader_backend is None:
        fallback_vision_info = {
            **vision_info,
            "min_pixels": 1,
            "max_pixels": 2147483647,
        }
        video_inputs = fetch_video(
            fallback_vision_info,
            return_video_sample_fps=return_fps,
            return_video_metadata=return_metadata,
        )
        frames, sampled_fps, metadata = _normalize_video_inputs(video_inputs, fallback_fps=video_fps)
        if return_fps and return_metadata:
            return frames, sampled_fps, metadata
        if return_metadata:
            return frames, metadata
        if return_fps:
            return frames, sampled_fps
        return frames

    video_reader_backend = get_reader_backend()
    try:
        video_inputs = reader_backends[video_reader_backend](vision_info)
    except Exception:
        video_inputs = reader_backends["torchvision"](vision_info)

    frames, sampled_fps, metadata = _normalize_video_inputs(video_inputs, fallback_fps=video_fps)
    if return_fps and return_metadata:
        return frames, sampled_fps, metadata
    if return_metadata:
        return frames, metadata
    if return_fps:
        return frames, sampled_fps

    return frames


def process_video(
    video: Union[str, list[Any], tuple[Any, ...]],
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    video_fps: float,
    return_fps: bool = False,
    return_metadata: bool = False,
    video_max_token_num: Optional[int] = None,
    fps_max_frames: Optional[int] = None,
    video_duration_sec: Optional[float] = None,
) -> Union[
    list[ImageObject],
    tuple[list[ImageObject], Optional[float]],
    tuple[list[ImageObject], dict[str, Any]],
    tuple[list[ImageObject], Optional[float], dict[str, Any]],
]:
    use_token_budget_resize = video_max_token_num is not None and video_max_token_num > 0

    if use_token_budget_resize:
        video_inputs = _fetch_video_without_resize(
            video=video,
            video_fps=video_fps,
            fps_max_frames=fps_max_frames,
            return_fps=return_fps,
            return_metadata=return_metadata,
        )
    else:
        vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
        if fps_max_frames is not None:
            vision_info["max_frames"] = fps_max_frames
        video_inputs = fetch_video(
            vision_info,
            return_video_sample_fps=return_fps,
            return_video_metadata=return_metadata,
        )

    metadata = None
    if return_fps and return_metadata:
        if isinstance(video_inputs, (list, tuple)) and len(video_inputs) == 3:
            frames, sampled_fps, metadata = video_inputs
        elif isinstance(video_inputs, (list, tuple)) and len(video_inputs) == 2:
            maybe_frames, sampled_fps = video_inputs
            if isinstance(maybe_frames, (list, tuple)) and len(maybe_frames) == 2:
                frames, metadata = maybe_frames
            else:
                frames = maybe_frames
        else:
            frames = video_inputs
            sampled_fps = None
    elif return_fps:
        frames, sampled_fps = video_inputs
    elif return_metadata:
        frames, metadata = video_inputs
        sampled_fps = None
    else:
        frames = video_inputs
        sampled_fps = None

    metadata = _metadata_to_dict(metadata)
    frames = _ensure_frame_list(frames)

    if fps_max_frames is not None and len(frames) > fps_max_frames:
        original_num_frames = len(frames)
        frame_indices = np.linspace(0, original_num_frames - 1, fps_max_frames).round().astype(int)
        frames = [frames[int(index)] for index in frame_indices]
        metadata = _slice_video_metadata(metadata, frame_indices)
        if sampled_fps is not None and original_num_frames > 0:
            sampled_fps = sampled_fps * len(frames) / original_num_frames

    if len(frames) > 0:
        frames = [_to_pil_video_frame(frame) for frame in frames]

    effective_min_pixels = None if use_token_budget_resize else min_pixels
    effective_max_pixels = None if use_token_budget_resize else max_pixels
    if video_max_token_num is not None and video_max_token_num > 0 and len(frames) > 0:
        factor = 16 * 2
        temporal_patch_size = 2
        aligned_num_frames = max(temporal_patch_size, round(len(frames) / temporal_patch_size) * temporal_patch_size)
        effective_max_pixels = int(video_max_token_num * (factor**2) * aligned_num_frames)

    if len(frames) > 0 and (effective_min_pixels is not None or effective_max_pixels is not None):
        height, width = frames[0].height, frames[0].width
        num_frames = len(frames)
        temporal_factor = 2
        resize_factor = 32
        min_total_pixels = effective_min_pixels
        max_total_pixels = effective_max_pixels

        if (
            min_total_pixels is not None
            and max_total_pixels is not None
            and min_total_pixels > max_total_pixels
        ):
            raise ValueError(
                f"video min_pixels ({min_total_pixels}) is greater than effective max_pixels ({max_total_pixels})"
            )

        resized_height = max(resize_factor, round(height / resize_factor) * resize_factor)
        resized_width = max(resize_factor, round(width / resize_factor) * resize_factor)
        aligned_num_frames = max(temporal_factor, round(num_frames / temporal_factor) * temporal_factor)
        current_total_pixels = aligned_num_frames * resized_height * resized_width

        if max_total_pixels is not None and current_total_pixels > max_total_pixels:
            beta = math.sqrt((num_frames * height * width) / max_total_pixels)
            resized_height = max(resize_factor, math.floor(height / beta / resize_factor) * resize_factor)
            resized_width = max(resize_factor, math.floor(width / beta / resize_factor) * resize_factor)
        elif min_total_pixels is not None and current_total_pixels < min_total_pixels:
            beta = math.sqrt(min_total_pixels / (num_frames * height * width))
            resized_height = max(resize_factor, math.ceil(height * beta / resize_factor) * resize_factor)
            resized_width = max(resize_factor, math.ceil(width * beta / resize_factor) * resize_factor)

        if resized_height != height or resized_width != width:
            frames = [frame.resize((resized_width, resized_height)) for frame in frames]

    if return_metadata:
        metadata = _finalize_video_metadata(
            metadata,
            frame_count=len(frames),
            fallback_fps=sampled_fps if sampled_fps is not None else video_fps,
            video_duration_sec=video_duration_sec,
        )

    if return_fps and return_metadata:
        return frames, sampled_fps, metadata
    if return_metadata:
        return frames, metadata
    if return_fps:
        return frames, sampled_fps
    return frames

class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        video_max_token_num: Optional[int] = None,
        fps_max_frames: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.video_max_token_num = video_max_token_num
        self.fps_max_frames = fps_max_frames

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_metadata_list = []
            video_durations = _video_durations_from_example(example, len(videos))
            for video_idx, video in enumerate(videos):
                processed_video, video_metadata = process_video(
                    video,
                    self.min_pixels,
                    self.max_pixels,
                    self.video_fps,
                    return_metadata=True,
                    video_max_token_num=self.video_max_token_num,
                    fps_max_frames=self.fps_max_frames,
                    video_duration_sec=video_durations[video_idx],
                )
                processed_videos.append(processed_video)
                video_metadata_list.append(video_metadata)

            processor_kwargs = {
                "videos": processed_videos,
                "text": [prompt],
                "add_special_tokens": False,
                "return_tensors": "pt",
            }
            if self.video_max_token_num is not None or self.fps_max_frames is not None:
                videos_kwargs = {}
                if self.video_max_token_num is not None:
                    videos_kwargs["frame_max_token"] = self.video_max_token_num
                    videos_kwargs["frame_token_only"] = True
                if self.fps_max_frames is not None:
                    videos_kwargs["max_frames"] = self.fps_max_frames
                processor_kwargs["videos_kwargs"] = videos_kwargs
            if video_metadata_list:
                videos_kwargs = processor_kwargs.setdefault("videos_kwargs", {})
                videos_kwargs["video_metadata"] = video_metadata_list
                videos_kwargs["do_sample_frames"] = False

            model_inputs = self.processor(**processor_kwargs)
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)
        timespan_video_duration_override: Optional[float] = None

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            video_metadata_list = []
            video_durations = _video_durations_from_example(example, len(videos))
            if len(video_durations) == 1 and video_durations[0] is not None:
                timespan_video_duration_override = float(video_durations[0])
            for video_idx, video in enumerate(videos):
                processed_video, video_fps, video_metadata = process_video(
                    video,
                    self.min_pixels,
                    self.max_pixels,
                    self.video_fps,
                    return_fps=True,
                    return_metadata=True,
                    video_max_token_num=self.video_max_token_num,
                    fps_max_frames=self.fps_max_frames,
                    video_duration_sec=video_durations[video_idx],
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)
                video_metadata_list.append(video_metadata)

            processor_kwargs = {
                "videos": processed_videos,
                "text": [prompt],
                "add_special_tokens": False,
                "return_tensors": "pt",
            }
            if (
                getattr(self.processor, "use_time_codec", False)
                or getattr(self.processor, "use_cis_codec", False)
                or getattr(self.processor, "use_timeple_codec", False)
                or getattr(self.processor, "use_timeple_codec", False)
                or getattr(self.processor, "use_timeed", False)
            ):
                processor_kwargs["return_time_labels"] = True
            if self.video_max_token_num is not None or self.fps_max_frames is not None:
                videos_kwargs = {}
                if self.video_max_token_num is not None:
                    videos_kwargs["frame_max_token"] = self.video_max_token_num
                    videos_kwargs["frame_token_only"] = True
                if self.fps_max_frames is not None:
                    videos_kwargs["max_frames"] = self.fps_max_frames
                processor_kwargs["videos_kwargs"] = videos_kwargs
            if video_metadata_list:
                videos_kwargs = processor_kwargs.setdefault("videos_kwargs", {})
                videos_kwargs["video_metadata"] = video_metadata_list
                videos_kwargs["do_sample_frames"] = False

            model_inputs = self.processor(**processor_kwargs)
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            timestamp_labels = model_inputs.pop("timestamp_labels", None)
            timestamp_positions = model_inputs.pop("timestamp_positions", None)
            timestamp_video_durations = model_inputs.pop("timestamp_video_durations", None)
            if timestamp_labels is not None and timestamp_positions is not None:
                ts_start = timestamp_labels.get("start", [])
                ts_end = timestamp_labels.get("end", [])
                ts_video_duration = timestamp_labels.get("video_duration", timestamp_video_durations)
                if len(ts_start) > 0 and isinstance(ts_start[0], list):
                    ts_start = ts_start[0]
                if len(ts_end) > 0 and isinstance(ts_end[0], list):
                    ts_end = ts_end[0]
                if ts_video_duration is not None and len(ts_video_duration) > 0 and isinstance(ts_video_duration[0], list):
                    ts_video_duration = ts_video_duration[0]
                if timespan_video_duration_override is not None:
                    ts_video_duration = [timespan_video_duration_override] * len(ts_start)
                example["timestamp_labels"] = {"start": ts_start, "end": ts_end}
                if ts_video_duration is not None:
                    example["timestamp_labels"]["video_duration"] = ts_video_duration
                    example["timestamp_video_durations"] = ts_video_duration
                example["timestamp_positions"] = (
                    timestamp_positions[0] if len(timestamp_positions) > 0 else timestamp_positions
                )
            example["multi_modal_data"] = {"videos": videos}
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        original_prompt_length = len(input_ids)
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        if "timestamp_positions" in example:
            example["timestamp_positions"] = _shift_token_positions_after_postprocess(
                example["timestamp_positions"],
                original_length=original_prompt_length,
                processed_length=input_ids.size(-1),
                truncation=self.truncation,
                left_pad=True,
            )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        ground_truth = example.pop(self.answer_key)
        timespan_labels = _parse_ground_truth_timespans(ground_truth)
        timestamp_video_duration = None
        if isinstance(example.get("timestamp_labels"), dict):
            timestamp_video_duration = example["timestamp_labels"].get("video_duration")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        if timespan_labels is not None:
            target_count = len(timespan_labels.get("start", []))
            if timespan_video_duration_override is not None and target_count > 0:
                timestamp_video_duration = [timespan_video_duration_override] * target_count
            elif timestamp_video_duration is not None and target_count > 0:
                duration_scalar = None
                if isinstance(timestamp_video_duration, (list, tuple)) and len(timestamp_video_duration) > 0:
                    duration_scalar = _coerce_optional_float(timestamp_video_duration[0])
                else:
                    duration_scalar = _coerce_optional_float(timestamp_video_duration)
                if duration_scalar is not None:
                    timestamp_video_duration = [duration_scalar] * target_count
            if timestamp_video_duration is not None:
                timespan_labels["video_duration"] = timestamp_video_duration
                example["timespan_video_durations"] = timestamp_video_duration
            example["timespan_labels"] = timespan_labels
        example["ground_truth"] = ground_truth
        return example
