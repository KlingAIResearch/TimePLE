# Copyright (c) Alibaba, Inc. and its affiliates.
"""TimePLE training dataset utilities for Qwen3-VL."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

try:
    from transformers.models.qwen3_vl.processing_qwen3_vl_timeple import Qwen3VLProcessorWithTimePLECodec
except Exception:  # pragma: no cover - handled by caller if missing
    Qwen3VLProcessorWithTimePLECodec = None  # type: ignore


TIMESPAN_TOKEN = "<|TIMESPAN|>"
IGNORE_INDEX = -100


def _set_module_attr(module, name: str, value):
    if not hasattr(module, name):
        return None
    old_value = getattr(module, name)
    setattr(module, name, value)
    return old_value


def resolve_annotation_path(dataset_path: str, annotation_file: Optional[str]) -> str:
    """Resolve the JSONL annotation file path from a dataset root.

    Priority:
      1) explicit annotation_file (absolute or relative to dataset_path)
      2) dataset_path itself if it is a .jsonl file
      3) dataset_path/anno/*.jsonl (single file expected)
      4) dataset_path/*.jsonl (single file expected)
    """
    if annotation_file:
        if os.path.isabs(annotation_file):
            return annotation_file
        return os.path.join(dataset_path, annotation_file)

    if os.path.isfile(dataset_path) and dataset_path.lower().endswith(".jsonl"):
        return dataset_path

    candidates: List[str] = []
    anno_dir = os.path.join(dataset_path, "anno")
    if os.path.isdir(anno_dir):
        candidates.extend(
            [os.path.join(anno_dir, f) for f in os.listdir(anno_dir) if f.lower().endswith(".jsonl")]
        )
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple .jsonl files found under {anno_dir}: {sorted(candidates)}. "
                "Please pass --annotation_file explicitly."
            )

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    root_candidates = [
        os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.lower().endswith(".jsonl")
    ]
    if len(root_candidates) == 1:
        return root_candidates[0]
    if len(root_candidates) > 1:
        raise ValueError(
            f"Multiple .jsonl files found under {dataset_path}: {sorted(root_candidates)}. "
            "Please pass --annotation_file explicitly."
        )

    raise FileNotFoundError(
        "Unable to locate a .jsonl annotation file. Provide --annotation_file or place one under "
        f"{anno_dir} or {dataset_path}."
    )


def _load_jsonl_samples(
    annotation_path: str,
    start_idx: int,
    end_idx: Optional[int],
) -> List[Dict]:
    samples: List[Dict] = []
    with open(annotation_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start_idx:
                continue
            if end_idx is not None and idx >= end_idx:
                break
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            sample["_line_idx"] = idx
            samples.append(sample)
    return samples


def _count_timespan_tokens(text: str) -> int:
    return text.count(TIMESPAN_TOKEN)


def _extract_message_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return str(content)


def _build_labels_from_im_start_end(
    input_ids: torch.Tensor,
    tokenizer,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Build causal-LM labels by keeping only assistant turns."""
    if input_ids.ndim != 1:
        raise ValueError(f"Expected 1D input_ids, got shape={tuple(input_ids.shape)}")

    im_start = "<|im_start|>"
    im_end = "<|im_end|>"

    im_start_id = tokenizer.convert_tokens_to_ids(im_start)
    im_end_id = tokenizer.convert_tokens_to_ids(im_end)

    assistant_role_ids = tokenizer("assistant", add_special_tokens=False).input_ids
    if not assistant_role_ids:
        raise ValueError("Failed to tokenize role string 'assistant'.")

    ids = input_ids.tolist()
    labels = torch.full_like(input_ids, ignore_index)

    i = 0
    L = len(ids)
    while i < L:
        if ids[i] != im_start_id:
            i += 1
            continue

        role_start = i + 1
        role_end = role_start + len(assistant_role_ids)
        if role_end <= L and ids[role_start:role_end] == assistant_role_ids:
            content_start = role_end
            j = content_start
            while j < L and ids[j] != im_end_id:
                j += 1
            if j < L and ids[j] == im_end_id:
                labels[content_start : j + 1] = input_ids[content_start : j + 1]
                i = j + 1
                continue

        i += 1

    return labels


class TimePLECodecTrainDataset(Dataset):
    def __init__(
        self,
        processor: "Qwen3VLProcessorWithTimePLECodec",
        dataset_path: str,
        video_base_dir: str,
        annotation_file: Optional[str],
        start_idx: int,
        end_idx: Optional[int],
        sample_fps: float,
        video_load_backend: Optional[str],
        max_frames: int,
        min_frames: int,
        total_pixels: Optional[int],
        min_pixels: Optional[int],
        max_timespans_per_sample: int,
        frame_min_token: Optional[int] = None,
        frame_max_token: Optional[int] = None,
        frame_token_only: bool = False,
        use_cot_thinking: bool = False,
        ignore_index: int = IGNORE_INDEX,
    ):
        if Qwen3VLProcessorWithTimePLECodec is None:
            raise ImportError(
                "Qwen3VLProcessorWithTimePLECodec is not available. "
                "Make sure the Qwen3-VL TimePLE transformers modules are on PYTHONPATH."
            )
        super().__init__()
        self.processor = processor
        self.video_base_dir = video_base_dir
        self.max_timespans_per_sample = max_timespans_per_sample
        self.use_cot_thinking = use_cot_thinking
        self.ignore_index = ignore_index
        self._max_pixels = total_pixels
        self._min_pixels = min_pixels
        self._video_kwargs_supported: Optional[set] = None
        if getattr(processor, "video_processor", None) is not None:
            vp = processor.video_processor
            supported = set()
            if hasattr(vp, "model_valid_processing_keys"):
                supported.update(vp.model_valid_processing_keys)
            elif hasattr(vp, "valid_kwargs") and hasattr(vp.valid_kwargs, "__annotations__"):
                supported.update(vp.valid_kwargs.__annotations__.keys())
            self._video_kwargs_supported = supported or None

        self.annotation_path = resolve_annotation_path(dataset_path, annotation_file)
        self._samples = _load_jsonl_samples(self.annotation_path, start_idx, end_idx)

        self._videos_kwargs = {
            "do_sample_frames": True,
            "fps": sample_fps,
            "min_frames": min_frames,
            "max_frames": max_frames,
        }
        if self._min_pixels is not None and self._max_pixels is not None:
            self._videos_kwargs["size"] = {"shortest_edge": self._min_pixels, "longest_edge": self._max_pixels}
        self._video_load_backend = video_load_backend
        if self._video_load_backend and getattr(self.processor, "video_processor", None) is not None:
            vp = self.processor.video_processor
            setattr(vp, "video_load_backend", self._video_load_backend)
        if frame_min_token is not None:
            self._videos_kwargs["frame_min_token"] = int(frame_min_token)
        if frame_max_token is not None:
            self._videos_kwargs["frame_max_token"] = int(frame_max_token)
        if frame_token_only:
            self._videos_kwargs["frame_token_only"] = True

    def __len__(self) -> int:
        return len(self._samples)

    def _resolve_video_path(self, video_path: str) -> str:
        if os.path.isabs(video_path) or not self.video_base_dir:
            return video_path
        return os.path.join(self.video_base_dir, video_path)

    def _normalize_sample_schema(self, sample: Dict) -> Dict:
        if "video_path" in sample and "query" in sample and "answer" in sample:
            return sample

        videos = sample.get("videos")
        messages = sample.get("messages")
        if videos is None or messages is None:
            return sample

        if not isinstance(videos, (list, tuple)) or len(videos) != 1:
            raise ValueError(
                "TimePLECodecTrainDataset currently expects exactly one video per sample when using official schema."
            )

        user_message = None
        assistant_message = None
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "user" and user_message is None:
                user_message = message
            elif role == "assistant" and assistant_message is None:
                assistant_message = message

        if user_message is None or assistant_message is None:
            raise ValueError("Official-format sample must include one user message and one assistant message.")

        query = _extract_message_text(user_message.get("content"))
        answer = _extract_message_text(assistant_message.get("content"))
        if query.startswith("<video>"):
            query = query[len("<video>") :].lstrip()

        normalized = dict(sample)
        normalized["video_path"] = videos[0]
        normalized["query"] = query
        normalized["answer"] = answer
        return normalized

    def _fetch_video_with_qwen_vl_utils(self, video_path: str):
        from qwen_vl_utils import fetch_video, vision_process

        # Align with the current official ms-swift Qwen3-VL path: use
        # qwen_vl_utils.fetch_video(...) for decode, temporal sampling and
        # spatial budget control, then pass the returned video tensor together
        # with its metadata straight into the processor with
        # do_sample_frames=False.
        restore = {}
        try:
            if self._video_load_backend:
                old_value = _set_module_attr(vision_process, "FORCE_QWENVL_VIDEO_READER", self._video_load_backend)
                if old_value is not None:
                    restore["FORCE_QWENVL_VIDEO_READER"] = old_value
                backend_getter = getattr(vision_process, "get_video_reader_backend", None)
                cache_clear = getattr(backend_getter, "cache_clear", None)
                if callable(cache_clear):
                    cache_clear()

            for module_name, kwarg_name in (
                ("FPS", "fps"),
                ("FPS_MIN_FRAMES", "min_frames"),
                ("FPS_MAX_FRAMES", "max_frames"),
            ):
                value = self._videos_kwargs.get(kwarg_name)
                if value is None:
                    continue
                old_value = _set_module_attr(vision_process, module_name, value)
                if old_value is not None:
                    restore[module_name] = old_value

            if self._videos_kwargs.get("frame_token_only"):
                frame_min_token = self._videos_kwargs.get("frame_min_token")
                frame_max_token = self._videos_kwargs.get("frame_max_token")
                if frame_min_token is not None:
                    old_value = _set_module_attr(vision_process, "VIDEO_MIN_TOKEN_NUM", int(frame_min_token))
                    if old_value is not None:
                        restore["VIDEO_MIN_TOKEN_NUM"] = old_value
                if frame_max_token is not None:
                    old_value = _set_module_attr(vision_process, "VIDEO_MAX_TOKEN_NUM", int(frame_max_token))
                    if old_value is not None:
                        restore["VIDEO_MAX_TOKEN_NUM"] = old_value

            video_ele = {"video": video_path}
            if self._min_pixels is not None:
                video_ele["min_pixels"] = int(self._min_pixels)
            if self._max_pixels is not None:
                video_ele["total_pixels"] = int(self._max_pixels)

            (video, video_metadata), _ = fetch_video(
                video_ele,
                return_video_sample_fps=True,
                return_video_metadata=True,
                image_patch_size=self.processor.image_processor.patch_size,
            )
        finally:
            for name, value in restore.items():
                setattr(vision_process, name, value)
            backend_getter = getattr(vision_process, "get_video_reader_backend", None)
            cache_clear = getattr(backend_getter, "cache_clear", None)
            if callable(cache_clear):
                cache_clear()

        if isinstance(video, torch.Tensor):
            video = video.to(torch.uint8)
        return video, video_metadata

    def __getitem__(self, idx: int) -> Dict:
        sample = self._normalize_sample_schema(self._samples[idx])

        if "video_path" not in sample:
            raise KeyError("Missing required field 'video_path' in training sample.")
        if "query" not in sample:
            raise KeyError("Missing required field 'query' in training sample.")
        if "answer" not in sample:
            raise KeyError("Missing required field 'answer' in training sample.")
        if "time_gt" not in sample:
            raise KeyError("Missing required field 'time_gt' in training sample.")

        video_path = self._resolve_video_path(sample["video_path"])
        query = sample["query"]
        answer = sample["answer"]

        if self.use_cot_thinking:
            cot = sample.get("cot_thinking")
            if cot:
                answer = f"{cot}\n{answer}"

        time_gt = sample["time_gt"]
        if not isinstance(time_gt, (list, tuple)):
            raise ValueError("time_gt must be a list of [start, end] pairs.")

        gt_segments: List[Tuple[float, float]] = []
        for seg in time_gt:
            if not isinstance(seg, (list, tuple)) or len(seg) != 2:
                raise ValueError("Each time_gt entry must be a [start, end] pair.")
            gt_segments.append((float(seg[0]), float(seg[1])))

        if self.max_timespans_per_sample > 0:
            gt_segments = gt_segments[: self.max_timespans_per_sample]

        expected_timespans = len(gt_segments)
        actual_timespans = _count_timespan_tokens(answer)
        if actual_timespans != expected_timespans:
            raise ValueError(
                "TIMESPAN token count mismatch: "
                f"answer has {actual_timespans}, time_gt has {expected_timespans}. "
                f"line_idx={sample.get('_line_idx')}"
            )

        size_dict = self._videos_kwargs.get("size") or {}
        max_pixels = self._max_pixels if self._max_pixels is not None else size_dict.get("longest_edge")
        min_pixels = self._min_pixels if self._min_pixels is not None else size_dict.get("shortest_edge")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                    },
                    {"type": "text", "text": query},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        if max_pixels is not None:
            messages[0]["content"][0]["max_pixels"] = max_pixels
        if min_pixels is not None:
            messages[0]["content"][0]["min_pixels"] = min_pixels

        try:
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            video, video_metadata = self._fetch_video_with_qwen_vl_utils(video_path)
            features = self.processor(
                text=prompt,
                videos=[video],
                return_time_labels=True,
                return_tensors="pt",
                videos_kwargs={
                    "do_sample_frames": False,
                    "do_resize": False,
                    "return_metadata": True,
                    "video_metadata": [video_metadata],
                },
            )
        except Exception as e:
            raise type(e)(
                f"{e} | line_idx={sample.get('_line_idx')} | video_path={video_path}"
            ) from e

        input_ids = features["input_ids"].squeeze(0)
        labels = _build_labels_from_im_start_end(input_ids, self.processor.tokenizer, ignore_index=self.ignore_index)

        timestamp_labels = features.get("timestamp_labels")
        timestamp_positions = features.get("timestamp_positions")
        if timestamp_labels is None or timestamp_positions is None:
            raise ValueError("Processor did not return timestamp_labels/timestamp_positions. Check use_timeple_codec=True.")

        timestamp_labels = {
            "start": timestamp_labels["start"][0],
            "end": timestamp_labels["end"][0],
        }
        if "video_duration" in features["timestamp_labels"]:
            timestamp_labels["video_duration"] = features["timestamp_labels"]["video_duration"][0]
        timestamp_positions = timestamp_positions[0]
        expected_timestamps = len(timestamp_labels["start"])
        if len(timestamp_labels["end"]) != expected_timestamps:
            raise ValueError(
                "Timestamp label shape mismatch: "
                f"start={expected_timestamps} vs end={len(timestamp_labels['end'])}. "
                f"line_idx={sample.get('_line_idx')}"
            )
        if len(timestamp_positions) != expected_timestamps:
            raise ValueError(
                "TIMESTAMP token count mismatch after tokenization: "
                f"positions={len(timestamp_positions)} vs labels={expected_timestamps}. "
                f"line_idx={sample.get('_line_idx')}"
            )

        timespan_token_id = self.processor.tokenizer.convert_tokens_to_ids(TIMESPAN_TOKEN)
        timespan_positions = (input_ids == timespan_token_id).nonzero(as_tuple=False).flatten().tolist()
        if len(timespan_positions) != expected_timespans:
            raise ValueError(
                "TIMESPAN token count mismatch after tokenization: "
                f"positions={len(timespan_positions)} vs time_gt={expected_timespans}. "
                f"line_idx={sample.get('_line_idx')}"
            )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": features.get("attention_mask", None).squeeze(0)
            if "attention_mask" in features
            else None,
            "pixel_values_videos": features.get("pixel_values_videos"),
            "video_grid_thw": features.get("video_grid_thw"),
            "timestamp_labels": timestamp_labels,
            "timestamp_positions": timestamp_positions,
            "timespan_labels": {
                "start": [float(s) for s, _ in gt_segments],
                "end": [float(e) for _, e in gt_segments],
                "video_duration": timestamp_labels.get("video_duration"),
            },
            "timespan_positions": timespan_positions,
        }


class TimePLECodecDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: List[Dict]) -> Dict[str, object]:
        input_ids = [ex["input_ids"] for ex in instances]
        labels = [ex["labels"] for ex in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        batch: Dict[str, object] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        # Video features are concatenated across samples (Qwen3-VL expects a flat tensor across all videos).
        videos = [ex.get("pixel_values_videos") for ex in instances if ex.get("pixel_values_videos") is not None]
        grids = [ex.get("video_grid_thw") for ex in instances if ex.get("video_grid_thw") is not None]
        if videos:
            batch["pixel_values_videos"] = torch.cat(videos, dim=0)
        if grids:
            batch["video_grid_thw"] = torch.cat(grids, dim=0)

        # Time labels/positions stay as python lists (Trainer will not move them to device).
        batch["timestamp_labels"] = {
            "start": [ex["timestamp_labels"]["start"] for ex in instances],
            "end": [ex["timestamp_labels"]["end"] for ex in instances],
        }
        if any(ex["timestamp_labels"].get("video_duration") is not None for ex in instances):
            batch["timestamp_labels"]["video_duration"] = [
                ex["timestamp_labels"].get("video_duration") for ex in instances
            ]
        batch["timestamp_positions"] = [ex["timestamp_positions"] for ex in instances]

        batch["timespan_labels"] = {
            "start": [ex["timespan_labels"]["start"] for ex in instances],
            "end": [ex["timespan_labels"]["end"] for ex in instances],
        }
        if any(ex["timespan_labels"].get("video_duration") is not None for ex in instances):
            batch["timespan_labels"]["video_duration"] = [
                ex["timespan_labels"].get("video_duration") for ex in instances
            ]
        batch["timespan_positions"] = [ex["timespan_positions"] for ex in instances]

        return batch
