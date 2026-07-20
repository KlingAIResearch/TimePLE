from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - runtime dependency
    torch = None

try:
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as tvF
except ImportError:  # pragma: no cover - runtime dependency
    InterpolationMode = None
    tvF = None

try:
    from decord import VideoReader, cpu
except ImportError:  # pragma: no cover - runtime dependency
    VideoReader = None
    cpu = None


QWEN_VIDEO_MIN_TOKEN_NUM = 128
QWEN_VIDEO_MAX_TOKEN_NUM = 768
@dataclass(slots=True)
class PreparedQwenVideoInput:
    video: Any
    metadata: dict[str, Any]
    frame_timestamps: list[float]
    sampled_num_frames: int
    resized_height: int
    resized_width: int
    processor_video_kwargs: dict[str, Any]


def _require_video_preprocess_dependencies() -> None:
    if VideoReader is None or cpu is None:
        raise RuntimeError(
            "Qwen official video preprocessing requires `decord==0.6.0` "
            "in the eval-suite environment."
        )
    if torch is None or tvF is None or InterpolationMode is None:
        raise RuntimeError(
            "Qwen official video preprocessing requires `torch` and `torchvision` "
            "in the eval-suite environment."
        )


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def _smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    if min(height, width) < factor:
        raise ValueError(
            f"Video frame size {height}x{width} is smaller than factor={factor}."
        )
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "Video aspect ratio exceeds the qwen_vl_utils constraint of 200."
        )

    height_bar = max(factor, _round_by_factor(height, factor))
    width_bar = max(factor, _round_by_factor(width, factor))

    if height_bar * width_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        height_bar = max(factor, _floor_by_factor(height / beta, factor))
        width_bar = max(factor, _floor_by_factor(width / beta, factor))
    elif height_bar * width_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        height_bar = max(factor, _ceil_by_factor(height * beta, factor))
        width_bar = max(factor, _ceil_by_factor(width * beta, factor))

    return height_bar, width_bar


def _smart_nframes(
    *,
    total_num_frames: int,
    source_fps: float,
    frame_factor: int,
    fps: float | None,
    num_frames: int | None,
    min_frames: int | None,
    max_frames: int | None,
) -> int:
    if total_num_frames <= 0:
        raise ValueError("Video contains no frames.")
    if source_fps <= 0:
        raise ValueError(f"Invalid source FPS: {source_fps}")

    if fps is not None and num_frames is not None:
        raise ValueError("Only one of `fps` or exact frame count may be specified.")

    if num_frames is not None:
        target_frames = int(num_frames)
    elif fps is not None:
        target_frames = int(total_num_frames / source_fps * fps)
    else:
        target_frames = total_num_frames

    if min_frames is None:
        min_frames = frame_factor
    if max_frames is None:
        max_frames = total_num_frames

    min_frames = max(int(min_frames), frame_factor)
    max_frames = min(int(max_frames), total_num_frames)
    target_frames = max(min(target_frames, max_frames), min_frames)

    if total_num_frames >= frame_factor:
        target_frames = max(frame_factor, _round_by_factor(target_frames, frame_factor))
        max_factor_aligned = _floor_by_factor(max_frames, frame_factor)
        if max_factor_aligned >= frame_factor:
            target_frames = min(target_frames, max_factor_aligned)

    return max(1, min(target_frames, total_num_frames))


def _resolve_qwen_frame_pixels(
    *,
    image_factor: int,
    frame_factor: int,
    sampled_num_frames: int,
    min_pixels: int | None,
    max_pixels: int | None,
    total_pixels: int | None,
) -> tuple[int, int]:
    if min_pixels is None:
        min_pixels = QWEN_VIDEO_MIN_TOKEN_NUM * image_factor * image_factor
    min_pixels = int(min_pixels)

    cap_pixels = QWEN_VIDEO_MAX_TOKEN_NUM * image_factor * image_factor
    resolved_max_pixels = cap_pixels

    if total_pixels is not None:
        total_pixels = int(total_pixels)
        resolved_max_pixels = int(
            max(
                min(cap_pixels, total_pixels / max(sampled_num_frames, 1) * frame_factor),
                int(min_pixels * 1.05),
            )
        )

    if max_pixels is not None:
        max_pixels = int(max_pixels)
        resolved_max_pixels = min(max_pixels, resolved_max_pixels)

    if resolved_max_pixels < min_pixels:
        resolved_max_pixels = int(min_pixels * 1.05)

    return min_pixels, resolved_max_pixels


def _resize_video_tensor(
    video_tensor: Any,
    *,
    height: int,
    width: int,
) -> Any:
    current_height = int(video_tensor.shape[-2])
    current_width = int(video_tensor.shape[-1])
    if current_height == height and current_width == width:
        return video_tensor

    return tvF.resize(
        video_tensor,
        [height, width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )


def _ensure_qwen_vl_utils_importable() -> None:
    try:
        import qwen_vl_utils  # noqa: F401
        return
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "qwen-vl-utils" / "src"
    if candidate.exists():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

    import qwen_vl_utils  # noqa: F401


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _to_python_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return [value]
    return value


def _normalize_video_metadata(metadata: dict[str, Any], *, sampled_num_frames: int) -> dict[str, Any]:
    normalized = {key: value for key, value in metadata.items()}
    if "frames_indices" in normalized:
        normalized["frames_indices"] = [
            int(_to_python_scalar(item)) for item in _to_python_list(normalized["frames_indices"])
        ]
    fps = normalized.get("fps")
    if fps is not None:
        normalized["fps"] = float(_to_python_scalar(fps))

    total_num_frames = normalized.get("total_num_frames")
    if total_num_frames is not None:
        total_num_frames = _to_python_scalar(total_num_frames)
        if isinstance(total_num_frames, float) and total_num_frames.is_integer():
            total_num_frames = int(total_num_frames)
        normalized["total_num_frames"] = total_num_frames

    if normalized.get("duration") is None:
        fps_value = normalized.get("fps")
        total_frames_value = normalized.get("total_num_frames")
        if fps_value not in (None, 0) and total_frames_value is not None:
            normalized["duration"] = float(total_frames_value) / float(fps_value)

    if "frames_indices" not in normalized:
        normalized["frames_indices"] = list(range(sampled_num_frames))

    return normalized


def _frame_timestamps_from_metadata(metadata: dict[str, Any], *, sampled_num_frames: int) -> list[float]:
    fps = metadata.get("fps")
    frame_indices = metadata.get("frames_indices")
    if fps in (None, 0) or frame_indices is None:
        return []
    return [float(frame_idx / fps) for frame_idx in frame_indices[:sampled_num_frames]]


def prepare_qwen_video_input(
    video_path: str,
    *,
    image_factor: int,
    frame_factor: int,
    video_kwargs: dict[str, Any],
) -> PreparedQwenVideoInput:
    """Prepare pre-sampled and pre-resized video inputs like qwen_vl_utils.

    The returned tensor is already sampled and resized so that Qwen-family
    processors can be called with `do_sample_frames=False` and `do_resize=False`,
    matching the official `process_vision_info` flow used by TimeLens.
    """

    _require_video_preprocess_dependencies()

    fps = video_kwargs.get("fps")
    exact_frames = video_kwargs.get("nframes", video_kwargs.get("num_frames"))
    min_frames = video_kwargs.get("min_frames")
    max_frames = video_kwargs.get("max_frames")
    min_pixels = video_kwargs.get("min_pixels")
    max_pixels = video_kwargs.get("max_pixels")
    total_pixels = video_kwargs.get("total_pixels")

    reader = VideoReader(str(Path(video_path).resolve()), ctx=cpu(0), num_threads=1)
    total_num_frames = len(reader)
    source_fps = float(reader.get_avg_fps())

    sampled_num_frames = _smart_nframes(
        total_num_frames=total_num_frames,
        source_fps=source_fps,
        frame_factor=frame_factor,
        fps=float(fps) if fps is not None else None,
        num_frames=int(exact_frames) if exact_frames is not None else None,
        min_frames=int(min_frames) if min_frames is not None else None,
        max_frames=int(max_frames) if max_frames is not None else None,
    )

    frame_indices = np.linspace(0, total_num_frames - 1, sampled_num_frames).round().astype(int)
    frames = reader.get_batch(frame_indices.tolist()).asnumpy()

    video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    original_height = int(video_tensor.shape[-2])
    original_width = int(video_tensor.shape[-1])

    resolved_min_pixels, resolved_max_pixels = _resolve_qwen_frame_pixels(
        image_factor=int(image_factor),
        frame_factor=frame_factor,
        sampled_num_frames=sampled_num_frames,
        min_pixels=int(min_pixels) if min_pixels is not None else None,
        max_pixels=int(max_pixels) if max_pixels is not None else None,
        total_pixels=int(total_pixels) if total_pixels is not None else None,
    )
    resized_height, resized_width = _smart_resize(
        original_height,
        original_width,
        factor=int(image_factor),
        min_pixels=resolved_min_pixels,
        max_pixels=resolved_max_pixels,
    )
    video_tensor = _resize_video_tensor(
        video_tensor,
        height=resized_height,
        width=resized_width,
    )

    duration = total_num_frames / source_fps if source_fps > 0 else 0.0
    frame_timestamps = [float(frame_index / source_fps) for frame_index in frame_indices]
    metadata = {
        "fps": source_fps,
        "duration": duration,
        "total_num_frames": total_num_frames,
        "frames_indices": frame_indices.tolist(),
        "video_backend": "decord",
        "width": original_width,
        "height": original_height,
    }

    return PreparedQwenVideoInput(
        video=video_tensor,
        metadata=metadata,
        frame_timestamps=frame_timestamps,
        sampled_num_frames=sampled_num_frames,
        resized_height=resized_height,
        resized_width=resized_width,
        processor_video_kwargs={
            "do_sample_frames": False,
            "do_resize": False,
            **({"fps": float(fps)} if fps is not None else {}),
        },
    )


def prepare_qwen_video_input_with_qwen_vl_utils(
    video_path: str,
    *,
    image_patch_size: int,
    video_kwargs: dict[str, Any],
    include_video_metadata: bool,
) -> PreparedQwenVideoInput:
    _require_video_preprocess_dependencies()
    _ensure_qwen_vl_utils_importable()

    from qwen_vl_utils.vision_process import fetch_video

    official_video_kwargs = dict(video_kwargs)
    if "num_frames" in official_video_kwargs and "nframes" not in official_video_kwargs:
        official_video_kwargs["nframes"] = official_video_kwargs["num_frames"]
    official_video_kwargs.pop("num_frames", None)
    official_video_kwargs["video"] = str(Path(video_path).resolve())

    video_input, sample_fps = fetch_video(
        official_video_kwargs,
        image_patch_size=int(image_patch_size),
        return_video_sample_fps=True,
        return_video_metadata=include_video_metadata,
    )

    if include_video_metadata:
        video_tensor, metadata = video_input
        metadata = _normalize_video_metadata(dict(metadata), sampled_num_frames=int(video_tensor.shape[0]))
        processor_video_kwargs = {
            "do_sample_frames": False,
            "do_resize": False,
        }
        if video_kwargs.get("fps") is not None:
            processor_video_kwargs["fps"] = float(video_kwargs["fps"])
    else:
        video_tensor = video_input
        metadata = {
            "fps": float(sample_fps),
            "duration": float(video_tensor.shape[0]) / float(sample_fps)
            if sample_fps not in (None, 0)
            else None,
            "total_num_frames": int(video_tensor.shape[0]),
            "frames_indices": list(range(int(video_tensor.shape[0]))),
            "video_backend": "qwen_vl_utils",
        }
        processor_video_kwargs = {
            "do_sample_frames": False,
            "fps": float(sample_fps),
        }

    sampled_num_frames = int(video_tensor.shape[0])
    resized_height = int(video_tensor.shape[-2])
    resized_width = int(video_tensor.shape[-1])
    frame_timestamps = _frame_timestamps_from_metadata(
        metadata,
        sampled_num_frames=sampled_num_frames,
    )

    return PreparedQwenVideoInput(
        video=video_tensor,
        metadata=metadata,
        frame_timestamps=frame_timestamps,
        sampled_num_frames=sampled_num_frames,
        resized_height=resized_height,
        resized_width=resized_width,
        processor_video_kwargs=processor_video_kwargs,
    )
