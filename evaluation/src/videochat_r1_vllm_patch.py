from __future__ import annotations

import logging
from collections.abc import Sequence
from numbers import Real
from typing import Any


LOGGER = logging.getLogger(__name__)

_RUNTIME_PATCHED = False


def _normalize_fps_values(fps: Any, *, expected_len: int) -> list[float]:
    if isinstance(fps, Real):
        value = float(fps)
        if value <= 0:
            raise ValueError(f"Invalid fps={value}.")
        return [value] * expected_len

    if isinstance(fps, Sequence) and not isinstance(fps, (str, bytes)):
        values = [float(item) for item in fps]
        if len(values) != expected_len:
            raise ValueError(
                "The length of fps "
                f"({len(values)}) must match the number of videos ({expected_len})."
            )
        if any(item <= 0 for item in values):
            raise ValueError(f"Invalid fps sequence: {values!r}")
        return values

    raise ValueError(f"Unsupported fps payload for VideoChat-R1 compatibility patch: {fps!r}")


def _overwrite_second_per_grid_ts(
    outputs: Any,
    *,
    fps_values: list[float],
    temporal_patch_size: int,
) -> None:
    patched_values = [float(temporal_patch_size) / fps for fps in fps_values]
    current = outputs.get("second_per_grid_ts")

    if hasattr(current, "new_tensor"):
        outputs["second_per_grid_ts"] = current.new_tensor(patched_values)
        return

    if isinstance(current, tuple):
        outputs["second_per_grid_ts"] = tuple(patched_values)
        return

    outputs["second_per_grid_ts"] = patched_values


def apply_videochat_r1_qwen25_processor_runtime_patch() -> None:
    global _RUNTIME_PATCHED
    if _RUNTIME_PATCHED:
        return

    from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor

    original_call = getattr(Qwen2_5_VLProcessor, "__call__", None)
    if original_call is None:
        LOGGER.warning("Qwen2_5_VLProcessor.__call__ is unavailable; skip VideoChat-R1 patch.")
        return

    if getattr(original_call, "_eval_suite_videochat_r1_patched", False):
        _RUNTIME_PATCHED = True
        return

    def patched_call(
        self,
        images=None,
        text=None,
        videos=None,
        **kwargs,
    ):
        outputs = original_call(self, images=images, text=text, videos=videos, **kwargs)

        if videos is None or "second_per_grid_ts" not in outputs:
            return outputs

        video_grid_thw = outputs.get("video_grid_thw")
        expected_len = len(video_grid_thw) if video_grid_thw is not None else 1
        fps = kwargs.get("fps", 2.0)
        fps_values = _normalize_fps_values(fps, expected_len=expected_len)

        _overwrite_second_per_grid_ts(
            outputs,
            fps_values=fps_values,
            temporal_patch_size=int(self.video_processor.temporal_patch_size),
        )
        return outputs

    patched_call._eval_suite_videochat_r1_patched = True  # type: ignore[attr-defined]
    Qwen2_5_VLProcessor.__call__ = patched_call
    _RUNTIME_PATCHED = True
    LOGGER.info(
        "Applied VideoChat-R1 vLLM compatibility patch: "
        "derive second_per_grid_ts from requested fps."
    )
