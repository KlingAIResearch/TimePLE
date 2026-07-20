"""
Canonical span interval transforms for TimePLE-Codec.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


class CanonicalSpanIntervalTransform:
    """
    Maps span intervals on ``[0, 1]`` to the canonical interval square and back.

    The square coordinates follow the CIS design, but this span-only package
    exposes the transform through a span-specific name:

    - ``d = e - s``
    - ``u = s / (1 - d + eps)``
    - ``v = log(1 + d / tau) / log(1 + 1 / tau)`` when ``tau > 0``
    - ``v = d`` when ``tau == 0`` (no duration warp)
    """

    def __init__(
        self,
        tau: float = 0.03,
        eps: float = 1e-6,
        point_threshold: float = 1e-4,
    ) -> None:
        if tau < 0.0:
            raise ValueError(f"`tau` must be non-negative, got {tau}.")
        if eps <= 0.0:
            raise ValueError(f"`eps` must be positive, got {eps}.")
        if point_threshold < 0.0:
            raise ValueError(f"`point_threshold` must be non-negative, got {point_threshold}.")

        self.tau = float(tau)
        self.eps = float(eps)
        self.point_threshold = float(point_threshold)
        self._warp_denominator = math.log1p(1.0 / self.tau) if self.tau > 0.0 else None

    def _broadcast_duration(
        self,
        video_duration_sec: torch.Tensor | float,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        duration = torch.as_tensor(
            video_duration_sec,
            device=reference.device,
            dtype=reference.dtype,
        )
        if duration.ndim == 0:
            duration = duration.expand_as(reference)
        elif duration.shape != reference.shape:
            duration = duration.reshape(-1)
            if duration.numel() == 1:
                duration = duration.expand_as(reference)
            elif duration.numel() != reference.numel():
                raise ValueError(
                    "Video duration shape does not match the number of intervals: "
                    f"{tuple(duration.shape)} vs {tuple(reference.shape)}."
                )
            else:
                duration = duration.reshape_as(reference)
        return duration.clamp_min(self.eps)

    def normalize_seconds(
        self,
        start_sec: torch.Tensor,
        end_sec: torch.Tensor,
        video_duration_sec: torch.Tensor | float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        duration = self._broadcast_duration(video_duration_sec, start_sec)
        start_rel = (start_sec / duration).clamp(0.0, 1.0)
        end_rel = (end_sec / duration).clamp(0.0, 1.0)
        start_rel = torch.minimum(start_rel, end_rel)
        end_rel = torch.maximum(start_rel, end_rel)
        return start_rel, end_rel

    def denormalize_seconds(
        self,
        start_rel: torch.Tensor,
        end_rel: torch.Tensor,
        video_duration_sec: torch.Tensor | float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        duration = self._broadcast_duration(video_duration_sec, start_rel)
        start_sec = start_rel.clamp(0.0, 1.0) * duration
        end_sec = end_rel.clamp(0.0, 1.0) * duration
        start_sec = torch.minimum(start_sec, end_sec)
        end_sec = torch.maximum(start_sec, end_sec)
        return start_sec, end_sec

    def warp_duration(self, duration_rel: torch.Tensor) -> torch.Tensor:
        duration_rel = duration_rel.clamp(0.0, 1.0)
        if self.tau == 0.0:
            return duration_rel
        return torch.log1p(duration_rel / self.tau) / self._warp_denominator

    def inverse_warp_duration(self, warped_duration: torch.Tensor) -> torch.Tensor:
        warped_duration = warped_duration.clamp(0.0, 1.0)
        if self.tau == 0.0:
            return warped_duration
        return (self.tau * torch.expm1(warped_duration * self._warp_denominator)).clamp(0.0, 1.0)

    def interval_to_square(
        self,
        start_rel: torch.Tensor,
        end_rel: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start_rel = start_rel.clamp(0.0, 1.0)
        end_rel = end_rel.clamp(0.0, 1.0)
        start_rel = torch.minimum(start_rel, end_rel)
        end_rel = torch.maximum(start_rel, end_rel)

        duration_rel = (end_rel - start_rel).clamp(0.0, 1.0)
        denom = (1.0 - duration_rel).clamp_min(self.eps)
        u = (start_rel / denom).clamp(0.0, 1.0)
        u = torch.where(duration_rel >= (1.0 - self.eps), torch.zeros_like(u), u)
        v = self.warp_duration(duration_rel)
        return u, v, duration_rel

    def square_to_interval(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = u.clamp(0.0, 1.0)
        v = v.clamp(0.0, 1.0)
        duration_rel = self.inverse_warp_duration(v)
        start_rel = (u * (1.0 - duration_rel)).clamp(0.0, 1.0)
        end_rel = (start_rel + duration_rel).clamp(0.0, 1.0)
        start_rel = torch.minimum(start_rel, end_rel)
        end_rel = torch.maximum(start_rel, end_rel)
        return start_rel, end_rel, duration_rel

    def is_point(self, start_rel: torch.Tensor, end_rel: torch.Tensor) -> torch.Tensor:
        return (end_rel - start_rel).abs() <= self.point_threshold

__all__ = ["CanonicalSpanIntervalTransform"]
