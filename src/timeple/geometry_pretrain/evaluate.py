"""Evaluation utilities for span-only CIS geometry pretraining."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from timeple.models.losses import interval_iou

DEFAULT_DURATION_BUCKET_COUNT = 8


@dataclass
class DurationBucketSummary:
    edges_sec: np.ndarray
    centers_sec: np.ndarray
    counts: np.ndarray
    mae_start: np.ndarray
    mae_end: np.ndarray
    mae_total: np.ndarray
    relative_mae_total: np.ndarray
    span_iou: np.ndarray


@dataclass
class SpanBucketSummary:
    edges: np.ndarray
    centers: np.ndarray
    counts: np.ndarray
    mae_start: np.ndarray
    mae_end: np.ndarray
    mae_total: np.ndarray
    relative_mae_over_span: np.ndarray
    span_iou: np.ndarray


@dataclass
class EvaluationArtifacts:
    span_mae_total: np.ndarray
    span_relative_mae_over_span: np.ndarray
    span_iou: np.ndarray
    span_counts: np.ndarray
    target_u: np.ndarray
    target_v: np.ndarray
    pred_u: np.ndarray
    pred_v: np.ndarray
    error_mae_total: np.ndarray
    duration_bucket_edges_sec: np.ndarray
    duration_bucket_centers_sec: np.ndarray
    duration_bucket_counts: np.ndarray
    duration_bucket_mae_start: np.ndarray
    duration_bucket_mae_end: np.ndarray
    duration_bucket_mae_total: np.ndarray
    duration_bucket_relative_mae_total: np.ndarray
    duration_bucket_span_iou: np.ndarray
    span_duration_bucket_edges_sec: np.ndarray
    span_duration_bucket_centers_sec: np.ndarray
    span_duration_bucket_counts: np.ndarray
    span_duration_bucket_mae_start: np.ndarray
    span_duration_bucket_mae_end: np.ndarray
    span_duration_bucket_mae_total: np.ndarray
    span_duration_bucket_relative_mae_over_span: np.ndarray
    span_duration_bucket_span_iou: np.ndarray
    relative_span_duration_bucket_edges: np.ndarray
    relative_span_duration_bucket_centers: np.ndarray
    relative_span_duration_bucket_counts: np.ndarray
    relative_span_duration_bucket_mae_start: np.ndarray
    relative_span_duration_bucket_mae_end: np.ndarray
    relative_span_duration_bucket_mae_total: np.ndarray
    relative_span_duration_bucket_relative_mae_over_span: np.ndarray
    relative_span_duration_bucket_span_iou: np.ndarray
    boundary_distance_bucket_edges: np.ndarray
    boundary_distance_bucket_centers: np.ndarray
    boundary_distance_bucket_counts: np.ndarray
    boundary_distance_bucket_mae_start: np.ndarray
    boundary_distance_bucket_mae_end: np.ndarray
    boundary_distance_bucket_mae_total: np.ndarray
    boundary_distance_bucket_relative_mae_over_span: np.ndarray
    boundary_distance_bucket_span_iou: np.ndarray


@dataclass
class EvaluationSummary:
    metrics: Dict[str, float]
    duration_buckets: DurationBucketSummary
    span_duration_buckets: SpanBucketSummary
    relative_span_duration_buckets: SpanBucketSummary
    boundary_distance_buckets: SpanBucketSummary
    artifacts: Optional[EvaluationArtifacts] = None


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full_like(numerator, np.nan, dtype=np.float64)
    mask = denominator > 0
    output[mask] = numerator[mask] / denominator[mask]
    return output


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def build_duration_bucket_edges(
    duration_min_sec: float,
    duration_max_sec: float,
    *,
    distribution: str,
    num_buckets: int = DEFAULT_DURATION_BUCKET_COUNT,
) -> np.ndarray:
    num_buckets = max(int(num_buckets), 1)
    lo = max(float(duration_min_sec), 1e-6)
    hi = max(float(duration_max_sec), lo * (1.0 + 1e-6))
    if distribution == "log_uniform":
        return np.geomspace(lo, hi, num_buckets + 1, dtype=np.float64)
    return np.linspace(lo, hi, num_buckets + 1, dtype=np.float64)


def _build_duration_bucket_centers(edges_sec: np.ndarray, *, distribution: str) -> np.ndarray:
    if distribution == "log_uniform":
        return np.sqrt(edges_sec[:-1] * edges_sec[1:])
    return 0.5 * (edges_sec[:-1] + edges_sec[1:])


def _build_log_bucket_edges_from_values(
    values: np.ndarray,
    *,
    num_buckets: int = DEFAULT_DURATION_BUCKET_COUNT,
) -> np.ndarray:
    num_buckets = max(int(num_buckets), 1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.geomspace(1e-6, 1.0, num_buckets + 1, dtype=np.float64)
    positive = finite[finite > 0.0]
    lo = float(positive.min()) if positive.size > 0 else 1e-6
    hi = float(finite.max())
    lo = max(lo, 1e-6)
    hi = max(hi, lo * (1.0 + 1e-6))
    return np.geomspace(lo, hi, num_buckets + 1, dtype=np.float64)


def _build_warp_aligned_relative_duration_edges(
    codec,
    *,
    num_buckets: int = DEFAULT_DURATION_BUCKET_COUNT,
) -> np.ndarray:
    num_buckets = max(int(num_buckets), 1)
    v_edges = torch.linspace(0.0, 1.0, num_buckets + 1, dtype=torch.float32)
    relative_edges = codec.transform.inverse_warp_duration(v_edges).detach().cpu().numpy().astype(np.float64)
    relative_edges[0] = 0.0
    relative_edges[-1] = 1.0
    return relative_edges


def _build_linear_bucket_edges(
    min_value: float,
    max_value: float,
    *,
    num_buckets: int = DEFAULT_DURATION_BUCKET_COUNT,
) -> np.ndarray:
    num_buckets = max(int(num_buckets), 1)
    return np.linspace(float(min_value), float(max_value), num_buckets + 1, dtype=np.float64)


def _build_bucket_centers(edges: np.ndarray, *, use_geometric_mean: bool) -> np.ndarray:
    if use_geometric_mean and np.all(edges[:-1] > 0.0) and np.all(edges[1:] > 0.0):
        return np.sqrt(edges[:-1] * edges[1:])
    return 0.5 * (edges[:-1] + edges[1:])


def _summarize_span_buckets(
    bucket_values: np.ndarray,
    bucket_edges: np.ndarray,
    *,
    mae_start: np.ndarray,
    mae_end: np.ndarray,
    mae_total: np.ndarray,
    relative_mae_over_span: np.ndarray,
    span_iou: np.ndarray,
    use_geometric_mean: bool,
) -> SpanBucketSummary:
    num_buckets = int(bucket_edges.shape[0] - 1)
    bucket_idx = np.searchsorted(bucket_edges, bucket_values, side="right") - 1
    bucket_idx = np.clip(bucket_idx, 0, num_buckets - 1)

    counts = np.bincount(bucket_idx, minlength=num_buckets).astype(np.float64)
    mae_start_sum = np.bincount(bucket_idx, weights=mae_start, minlength=num_buckets).astype(np.float64)
    mae_end_sum = np.bincount(bucket_idx, weights=mae_end, minlength=num_buckets).astype(np.float64)
    mae_total_sum = np.bincount(bucket_idx, weights=mae_total, minlength=num_buckets).astype(np.float64)
    relative_mae_over_span_sum = np.bincount(
        bucket_idx,
        weights=relative_mae_over_span,
        minlength=num_buckets,
    ).astype(np.float64)
    span_iou_sum = np.bincount(bucket_idx, weights=span_iou, minlength=num_buckets).astype(np.float64)

    return SpanBucketSummary(
        edges=bucket_edges,
        centers=_build_bucket_centers(bucket_edges, use_geometric_mean=use_geometric_mean),
        counts=counts,
        mae_start=_safe_divide(mae_start_sum, counts),
        mae_end=_safe_divide(mae_end_sum, counts),
        mae_total=_safe_divide(mae_total_sum, counts),
        relative_mae_over_span=_safe_divide(relative_mae_over_span_sum, counts),
        span_iou=_safe_divide(span_iou_sum, counts),
    )


@torch.inference_mode()
def evaluate_codec(
    codec,
    dataloader,
    device: torch.device,
    *,
    noise_std: float = 0.0,
    collect_artifacts: bool = False,
) -> EvaluationSummary:
    codec.eval()

    dataset = dataloader.dataset
    span_shape = (dataset.config.span_v_bins, dataset.config.span_u_bins)
    duration_bucket_edges_sec = build_duration_bucket_edges(
        dataset.config.duration_min_sec,
        dataset.config.duration_max_sec,
        distribution=dataset.config.duration_distribution,
    )
    duration_bucket_centers_sec = _build_duration_bucket_centers(
        duration_bucket_edges_sec,
        distribution=dataset.config.duration_distribution,
    )
    num_duration_buckets = duration_bucket_edges_sec.shape[0] - 1

    loss_sums = defaultdict(float)
    metric_sums = defaultdict(float)
    sample_count = 0

    span_mae_sum = np.zeros(span_shape, dtype=np.float64)
    span_relative_mae_over_span_sum = np.zeros(span_shape, dtype=np.float64)
    span_iou_sum = np.zeros(span_shape, dtype=np.float64)
    span_counts = np.zeros(span_shape, dtype=np.float64)
    duration_counts = np.zeros(num_duration_buckets, dtype=np.float64)
    duration_mae_start_sum = np.zeros(num_duration_buckets, dtype=np.float64)
    duration_mae_end_sum = np.zeros(num_duration_buckets, dtype=np.float64)
    duration_mae_total_sum = np.zeros(num_duration_buckets, dtype=np.float64)
    duration_relative_mae_total_sum = np.zeros(num_duration_buckets, dtype=np.float64)
    duration_span_iou_sum = np.zeros(num_duration_buckets, dtype=np.float64)

    target_u_all = []
    target_v_all = []
    pred_u_all = []
    pred_v_all = []
    mae_total_all = []
    span_duration_sec_all = []
    span_duration_rel_all = []
    span_boundary_distance_all = []
    span_mae_start_all = []
    span_mae_end_all = []
    span_mae_total_all = []
    span_relative_mae_over_span_all = []
    span_iou_all = []

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        start_sec = batch["start_sec"]
        end_sec = batch["end_sec"]
        duration_sec = batch["duration_sec"]
        square_u = batch["square_u"]
        square_v = batch["square_v"]
        u_cell_idx = batch["u_cell_idx"]
        v_cell_idx = batch["v_cell_idx"]

        token = codec.encode(start_sec, end_sec, duration_sec)
        if noise_std > 0:
            token = token + noise_std * torch.randn_like(token)

        loss_dict = codec.compute_loss(token, start_sec, end_sec, duration_sec)
        batch_size = int(start_sec.shape[0])
        sample_count += batch_size

        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor) and value.ndim == 0:
                loss_sums[key] += float(value.item()) * batch_size

        pred_start_sec, pred_end_sec, _ = codec.decode(
            token,
            video_duration_sec=duration_sec,
            hard=True,
            return_details=True,
        )
        target_start_rel, target_end_rel = codec.transform.normalize_seconds(start_sec, end_sec, duration_sec)
        pred_start_rel, pred_end_rel = codec.transform.normalize_seconds(pred_start_sec, pred_end_sec, duration_sec)
        pred_u, pred_v, _ = codec.transform.interval_to_square(pred_start_rel, pred_end_rel)

        mae_start = (pred_start_sec - start_sec).abs()
        mae_end = (pred_end_sec - end_sec).abs()
        mae_total = 0.5 * (mae_start + mae_end)
        square_l1 = 0.5 * ((pred_u - square_u).abs() + (pred_v - square_v).abs())
        iou = interval_iou(pred_start_rel, pred_end_rel, target_start_rel, target_end_rel)
        span_duration_sec = (end_sec - start_sec).clamp_min(1e-8)
        span_duration_rel = (target_end_rel - target_start_rel).clamp_min(1e-8)
        span_boundary_distance = torch.minimum(target_start_rel, 1.0 - target_end_rel).clamp_min(0.0)
        relative_mae_over_span = mae_total / span_duration_sec

        metric_sums["mae_start"] += float(mae_start.sum().item())
        metric_sums["mae_end"] += float(mae_end.sum().item())
        metric_sums["mae_total"] += float(mae_total.sum().item())
        metric_sums["square_l1"] += float(square_l1.sum().item())
        metric_sums["span_iou"] += float(iou.sum().item())

        u_cell_np = u_cell_idx.detach().cpu().numpy()
        v_cell_np = v_cell_idx.detach().cpu().numpy()
        duration_sec_np = duration_sec.detach().cpu().numpy()
        mae_start_np = mae_start.detach().cpu().numpy()
        mae_end_np = mae_end.detach().cpu().numpy()
        mae_total_np = mae_total.detach().cpu().numpy()
        relative_mae_np = relative_mae_over_span.detach().cpu().numpy()
        iou_np = iou.detach().cpu().numpy()
        span_duration_sec_np = span_duration_sec.detach().cpu().numpy()

        duration_bucket_idx_np = np.searchsorted(
            duration_bucket_edges_sec,
            duration_sec_np,
            side="right",
        ) - 1
        duration_bucket_idx_np = np.clip(duration_bucket_idx_np, 0, num_duration_buckets - 1)

        for idx in range(batch_size):
            u_idx = int(u_cell_np[idx])
            v_idx = int(v_cell_np[idx])
            span_mae_sum[v_idx, u_idx] += float(mae_total_np[idx])
            span_relative_mae_over_span_sum[v_idx, u_idx] += float(relative_mae_np[idx])
            span_iou_sum[v_idx, u_idx] += float(iou_np[idx])
            span_counts[v_idx, u_idx] += 1.0

            bucket_idx = int(duration_bucket_idx_np[idx])
            duration_value = max(float(duration_sec_np[idx]), 1e-8)
            duration_counts[bucket_idx] += 1.0
            duration_mae_start_sum[bucket_idx] += float(mae_start_np[idx])
            duration_mae_end_sum[bucket_idx] += float(mae_end_np[idx])
            duration_mae_total_sum[bucket_idx] += float(mae_total_np[idx])
            duration_relative_mae_total_sum[bucket_idx] += float(mae_total_np[idx]) / duration_value
            duration_span_iou_sum[bucket_idx] += float(iou_np[idx])

        span_mae_start_all.append(mae_start_np)
        span_mae_end_all.append(mae_end_np)
        span_mae_total_all.append(mae_total_np)
        span_relative_mae_over_span_all.append(relative_mae_np)
        span_duration_sec_all.append(span_duration_sec_np)
        span_duration_rel_all.append(span_duration_rel.detach().cpu().numpy())
        span_boundary_distance_all.append(span_boundary_distance.detach().cpu().numpy())
        span_iou_all.append(iou_np)

        if collect_artifacts:
            target_u_all.append(square_u.detach().cpu().numpy())
            target_v_all.append(square_v.detach().cpu().numpy())
            pred_u_all.append(pred_u.detach().cpu().numpy())
            pred_v_all.append(pred_v.detach().cpu().numpy())
            mae_total_all.append(mae_total_np)

    metrics = {
        key: value / max(sample_count, 1)
        for key, value in loss_sums.items()
    }
    metrics.update({
        "mae_start": metric_sums["mae_start"] / max(sample_count, 1),
        "mae_end": metric_sums["mae_end"] / max(sample_count, 1),
        "mae_total": metric_sums["mae_total"] / max(sample_count, 1),
        "square_l1": metric_sums["square_l1"] / max(sample_count, 1),
        "span_iou": metric_sums["span_iou"] / max(sample_count, 1),
        "span_fraction": 1.0 if sample_count > 0 else 0.0,
        "sample_count": float(sample_count),
    })

    duration_summary = DurationBucketSummary(
        edges_sec=duration_bucket_edges_sec,
        centers_sec=duration_bucket_centers_sec,
        counts=duration_counts,
        mae_start=_safe_divide(duration_mae_start_sum, duration_counts),
        mae_end=_safe_divide(duration_mae_end_sum, duration_counts),
        mae_total=_safe_divide(duration_mae_total_sum, duration_counts),
        relative_mae_total=_safe_divide(duration_relative_mae_total_sum, duration_counts),
        span_iou=_safe_divide(duration_span_iou_sum, duration_counts),
    )

    span_duration_sec = (
        np.concatenate(span_duration_sec_all, axis=0)
        if span_duration_sec_all else np.array([], dtype=np.float32)
    )
    span_duration_rel = (
        np.concatenate(span_duration_rel_all, axis=0)
        if span_duration_rel_all else np.array([], dtype=np.float32)
    )
    span_boundary_distance = (
        np.concatenate(span_boundary_distance_all, axis=0)
        if span_boundary_distance_all else np.array([], dtype=np.float32)
    )
    span_mae_start = (
        np.concatenate(span_mae_start_all, axis=0)
        if span_mae_start_all else np.array([], dtype=np.float32)
    )
    span_mae_end = (
        np.concatenate(span_mae_end_all, axis=0)
        if span_mae_end_all else np.array([], dtype=np.float32)
    )
    span_mae_total = (
        np.concatenate(span_mae_total_all, axis=0)
        if span_mae_total_all else np.array([], dtype=np.float32)
    )
    span_relative_mae_over_span = (
        np.concatenate(span_relative_mae_over_span_all, axis=0)
        if span_relative_mae_over_span_all else np.array([], dtype=np.float32)
    )
    span_iou_values = (
        np.concatenate(span_iou_all, axis=0)
        if span_iou_all else np.array([], dtype=np.float32)
    )

    span_duration_bucket_edges_sec = _build_log_bucket_edges_from_values(span_duration_sec)
    relative_span_duration_bucket_edges = _build_warp_aligned_relative_duration_edges(codec)
    boundary_distance_bucket_edges = _build_linear_bucket_edges(0.0, 0.5)

    span_duration_summary = _summarize_span_buckets(
        span_duration_sec.astype(np.float64, copy=False),
        span_duration_bucket_edges_sec,
        mae_start=span_mae_start.astype(np.float64, copy=False),
        mae_end=span_mae_end.astype(np.float64, copy=False),
        mae_total=span_mae_total.astype(np.float64, copy=False),
        relative_mae_over_span=span_relative_mae_over_span.astype(np.float64, copy=False),
        span_iou=span_iou_values.astype(np.float64, copy=False),
        use_geometric_mean=True,
    )
    relative_span_duration_summary = _summarize_span_buckets(
        span_duration_rel.astype(np.float64, copy=False),
        relative_span_duration_bucket_edges,
        mae_start=span_mae_start.astype(np.float64, copy=False),
        mae_end=span_mae_end.astype(np.float64, copy=False),
        mae_total=span_mae_total.astype(np.float64, copy=False),
        relative_mae_over_span=span_relative_mae_over_span.astype(np.float64, copy=False),
        span_iou=span_iou_values.astype(np.float64, copy=False),
        use_geometric_mean=False,
    )
    boundary_distance_summary = _summarize_span_buckets(
        span_boundary_distance.astype(np.float64, copy=False),
        boundary_distance_bucket_edges,
        mae_start=span_mae_start.astype(np.float64, copy=False),
        mae_end=span_mae_end.astype(np.float64, copy=False),
        mae_total=span_mae_total.astype(np.float64, copy=False),
        relative_mae_over_span=span_relative_mae_over_span.astype(np.float64, copy=False),
        span_iou=span_iou_values.astype(np.float64, copy=False),
        use_geometric_mean=False,
    )

    artifacts = None
    if collect_artifacts:
        target_u = np.concatenate(target_u_all, axis=0) if target_u_all else np.array([], dtype=np.float32)
        target_v = np.concatenate(target_v_all, axis=0) if target_v_all else np.array([], dtype=np.float32)
        pred_u = np.concatenate(pred_u_all, axis=0) if pred_u_all else np.array([], dtype=np.float32)
        pred_v = np.concatenate(pred_v_all, axis=0) if pred_v_all else np.array([], dtype=np.float32)
        error_mae_total = np.concatenate(mae_total_all, axis=0) if mae_total_all else np.array([], dtype=np.float32)
        artifacts = EvaluationArtifacts(
            span_mae_total=_safe_divide(span_mae_sum, span_counts),
            span_relative_mae_over_span=_safe_divide(span_relative_mae_over_span_sum, span_counts),
            span_iou=_safe_divide(span_iou_sum, span_counts),
            span_counts=span_counts,
            target_u=target_u,
            target_v=target_v,
            pred_u=pred_u,
            pred_v=pred_v,
            error_mae_total=error_mae_total,
            duration_bucket_edges_sec=duration_summary.edges_sec,
            duration_bucket_centers_sec=duration_summary.centers_sec,
            duration_bucket_counts=duration_summary.counts,
            duration_bucket_mae_start=duration_summary.mae_start,
            duration_bucket_mae_end=duration_summary.mae_end,
            duration_bucket_mae_total=duration_summary.mae_total,
            duration_bucket_relative_mae_total=duration_summary.relative_mae_total,
            duration_bucket_span_iou=duration_summary.span_iou,
            span_duration_bucket_edges_sec=span_duration_summary.edges,
            span_duration_bucket_centers_sec=span_duration_summary.centers,
            span_duration_bucket_counts=span_duration_summary.counts,
            span_duration_bucket_mae_start=span_duration_summary.mae_start,
            span_duration_bucket_mae_end=span_duration_summary.mae_end,
            span_duration_bucket_mae_total=span_duration_summary.mae_total,
            span_duration_bucket_relative_mae_over_span=span_duration_summary.relative_mae_over_span,
            span_duration_bucket_span_iou=span_duration_summary.span_iou,
            relative_span_duration_bucket_edges=relative_span_duration_summary.edges,
            relative_span_duration_bucket_centers=relative_span_duration_summary.centers,
            relative_span_duration_bucket_counts=relative_span_duration_summary.counts,
            relative_span_duration_bucket_mae_start=relative_span_duration_summary.mae_start,
            relative_span_duration_bucket_mae_end=relative_span_duration_summary.mae_end,
            relative_span_duration_bucket_mae_total=relative_span_duration_summary.mae_total,
            relative_span_duration_bucket_relative_mae_over_span=relative_span_duration_summary.relative_mae_over_span,
            relative_span_duration_bucket_span_iou=relative_span_duration_summary.span_iou,
            boundary_distance_bucket_edges=boundary_distance_summary.edges,
            boundary_distance_bucket_centers=boundary_distance_summary.centers,
            boundary_distance_bucket_counts=boundary_distance_summary.counts,
            boundary_distance_bucket_mae_start=boundary_distance_summary.mae_start,
            boundary_distance_bucket_mae_end=boundary_distance_summary.mae_end,
            boundary_distance_bucket_mae_total=boundary_distance_summary.mae_total,
            boundary_distance_bucket_relative_mae_over_span=boundary_distance_summary.relative_mae_over_span,
            boundary_distance_bucket_span_iou=boundary_distance_summary.span_iou,
        )

    return EvaluationSummary(
        metrics=metrics,
        duration_buckets=duration_summary,
        span_duration_buckets=span_duration_summary,
        relative_span_duration_buckets=relative_span_duration_summary,
        boundary_distance_buckets=boundary_distance_summary,
        artifacts=artifacts,
    )
