"""Visualization helpers for span-only CIS geometry pretraining."""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List

import numpy as np

from timeple.geometry_pretrain.evaluate import EvaluationArtifacts


def _get_plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(
            "Visualization requires `matplotlib`. Install it with `pip install matplotlib`."
        ) from e
    return plt


def save_artifacts_npz(artifacts: EvaluationArtifacts, path: str) -> None:
    np.savez_compressed(
        path,
        span_mae_total=artifacts.span_mae_total,
        span_relative_mae_over_span=artifacts.span_relative_mae_over_span,
        span_iou=artifacts.span_iou,
        span_counts=artifacts.span_counts,
        target_u=artifacts.target_u,
        target_v=artifacts.target_v,
        pred_u=artifacts.pred_u,
        pred_v=artifacts.pred_v,
        error_mae_total=artifacts.error_mae_total,
        duration_bucket_edges_sec=artifacts.duration_bucket_edges_sec,
        duration_bucket_centers_sec=artifacts.duration_bucket_centers_sec,
        duration_bucket_counts=artifacts.duration_bucket_counts,
        duration_bucket_mae_start=artifacts.duration_bucket_mae_start,
        duration_bucket_mae_end=artifacts.duration_bucket_mae_end,
        duration_bucket_mae_total=artifacts.duration_bucket_mae_total,
        duration_bucket_relative_mae_total=artifacts.duration_bucket_relative_mae_total,
        duration_bucket_span_iou=artifacts.duration_bucket_span_iou,
        span_duration_bucket_edges_sec=artifacts.span_duration_bucket_edges_sec,
        span_duration_bucket_centers_sec=artifacts.span_duration_bucket_centers_sec,
        span_duration_bucket_counts=artifacts.span_duration_bucket_counts,
        span_duration_bucket_mae_start=artifacts.span_duration_bucket_mae_start,
        span_duration_bucket_mae_end=artifacts.span_duration_bucket_mae_end,
        span_duration_bucket_mae_total=artifacts.span_duration_bucket_mae_total,
        span_duration_bucket_relative_mae_over_span=artifacts.span_duration_bucket_relative_mae_over_span,
        span_duration_bucket_span_iou=artifacts.span_duration_bucket_span_iou,
        relative_span_duration_bucket_edges=artifacts.relative_span_duration_bucket_edges,
        relative_span_duration_bucket_centers=artifacts.relative_span_duration_bucket_centers,
        relative_span_duration_bucket_counts=artifacts.relative_span_duration_bucket_counts,
        relative_span_duration_bucket_mae_start=artifacts.relative_span_duration_bucket_mae_start,
        relative_span_duration_bucket_mae_end=artifacts.relative_span_duration_bucket_mae_end,
        relative_span_duration_bucket_mae_total=artifacts.relative_span_duration_bucket_mae_total,
        relative_span_duration_bucket_relative_mae_over_span=artifacts.relative_span_duration_bucket_relative_mae_over_span,
        relative_span_duration_bucket_span_iou=artifacts.relative_span_duration_bucket_span_iou,
        boundary_distance_bucket_edges=artifacts.boundary_distance_bucket_edges,
        boundary_distance_bucket_centers=artifacts.boundary_distance_bucket_centers,
        boundary_distance_bucket_counts=artifacts.boundary_distance_bucket_counts,
        boundary_distance_bucket_mae_start=artifacts.boundary_distance_bucket_mae_start,
        boundary_distance_bucket_mae_end=artifacts.boundary_distance_bucket_mae_end,
        boundary_distance_bucket_mae_total=artifacts.boundary_distance_bucket_mae_total,
        boundary_distance_bucket_relative_mae_over_span=artifacts.boundary_distance_bucket_relative_mae_over_span,
        boundary_distance_bucket_span_iou=artifacts.boundary_distance_bucket_span_iou,
    )


def save_history_json(history: List[Dict[str, float]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def _plot_heatmap(matrix: np.ndarray, title: str, save_path: str, *, cmap: str) -> None:
    plt = _get_plt()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#d9d9d9")
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap_obj)
    ax.set_title(title)
    ax.set_xlabel("u cell")
    ax.set_ylabel("v cell")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _plot_curve(values: np.ndarray, title: str, ylabel: str, save_path: str) -> None:
    plt = _get_plt()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    x = np.arange(values.shape[0], dtype=np.float64) + 0.5
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(x, values, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("u cell")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _format_bucket_value(value: float) -> str:
    if value >= 100.0:
        return f"{value:.0f}"
    if value >= 10.0:
        return f"{value:.1f}"
    if value >= 1.0:
        return f"{value:.2f}"
    if value >= 0.1:
        return f"{value:.3f}"
    if value >= 0.01:
        return f"{value:.4f}"
    if value >= 0.001:
        return f"{value:.5f}"
    if value > 0.0:
        return f"{value:.2e}"
    return f"{value:.2f}"


def _format_duration_bucket_labels(edges_sec: np.ndarray) -> List[str]:
    return [
        f"{_format_bucket_value(float(lo))}-{_format_bucket_value(float(hi))}s"
        for lo, hi in zip(edges_sec[:-1], edges_sec[1:])
    ]


def _plot_duration_bucket_metrics(
    edges_sec: np.ndarray,
    series: Dict[str, np.ndarray],
    save_path: str,
    *,
    title: str,
    ylabel: str,
) -> None:
    plt = _get_plt()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    labels = _format_duration_bucket_labels(edges_sec)
    x = np.arange(len(labels), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    for label, values in series.items():
        ax.plot(x, values, marker="o", linewidth=1.8, label=label)
    ax.set_title(title)
    ax.set_xlabel("duration bucket")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(alpha=0.25)
    if len(series) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _plot_scalar_bucket_counts(
    edges: np.ndarray,
    counts: np.ndarray,
    save_path: str,
    *,
    title: str,
    xlabel: str,
) -> None:
    plt = _get_plt()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    labels = _format_duration_bucket_labels(edges)
    x = np.arange(len(labels), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    ax.bar(x, counts, width=0.72, color="#20c997")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("span samples")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _plot_scalar_bucket_suite(
    edges: np.ndarray,
    counts: np.ndarray,
    mae_start: np.ndarray,
    mae_end: np.ndarray,
    mae_total: np.ndarray,
    relative_mae_over_span: np.ndarray,
    span_iou: np.ndarray,
    output_dir: str,
    prefix: str,
    *,
    stem: str,
    title_prefix: str,
    xlabel: str,
) -> None:
    _plot_scalar_bucket_counts(
        edges,
        counts,
        save_path=os.path.join(output_dir, f"{prefix}_{stem}_counts.png"),
        title=f"{title_prefix} Counts",
        xlabel=xlabel,
    )
    _plot_duration_bucket_metrics(
        edges,
        {
            "mae_start": mae_start,
            "mae_end": mae_end,
            "mae_total": mae_total,
        },
        save_path=os.path.join(output_dir, f"{prefix}_{stem}_mae.png"),
        title=f"{title_prefix} MAE",
        ylabel="error (sec)",
    )
    _plot_duration_bucket_metrics(
        edges,
        {
            "mae_total / span_duration": relative_mae_over_span,
        },
        save_path=os.path.join(output_dir, f"{prefix}_{stem}_relative_mae_over_span.png"),
        title=f"{title_prefix} Relative MAE over Span",
        ylabel="mae_total / span_duration",
    )
    _plot_duration_bucket_metrics(
        edges,
        {
            "span_iou": span_iou,
        },
        save_path=os.path.join(output_dir, f"{prefix}_{stem}_span_iou.png"),
        title=f"{title_prefix} Span IoU",
        ylabel="IoU",
    )


def _plot_heatmap_on_axes(ax, fig, matrix: np.ndarray, title: str, *, cmap: str, colorbar_label: str) -> None:
    plt = _get_plt()
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#d9d9d9")
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap_obj)
    ax.set_title(title)
    ax.set_xlabel("u cell")
    ax.set_ylabel("v cell")
    fig.colorbar(image, ax=ax, label=colorbar_label, fraction=0.046, pad=0.04)


def _plot_bucket_curve_on_axes(
    ax,
    edges: np.ndarray,
    values: np.ndarray,
    title: str,
    *,
    ylabel: str,
    color: str,
) -> None:
    labels = _format_duration_bucket_labels(edges)
    x = np.arange(len(labels), dtype=np.float64)
    ax.plot(x, values, marker="o", linewidth=1.8, color=color)
    ax.set_title(title)
    ax.set_xlabel("bucket")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(alpha=0.25)


def _plot_span_focus_overview(artifacts: EvaluationArtifacts, output_dir: str, prefix: str) -> None:
    plt = _get_plt()
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
    fig.suptitle(f"{prefix} Span Focus Overview", fontsize=16)

    _plot_heatmap_on_axes(
        axes[0, 0],
        fig,
        artifacts.span_iou,
        title="Span IoU over Canonical Square",
        cmap="plasma",
        colorbar_label="IoU",
    )
    _plot_heatmap_on_axes(
        axes[0, 1],
        fig,
        artifacts.span_relative_mae_over_span,
        title="Relative MAE over Span over Canonical Square",
        cmap="viridis_r",
        colorbar_label="mae_total / span_duration",
    )
    _plot_bucket_curve_on_axes(
        axes[1, 0],
        artifacts.relative_span_duration_bucket_edges,
        artifacts.relative_span_duration_bucket_span_iou,
        title="Span IoU by Relative Span Duration",
        ylabel="IoU",
        color="#d9480f",
    )
    _plot_bucket_curve_on_axes(
        axes[1, 1],
        artifacts.boundary_distance_bucket_edges,
        artifacts.boundary_distance_bucket_span_iou,
        title="Span IoU by Boundary Distance",
        ylabel="IoU",
        color="#1c7ed6",
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(os.path.join(output_dir, f"{prefix}_span_focus_overview.png"), dpi=180)
    plt.close(fig)


def _plot_square_scatter(
    target_u: np.ndarray,
    target_v: np.ndarray,
    error_mae_total: np.ndarray,
    save_path: str,
    *,
    max_points: int = 8192,
) -> None:
    plt = _get_plt()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if target_u.shape[0] > max_points:
        stride = max(target_u.shape[0] // max_points, 1)
        target_u = target_u[::stride]
        target_v = target_v[::stride]
        error_mae_total = error_mae_total[::stride]

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    scatter = ax.scatter(
        target_u,
        target_v,
        c=error_mae_total,
        cmap="viridis",
        s=10,
        alpha=0.75,
        linewidths=0.0,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("target u")
    ax.set_ylabel("target v")
    ax.set_title("Per-sample Error over Canonical Square")
    fig.colorbar(scatter, ax=ax, label="mae_total (sec)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_dataset_coverage(coverage: Dict[str, np.ndarray], output_dir: str, prefix: str) -> None:
    span_counts = coverage["span_counts"].astype(np.float64)
    if span_counts.ndim == 3:
        span_counts = span_counts.sum(axis=0)
    _plot_heatmap(
        span_counts,
        title=f"{prefix} Span Coverage",
        save_path=os.path.join(output_dir, f"{prefix}_span_coverage.png"),
        cmap="magma",
    )


def plot_evaluation_artifacts(artifacts: EvaluationArtifacts, output_dir: str, prefix: str) -> None:
    _plot_span_focus_overview(artifacts, output_dir, prefix)


def plot_training_history(history: Iterable[Dict[str, float]], save_path: str) -> None:
    plt = _get_plt()
    history = list(history)
    if not history:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = [record["epoch"] for record in history]
    train_loss = [record.get("train/loss", np.nan) for record in history]
    clean_mae = [record.get("eval_clean/mae_total", np.nan) for record in history]
    noisy_mae = [record.get("eval_noisy/mae_total", np.nan) for record in history]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].plot(epochs, train_loss, marker="o", linewidth=1.7)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, clean_mae, marker="o", linewidth=1.7, label="clean")
    if not np.all(np.isnan(noisy_mae)):
        axes[1].plot(epochs, noisy_mae, marker="o", linewidth=1.7, label="noisy")
    axes[1].set_title("Eval MAE")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("mae_total (sec)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
