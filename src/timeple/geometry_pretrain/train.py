"""Training entrypoint for span-only CIS square-geometry pretraining."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from timeple.geometry_pretrain.config import (
    GeometryPretrainConfig,
    build_geometry_pretrain_config,
    load_codec_config_file,
    load_geometry_pretrain_config_file,
)
from timeple.geometry_pretrain.data import SyntheticSquareGeometryDataset
from timeple.geometry_pretrain.evaluate import SpanBucketSummary, evaluate_codec
from timeple.geometry_pretrain.visualize import (
    plot_dataset_coverage,
    plot_evaluation_artifacts,
    plot_training_history,
    save_artifacts_npz,
    save_history_json,
)
from timeple.models import TimePLECodec


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def make_run_output_dir(base_output_dir: str) -> str:
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_output_dir, run_timestamp)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    device = device.lower()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def prefix_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}/{key}": float(value) for key, value in metrics.items()}


def _gaussian_target_entropy_lower_bound(
    num_bins: int,
    sigma: float,
    *,
    num_samples: int = 4001,
) -> float:
    centers = np.linspace(0.0, 1.0, int(num_bins), dtype=np.float64)
    xs = np.linspace(0.0, 1.0, int(num_samples), dtype=np.float64)
    sigma = max(float(sigma), 1e-4)
    entropy_sum = 0.0
    for x in xs:
        weights = np.exp(-((centers - x) ** 2) / (2.0 * sigma * sigma))
        weights = weights / np.clip(weights.sum(), 1e-12, None)
        weights = np.clip(weights, 1e-12, None)
        entropy_sum += float(-(weights * np.log(weights)).sum())
    return entropy_sum / float(len(xs))


def compute_theoretical_loss_lower_bounds(codec: TimePLECodec) -> Dict[str, float]:
    grid_cfg = dict(codec.config["grid"])
    loss_cfg = dict(codec.config["loss"])

    span_u_loss_lb = _gaussian_target_entropy_lower_bound(
        num_bins=int(grid_cfg["num_u_bins"]),
        sigma=float(grid_cfg["span_sigma_u"]),
    )
    span_v_loss_lb = _gaussian_target_entropy_lower_bound(
        num_bins=int(grid_cfg["num_v_bins"]),
        sigma=float(grid_cfg["span_sigma_v"]),
    )
    span_loss_lb = span_u_loss_lb + span_v_loss_lb
    total_loss_lb = float(loss_cfg.get("lambda_span", 1.0)) * span_loss_lb

    return {
        "span_u_loss_lb": span_u_loss_lb,
        "span_v_loss_lb": span_v_loss_lb,
        "span_loss_lb": span_loss_lb,
        "total_loss_lb": total_loss_lb,
    }


def ratio_to_lower_bound(value: float, lower_bound: float) -> float:
    lower_bound = max(float(lower_bound), 1e-12)
    return float(value) / lower_bound


def _format_bucket_bound(value: float) -> str:
    value = float(value)
    if value >= 100.0:
        return f"{value:.0f}"
    if value >= 10.0:
        return f"{value:.1f}"
    return f"{value:.2f}"


def log_span_bucket_summary(tag: str, summary: SpanBucketSummary) -> None:
    segments: List[str] = []
    for idx, (lo, hi) in enumerate(zip(summary.edges[:-1], summary.edges[1:])):
        count = int(summary.counts[idx])
        if count <= 0:
            continue
        segments.append(
            f"{_format_bucket_bound(lo)}-{_format_bucket_bound(hi)}"
            f":n={count}"
            f",mae={summary.mae_total[idx]:.4f}"
            f",rel_span={summary.relative_mae_over_span[idx]:.4f}"
            f",iou={summary.span_iou[idx]:.4f}"
        )
    if segments:
        log(f"{tag} | " + " | ".join(segments))


def save_json(data, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl_record(path: str, record: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_dataloader(dataset, batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )


def save_checkpoint(
    codec: TimePLECodec,
    output_dir: str,
    name: str,
    *,
    epoch: int,
    metrics: Dict[str, float],
    config_dict: Dict,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    state_dict_path = os.path.join(output_dir, f"{name}_timeple_codec_state_dict.pt")
    checkpoint_path = os.path.join(output_dir, f"{name}_checkpoint.pt")
    torch.save(codec.state_dict(), state_dict_path)
    torch.save(
        {
            "epoch": epoch,
            "metrics": metrics,
            "codec_config": codec.config,
            "pretrain_config": config_dict,
            "model_state_dict": codec.state_dict(),
        },
        checkpoint_path,
    )


def train_one_epoch(
    codec: TimePLECodec,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: GeometryPretrainConfig,
    *,
    epoch: int,
) -> Dict[str, float]:
    codec.train()

    accum = {
        "loss": 0.0,
        "clean_total_loss": 0.0,
        "clean_mae_total": 0.0,
        "noisy_total_loss": 0.0,
        "cycle_loss": 0.0,
    }
    sample_count = 0
    total_steps = len(dataloader)
    log_every = max(total_steps // 10, 1)

    for step_idx, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)
        start_sec = batch["start_sec"]
        end_sec = batch["end_sec"]
        duration_sec = batch["duration_sec"]
        batch_size = int(start_sec.shape[0])
        sample_count += batch_size

        optimizer.zero_grad(set_to_none=True)

        clean_token = codec.encode(start_sec, end_sec, duration_sec)
        clean_loss_dict = codec.compute_loss(clean_token, start_sec, end_sec, duration_sec)
        loss = config.clean_loss_weight * clean_loss_dict["total_loss"]

        accum["clean_total_loss"] += float(clean_loss_dict["total_loss"].item()) * batch_size
        accum["clean_mae_total"] += float(clean_loss_dict["mae_total"].item()) * batch_size

        if config.noisy_loss_weight > 0 and config.token_noise_std > 0:
            noisy_token = clean_token + config.token_noise_std * torch.randn_like(clean_token)
            noisy_loss_dict = codec.compute_loss(noisy_token, start_sec, end_sec, duration_sec)
            loss = loss + config.noisy_loss_weight * noisy_loss_dict["total_loss"]
            accum["noisy_total_loss"] += float(noisy_loss_dict["total_loss"].item()) * batch_size

        if config.cycle_loss_weight > 0:
            reencoded_token = codec.reencode_from_token(clean_token, duration_sec, hard=False)
            cycle_loss = F.mse_loss(reencoded_token, clean_token.detach())
            loss = loss + config.cycle_loss_weight * cycle_loss
            accum["cycle_loss"] += float(cycle_loss.item()) * batch_size

        loss.backward()
        if config.grad_clip_norm > 0:
            clip_grad_norm_(codec.parameters(), max_norm=config.grad_clip_norm)
        optimizer.step()

        accum["loss"] += float(loss.item()) * batch_size

        if step_idx == 1 or step_idx % log_every == 0 or step_idx == total_steps:
            noisy_loss_value = float(noisy_loss_dict["total_loss"].item()) if (
                config.noisy_loss_weight > 0 and config.token_noise_std > 0
            ) else 0.0
            cycle_loss_value = float(cycle_loss.item()) if config.cycle_loss_weight > 0 else 0.0
            log(
                f"Epoch {epoch}/{config.epochs} step {step_idx}/{total_steps} "
                f"| loss={float(loss.item()):.4f} "
                f"| clean={float(clean_loss_dict['total_loss'].item()):.4f} "
                f"| noisy={noisy_loss_value:.4f} "
                f"| cycle={cycle_loss_value:.4f}"
            )

    denom = max(sample_count, 1)
    return {
        key: value / denom
        for key, value in accum.items()
    }


def build_config_from_sources(args_dict: Dict[str, Any]) -> GeometryPretrainConfig:
    config_path = args_dict.pop("config", None)
    overrides: Dict[str, Any] = {}

    if config_path is not None:
        config_path = os.path.abspath(config_path)
        overrides.update(load_geometry_pretrain_config_file(config_path))

    flat_keys = {
        "output_dir",
        "codec_config_path",
        "epochs",
        "batch_size",
        "eval_batch_size",
        "lr",
        "device",
        "seed",
        "disable_plots",
    }
    for key in flat_keys:
        if key in args_dict:
            overrides[key] = args_dict[key]

    return build_geometry_pretrain_config(overrides)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Span-only TimePLE-Codec square-geometry pretraining (YAML-first CLI)"
    )
    parser.add_argument("--config", type=str, default="timeple/config/geometry_pretrain_v1.yaml")
    parser.add_argument("--output_dir", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--codec_config_path", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--batch_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--eval_batch_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--device", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--disable_plots", action="store_true", default=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args_dict = vars(args)
    input_config_path = os.path.abspath(args_dict["config"]) if args_dict.get("config") else None
    config = build_config_from_sources(dict(args_dict))
    config.output_dir = make_run_output_dir(config.output_dir)

    os.makedirs(config.output_dir, exist_ok=True)
    save_json(asdict(config), os.path.join(config.output_dir, "pretrain_config.json"))
    if input_config_path is not None:
        shutil.copy2(input_config_path, os.path.join(config.output_dir, os.path.basename(input_config_path)))

    seed_everything(config.seed)
    device = resolve_device(config.device)

    codec_config = load_codec_config_file(config.codec_config_path)
    codec = TimePLECodec(codec_config).to(device)
    loss_lower_bounds = compute_theoretical_loss_lower_bounds(codec)

    train_dataset = SyntheticSquareGeometryDataset(config.train_data, codec.transform)
    eval_dataset = SyntheticSquareGeometryDataset(config.eval_data, codec.transform)
    train_loader = build_dataloader(train_dataset, config.batch_size, config.num_workers, device)
    eval_loader = build_dataloader(eval_dataset, config.eval_batch_size, config.num_workers, device)

    log(
        f"Starting span-only TimePLE-Codec geometry pretraining "
        f"| device={device} "
        f"| output_dir={config.output_dir}"
    )
    log(
        f"Resolved config "
        f"| codec_config_path={config.codec_config_path} "
        f"| epochs={config.epochs} "
        f"| batch_size={config.batch_size} "
        f"| eval_batch_size={config.eval_batch_size} "
        f"| lr={config.lr}"
    )
    log(
        f"Dataset summary "
        f"| train_samples={len(train_dataset)} "
        f"| eval_samples={len(eval_dataset)} "
        f"| train_steps_per_epoch={len(train_loader)} "
        f"| eval_steps={len(eval_loader)}"
    )
    log(
        f"Theoretical loss lower bounds "
        f"| span_u_loss_lb={loss_lower_bounds['span_u_loss_lb']:.4f} "
        f"| span_v_loss_lb={loss_lower_bounds['span_v_loss_lb']:.4f} "
        f"| span_loss_lb={loss_lower_bounds['span_loss_lb']:.4f} "
        f"| total_loss_lb={loss_lower_bounds['total_loss_lb']:.4f}"
    )

    save_json(codec.config, os.path.join(config.output_dir, "codec_config_resolved.json"))

    if not config.disable_plots:
        plot_dataset_coverage(
            train_dataset.planned_coverage(),
            os.path.join(config.output_dir, "plots"),
            prefix="train",
        )
        plot_dataset_coverage(
            eval_dataset.planned_coverage(),
            os.path.join(config.output_dir, "plots"),
            prefix="eval",
        )

    optimizer = torch.optim.AdamW(codec.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_mae_total = float("inf")
    best_record = None
    history: List[Dict[str, float]] = []
    metrics_jsonl = os.path.join(config.output_dir, "metrics.jsonl")

    for epoch in range(1, config.epochs + 1):
        log(f"Epoch {epoch}/{config.epochs} started")
        train_metrics = train_one_epoch(codec, train_loader, optimizer, device, config, epoch=epoch)
        should_visualize = (epoch % config.viz_every_n_epochs == 0) or (epoch == config.epochs)

        eval_clean = evaluate_codec(
            codec,
            eval_loader,
            device,
            noise_std=0.0,
            collect_artifacts=should_visualize,
        )
        eval_noisy = evaluate_codec(
            codec,
            eval_loader,
            device,
            noise_std=config.eval_noise_std,
            collect_artifacts=should_visualize,
        )

        record = {
            "epoch": epoch,
            **prefix_metrics(train_metrics, "train"),
            **prefix_metrics(eval_clean.metrics, "eval_clean"),
            **prefix_metrics(eval_noisy.metrics, "eval_noisy"),
        }
        history.append(record)
        save_jsonl_record(metrics_jsonl, record)
        save_history_json(history, os.path.join(config.output_dir, "history.json"))

        log(
            f"Epoch {epoch}/{config.epochs} finished "
            f"| train/loss={record['train/loss']:.4f} "
            f"| train/clean_total_loss={record['train/clean_total_loss']:.4f} "
            f"| eval_clean/total_loss={record['eval_clean/total_loss']:.4f} "
            f"| eval_noisy/total_loss={record['eval_noisy/total_loss']:.4f} "
            f"| eval_clean/span_loss={record['eval_clean/span_loss']:.4f} "
            f"| eval_clean/interval_l1={record['eval_clean/interval_l1']:.4f} "
            f"| eval_clean/square_l1={record['eval_clean/square_l1']:.4f} "
            f"| eval_clean/mae_total={record['eval_clean/mae_total']:.4f} "
            f"| eval_noisy/mae_total={record['eval_noisy/mae_total']:.4f} "
            f"| eval_clean/span_iou={record['eval_clean/span_iou']:.4f}"
        )
        log(
            f"Epoch {epoch}/{config.epochs} lower-bound ratios "
            f"| clean_total_vs_lb={ratio_to_lower_bound(record['eval_clean/total_loss'], loss_lower_bounds['total_loss_lb']):.4f}x "
            f"| noisy_total_vs_lb={ratio_to_lower_bound(record['eval_noisy/total_loss'], loss_lower_bounds['total_loss_lb']):.4f}x "
            f"| span_vs_lb={ratio_to_lower_bound(record['eval_clean/span_loss'], loss_lower_bounds['span_loss_lb']):.4f}x"
        )
        log_span_bucket_summary(
            "eval_clean relative_span_duration buckets",
            eval_clean.relative_span_duration_buckets,
        )
        log_span_bucket_summary(
            "eval_clean boundary_distance buckets",
            eval_clean.boundary_distance_buckets,
        )

        if should_visualize:
            if eval_clean.artifacts is not None:
                save_artifacts_npz(
                    eval_clean.artifacts,
                    os.path.join(config.output_dir, f"epoch_{epoch:03d}_clean_eval.npz"),
                )
            if eval_noisy.artifacts is not None:
                save_artifacts_npz(
                    eval_noisy.artifacts,
                    os.path.join(config.output_dir, f"epoch_{epoch:03d}_noisy_eval.npz"),
                )

        if should_visualize and not config.disable_plots:
            plots_dir = os.path.join(config.output_dir, "plots")
            if eval_clean.artifacts is not None:
                plot_evaluation_artifacts(eval_clean.artifacts, plots_dir, prefix=f"epoch_{epoch:03d}_clean")
            plot_training_history(history, os.path.join(plots_dir, "training_history.png"))

        current_mae_total = float(eval_clean.metrics["mae_total"])
        if current_mae_total < best_mae_total:
            best_mae_total = current_mae_total
            best_record = dict(record)
            save_checkpoint(
                codec,
                config.output_dir,
                "best",
                epoch=epoch,
                metrics=record,
                config_dict=asdict(config),
            )
            log(
                f"New best checkpoint "
                f"| epoch={epoch} "
                f"| eval_clean/mae_total={current_mae_total:.4f} "
                f"| path={os.path.join(config.output_dir, 'best_timeple_codec_state_dict.pt')}"
            )

    final_record = history[-1] if history else {}
    save_checkpoint(
        codec,
        config.output_dir,
        "final",
        epoch=config.epochs,
        metrics=final_record,
        config_dict=asdict(config),
    )
    if best_record is not None:
        save_json(best_record, os.path.join(config.output_dir, "best_metrics.json"))
    save_json(final_record, os.path.join(config.output_dir, "final_metrics.json"))
    log(
        f"Training completed "
        f"| final_checkpoint={os.path.join(config.output_dir, 'final_timeple_codec_state_dict.pt')} "
        f"| best_checkpoint={os.path.join(config.output_dir, 'best_timeple_codec_state_dict.pt')}"
    )


if __name__ == "__main__":
    main()
