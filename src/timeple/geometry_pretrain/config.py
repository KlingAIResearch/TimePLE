"""Configuration helpers for duration-adaptive TimePLE-Codec pretraining."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional


@dataclass
class SyntheticDatasetConfig:
    span_u_bins: int = 64
    span_v_bins: int = 32
    span_samples_per_cell: int = 64
    duration_min_sec: float = 4.0
    duration_max_sec: float = 7200.0
    duration_distribution: str = "log_uniform"
    seed: Optional[int] = None

    def validate(self) -> None:
        if self.span_u_bins <= 0 or self.span_v_bins <= 0:
            raise ValueError("All sampling bin counts must be positive.")
        if self.span_samples_per_cell <= 0:
            raise ValueError("Samples per cell must be positive.")
        if self.duration_min_sec <= 0 or self.duration_max_sec <= 0:
            raise ValueError("Duration bounds must be positive.")
        if self.duration_max_sec < self.duration_min_sec:
            raise ValueError("duration_max_sec must be >= duration_min_sec.")
        if self.duration_distribution not in {"log_uniform", "uniform"}:
            raise ValueError(
                f"Unsupported duration_distribution={self.duration_distribution!r}. "
                "Expected one of: {'log_uniform', 'uniform'}."
            )


@dataclass
class GeometryPretrainConfig:
    output_dir: str = "timeple/output/geometry_pretrain_v1"
    codec_config_path: Optional[str] = "timeple/config/timeple_codec.json"
    epochs: int = 30
    batch_size: int = 512
    eval_batch_size: int = 1024
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    token_noise_std: float = 0.02
    eval_noise_std: float = 0.02
    clean_loss_weight: float = 1.0
    noisy_loss_weight: float = 0.5
    cycle_loss_weight: float = 0.1
    device: str = "auto"
    seed: int = 42
    viz_every_n_epochs: int = 5
    disable_plots: bool = False
    train_data: SyntheticDatasetConfig = field(default_factory=SyntheticDatasetConfig)
    eval_data: SyntheticDatasetConfig = field(
        default_factory=lambda: SyntheticDatasetConfig(
            span_samples_per_cell=16,
            seed=1042,
        )
    )

    def validate(self) -> None:
        if not self.output_dir:
            raise ValueError("output_dir must not be empty.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.batch_size <= 0 or self.eval_batch_size <= 0:
            raise ValueError("batch sizes must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0.")
        if self.lr <= 0:
            raise ValueError("lr must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0.")
        if self.grad_clip_norm < 0:
            raise ValueError("grad_clip_norm must be >= 0.")
        if self.token_noise_std < 0 or self.eval_noise_std < 0:
            raise ValueError("Noise std values must be >= 0.")
        if self.clean_loss_weight < 0 or self.noisy_loss_weight < 0 or self.cycle_loss_weight < 0:
            raise ValueError("Loss weights must be >= 0.")
        if self.viz_every_n_epochs <= 0:
            raise ValueError("viz_every_n_epochs must be positive.")
        if self.train_data.seed is None:
            self.train_data.seed = int(self.seed)
        if self.eval_data.seed is None:
            self.eval_data.seed = int(self.seed) + 10_000
        self.train_data.validate()
        self.eval_data.validate()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_mapping_file(path: str) -> Dict[str, Any]:
    if path is None:
        return {}
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext == ".json":
            data = json.load(f)
        elif ext in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as e:
                raise RuntimeError("Loading YAML codec configs requires `pyyaml` to be installed.") from e
            data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported codec config file format: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a mapping/dict, got {type(data)}.")
    return data


def load_codec_config_file(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    data = _load_mapping_file(path)
    if not isinstance(data, dict):
        raise ValueError(f"Codec config must be a mapping/dict, got {type(data)}.")
    return data


def _resolve_relative_path(base_dir: str, value: Optional[str]) -> Optional[str]:
    if value is None or not isinstance(value, str) or not value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(base_dir, value))


def _copy_known_keys(src: Mapping[str, Any], dst: Dict[str, Any], keys) -> None:
    for key in keys:
        if key in src:
            dst[key] = src[key]


def load_geometry_pretrain_config_file(path: str) -> Dict[str, Any]:
    raw = _load_mapping_file(path)
    base_dir = os.path.dirname(os.path.abspath(path))

    overrides: Dict[str, Any] = {}
    flat_keys = {
        "output_dir",
        "codec_config_path",
        "epochs",
        "batch_size",
        "eval_batch_size",
        "num_workers",
        "lr",
        "weight_decay",
        "grad_clip_norm",
        "token_noise_std",
        "eval_noise_std",
        "clean_loss_weight",
        "noisy_loss_weight",
        "cycle_loss_weight",
        "device",
        "seed",
        "viz_every_n_epochs",
        "disable_plots",
        "train_data",
        "eval_data",
    }
    _copy_known_keys(raw, overrides, flat_keys)

    model_cfg = raw.get("model", {}) or {}
    training_cfg = raw.get("training", {}) or {}
    loss_cfg = raw.get("loss", {}) or {}
    visualization_cfg = raw.get("visualization", {}) or {}
    data_cfg = raw.get("data", {}) or {}

    _copy_known_keys(model_cfg, overrides, {"codec_config_path"})
    _copy_known_keys(
        training_cfg,
        overrides,
        {
            "output_dir",
            "epochs",
            "batch_size",
            "eval_batch_size",
            "num_workers",
            "lr",
            "weight_decay",
            "grad_clip_norm",
            "device",
            "seed",
        },
    )
    _copy_known_keys(
        loss_cfg,
        overrides,
        {
            "token_noise_std",
            "eval_noise_std",
            "clean_loss_weight",
            "noisy_loss_weight",
            "cycle_loss_weight",
        },
    )
    _copy_known_keys(visualization_cfg, overrides, {"viz_every_n_epochs", "disable_plots"})

    if "train_data" in raw and isinstance(raw["train_data"], dict):
        overrides["train_data"] = dict(raw["train_data"])
    if "eval_data" in raw and isinstance(raw["eval_data"], dict):
        overrides["eval_data"] = dict(raw["eval_data"])

    if data_cfg:
        data_defaults = data_cfg.get("defaults", {}) or {}
        train_data_cfg = dict(data_defaults)
        train_data_cfg.update(data_cfg.get("train", {}) or {})
        eval_data_cfg = dict(data_defaults)
        eval_data_cfg.update(data_cfg.get("eval", {}) or {})
        if train_data_cfg:
            overrides["train_data"] = train_data_cfg
        if eval_data_cfg:
            overrides["eval_data"] = eval_data_cfg

    overrides["output_dir"] = _resolve_relative_path(base_dir, overrides.get("output_dir"))
    overrides["codec_config_path"] = _resolve_relative_path(base_dir, overrides.get("codec_config_path"))
    return overrides


def build_geometry_pretrain_config(overrides: Optional[Dict[str, Any]] = None) -> GeometryPretrainConfig:
    base = GeometryPretrainConfig()
    merged = asdict(base)
    overrides = overrides or {}

    for key, value in overrides.items():
        if value is None:
            continue
        if key in {"train_data", "eval_data"}:
            if not isinstance(value, Mapping):
                raise ValueError(f"{key} must be a mapping/dict, got {type(value)}.")
            nested = dict(merged[key])
            nested.update(dict(value))
            merged[key] = nested
        else:
            merged[key] = value

    merged["train_data"] = SyntheticDatasetConfig(**merged["train_data"])
    merged["eval_data"] = SyntheticDatasetConfig(**merged["eval_data"])
    config = GeometryPretrainConfig(**merged)
    config.validate()
    return config
