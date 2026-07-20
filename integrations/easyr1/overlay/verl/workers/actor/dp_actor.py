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
"""
Implement Actor
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from collections import defaultdict
import inspect
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import average_loss, compute_kl, compute_policy_loss
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .tr_spd import compute_tr_spd_loss
from .config import ActorConfig


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]

TIME_CODEC_NON_TENSOR_KEYS = [
    "timestamp_labels",
    "timestamp_positions",
    "timestamp_video_durations",
    "timespan_labels",
    "timespan_positions",
    "timespan_video_durations",
]

TIMEED_SPAN_POLICY_KEYS = [
    "timeed_span_old_logits",
    "timeed_span_valid",
    "timeed_span_ref_logits",
    "timeed_text_response_rewards",
]

CSDO_POLICY_KEYS = [
    "csdo_old_logits",
    "csdo_old_features",
    "csdo_valid",
    "csdo_ref_logits",
]

TR_SPD_POLICY_KEYS = [
    "tr_spd_old_logits",
    "tr_spd_old_features",
    "tr_spd_valid",
    "tr_spd_text_response_rewards",
]


def _build_time_codec_kwargs(micro_batch: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}

    def _to_list(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def _build_label_dict(prefix: str) -> Optional[dict[str, Any]]:
        label_key = f"{prefix}_labels"
        duration_key = f"{prefix}_video_durations"

        labels = _to_list(micro_batch.get(label_key))
        explicit_durations = _to_list(micro_batch.get(duration_key))

        def _normalize_duration_value(value: Any) -> list[Any]:
            if value is None:
                return []
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif hasattr(value, "tolist") and not isinstance(value, (list, tuple, dict)):
                value = value.tolist()
            if isinstance(value, tuple):
                value = list(value)
            if isinstance(value, list):
                return value
            return [value]

        if isinstance(labels, list) and len(labels) > 0 and isinstance(labels[0], dict):
            label_dict = {
                "start": [item.get("start", []) for item in labels],
                "end": [item.get("end", []) for item in labels],
            }
            if any(item.get("video_duration") is not None for item in labels):
                label_dict["video_duration"] = [
                    _normalize_duration_value(item.get("video_duration")) for item in labels
                ]
            elif explicit_durations is not None:
                label_dict["video_duration"] = explicit_durations
            return label_dict

        if isinstance(labels, dict):
            label_dict = dict(labels)
            if "video_duration" not in label_dict and explicit_durations is not None:
                label_dict["video_duration"] = explicit_durations
            return label_dict

        return None

    timestamp_labels = _build_label_dict("timestamp")
    if timestamp_labels is not None:
        kwargs["timestamp_labels"] = timestamp_labels

    timestamp_positions = _to_list(micro_batch.get("timestamp_positions"))
    if timestamp_positions is not None:
        kwargs["timestamp_positions"] = timestamp_positions
    timestamp_video_durations = _to_list(micro_batch.get("timestamp_video_durations"))
    if timestamp_video_durations is not None:
        kwargs["timestamp_video_durations"] = timestamp_video_durations

    timespan_labels = _build_label_dict("timespan")
    if timespan_labels is not None:
        kwargs["timespan_labels"] = timespan_labels

    timespan_positions = _to_list(micro_batch.get("timespan_positions"))
    if timespan_positions is not None:
        kwargs["timespan_positions"] = timespan_positions
    timespan_video_durations = _to_list(micro_batch.get("timespan_video_durations"))
    if timespan_video_durations is not None:
        kwargs["timespan_video_durations"] = timespan_video_durations

    return kwargs


def _normalize_nested_sequences(value: Any) -> list[list[Any]]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif hasattr(value, "tolist") and not isinstance(value, (list, tuple, dict)):
        value = value.tolist()

    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return [[value]]
    if len(value) == 0:
        return []

    first = value[0]
    if isinstance(first, (list, tuple)):
        return [list(item) for item in value]
    return [list(value)]


def _flatten_time_codec_kwargs_for_padding_free(
    time_codec_kwargs: dict[str, Any],
    *,
    batch_size: int,
    seqlen: int,
    indices: torch.Tensor,
) -> dict[str, Any]:
    flat_index_to_unpadded = torch.full((batch_size * seqlen,), -1, device=indices.device, dtype=torch.long)
    flat_index_to_unpadded[indices] = torch.arange(indices.numel(), device=indices.device)

    flattened_kwargs: dict[str, Any] = {
        key: value
        for key, value in time_codec_kwargs.items()
        if not key.endswith("_labels") and not key.endswith("_positions") and not key.endswith("_video_durations")
    }
    for prefix in ("timestamp", "timespan"):
        labels = time_codec_kwargs.get(f"{prefix}_labels")
        positions = time_codec_kwargs.get(f"{prefix}_positions")
        if not isinstance(labels, dict) or positions is None:
            continue

        starts_per_sample = _normalize_nested_sequences(labels.get("start"))
        ends_per_sample = _normalize_nested_sequences(labels.get("end"))
        durations_per_sample = _normalize_nested_sequences(
            labels.get("video_duration", time_codec_kwargs.get(f"{prefix}_video_durations"))
        )
        positions_per_sample = _normalize_nested_sequences(positions)

        flattened_entries: list[tuple[int, float, float, Optional[float]]] = []
        for sample_idx in range(batch_size):
            starts = starts_per_sample[sample_idx] if sample_idx < len(starts_per_sample) else []
            ends = ends_per_sample[sample_idx] if sample_idx < len(ends_per_sample) else []
            durations = durations_per_sample[sample_idx] if sample_idx < len(durations_per_sample) else []
            sample_positions = positions_per_sample[sample_idx] if sample_idx < len(positions_per_sample) else []
            count = min(len(starts), len(ends), len(sample_positions))

            for item_idx in range(count):
                position = int(sample_positions[item_idx])
                if position < 0 or position >= seqlen:
                    raise ValueError(
                        f"{prefix}_positions contains out-of-range index {position} for sequence length {seqlen}."
                    )

                flat_index = sample_idx * seqlen + position
                unpadded_position = int(flat_index_to_unpadded[flat_index].item())
                if unpadded_position < 0:
                    raise ValueError(
                        f"{prefix}_positions points to a padded token after padding-free compaction: "
                        f"sample_idx={sample_idx}, position={position}."
                    )

                flattened_entries.append(
                    (
                        unpadded_position,
                        float(starts[item_idx]),
                        float(ends[item_idx]),
                        float(durations[min(item_idx, len(durations) - 1)]) if len(durations) > 0 else None,
                    )
                )

        if not flattened_entries:
            continue

        flattened_entries.sort(key=lambda entry: entry[0])
        flattened_kwargs[f"{prefix}_labels"] = {
            "start": [[entry[1] for entry in flattened_entries]],
            "end": [[entry[2] for entry in flattened_entries]],
        }
        flattened_durations = [entry[3] for entry in flattened_entries if entry[3] is not None]
        if len(flattened_durations) == len(flattened_entries):
            flattened_kwargs[f"{prefix}_labels"]["video_duration"] = [flattened_durations]
            flattened_kwargs[f"{prefix}_video_durations"] = [flattened_durations]
        flattened_kwargs[f"{prefix}_positions"] = [[entry[0] for entry in flattened_entries]]

    return flattened_kwargs


def _is_timecodec_model(model: nn.Module) -> bool:
    model_config = getattr(model, "config", None)
    return getattr(model_config, "model_type", None) == "qwen3_vl_time_codec" and bool(
        getattr(model, "use_time_codec", False)
    )


def _is_cis_model(model: nn.Module) -> bool:
    model_config = getattr(model, "config", None)
    return getattr(model_config, "model_type", None) == "qwen3_vl_cis_codec" and bool(
        getattr(model, "use_cis_codec", False)
    )


def _is_timeple_model(model: nn.Module) -> bool:
    model_config = getattr(model, "config", None)
    return getattr(model_config, "model_type", None) in {"qwen3_vl_timeple_codec", "qwen3_vl_timeple"} and bool(
        getattr(model, "use_timeple_codec", False)
    )


def _is_timeed_model(model: nn.Module) -> bool:
    model_config = getattr(model, "config", None)
    return getattr(model_config, "model_type", None) == "qwen3_vl_timeed" and bool(
        getattr(model, "use_timeed", False)
    )


def _unwrap_model(module: nn.Module) -> nn.Module:
    return getattr(module, "module", module)


def _timeed_forward_supports_span_policy(model: nn.Module) -> bool:
    try:
        return "timeed_span_sample_positions" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def _segment_iou(
    pred_start: float,
    pred_end: float,
    target_start: float,
    target_end: float,
) -> float:
    if pred_end < pred_start:
        pred_start, pred_end = pred_end, pred_start
    if target_end < target_start:
        target_start, target_end = target_end, target_start
    if pred_end <= pred_start or target_end <= target_start:
        return 0.0

    intersection = max(0.0, min(pred_end, target_end) - max(pred_start, target_start))
    union = max(pred_end, target_end) - min(pred_start, target_start)
    if union <= 0.0:
        return 0.0
    return intersection / union


def _build_empty_timecodec_features(
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    zeros = torch.zeros(batch_size, device=device, dtype=torch.float32)
    features = {
        "timecodec_timespan_count": zeros.clone(),
        "timecodec_metrics_valid": zeros.clone(),
        "timecodec_decoded_valid": zeros.clone(),
        "timecodec_decode_source": zeros.clone(),
        "timecodec_tpfd_total_loss": zeros.clone(),
        "timecodec_tpfd_global_loss": zeros.clone(),
        "timecodec_tpfd_start_loss": zeros.clone(),
        "timecodec_tpfd_end_loss": zeros.clone(),
        "timecodec_embedding_cosine": zeros.clone(),
        "timecodec_decoded_iou": zeros.clone(),
        "timecodec_pred_start": zeros.clone(),
        "timecodec_pred_end": zeros.clone(),
    }
    features.update(
        {
            "cis_timespan_count": zeros.clone(),
            "cis_metrics_valid": zeros.clone(),
            "cis_decoded_valid": zeros.clone(),
            "cis_decode_source": zeros.clone(),
            "cis_total_loss": zeros.clone(),
            "cis_type_loss": zeros.clone(),
            "cis_point_loss": zeros.clone(),
            "timeple_loss": zeros.clone(),
            "cis_interval_l1": zeros.clone(),
            "cis_point_l1": zeros.clone(),
            "timeple_giou_loss": zeros.clone(),
            "timeple_iou": zeros.clone(),
            "cis_mae_start": zeros.clone(),
            "cis_mae_end": zeros.clone(),
            "cis_mae_total": zeros.clone(),
            "cis_embedding_valid": zeros.clone(),
            "cis_embedding_cosine": zeros.clone(),
            "cis_embedding_mse_loss": zeros.clone(),
            "cis_reencoding_loss": zeros.clone(),
            "cis_base_path_decode_loss": zeros.clone(),
            "cis_base_vs_full_gap": zeros.clone(),
            "cis_decoded_iou": zeros.clone(),
            "cis_pred_start": zeros.clone(),
            "cis_pred_end": zeros.clone(),
        }
    )
    features.update(
        {
            "timeple_timespan_count": zeros.clone(),
            "timeple_metrics_valid": zeros.clone(),
            "timeple_decoded_valid": zeros.clone(),
            "timeple_decode_source": zeros.clone(),
            "timeple_total_loss": zeros.clone(),
            "timeple_dfl_loss": zeros.clone(),
            "timeple_iou_loss": zeros.clone(),
            "timeple_span_loss": zeros.clone(),
            "timeple_interval_l1": zeros.clone(),
            "timeple_boundary_loss": zeros.clone(),
            "timeple_span_giou_loss": zeros.clone(),
            "timeple_span_iou": zeros.clone(),
            "timeple_mae_start": zeros.clone(),
            "timeple_mae_end": zeros.clone(),
            "timeple_mae_total": zeros.clone(),
            "timeple_decoded_iou": zeros.clone(),
            "timeple_pred_start": zeros.clone(),
            "timeple_pred_end": zeros.clone(),
            "timeple_debug_is_model": zeros.clone(),
            "timeple_debug_has_timespan_token_id": zeros.clone(),
            "timeple_debug_has_codec_module": zeros.clone(),
            "timeple_debug_has_timespan_labels": zeros.clone(),
            "timeple_debug_target_count": zeros.clone(),
            "timeple_debug_sample_info_count": zeros.clone(),
            "timeple_debug_generated_position_count": zeros.clone(),
            "timeple_debug_forward_has_timespan_positions": zeros.clone(),
            "timeple_debug_hidden_available": zeros.clone(),
            "timeple_debug_decode_entered": zeros.clone(),
            "timeple_debug_decode_position_count": zeros.clone(),
            "timeple_debug_decode_prediction_count": zeros.clone(),
            "timeple_debug_decode_populated": zeros.clone(),
        }
    )
    features.update(
        {
            "timeed_timespan_count": zeros.clone(),
            "timeed_metrics_valid": zeros.clone(),
            "timeed_decoded_valid": zeros.clone(),
            "timeed_decode_source": zeros.clone(),
            "timeed_loss_total": zeros.clone(),
            "timeed_timespan_loss_total": zeros.clone(),
            "timeed_timespan_loss_dfl": zeros.clone(),
            "timeed_timespan_loss_giou": zeros.clone(),
            "timeed_span_iou": zeros.clone(),
            "timeed_span_iou_argmax": zeros.clone(),
            "timeed_mae_start": zeros.clone(),
            "timeed_mae_end": zeros.clone(),
            "timeed_mae_total": zeros.clone(),
            "timeed_distribution_entropy": zeros.clone(),
            "timeed_invalid_span_rate": zeros.clone(),
            "timeed_decoded_iou": zeros.clone(),
            "timeed_pred_start": zeros.clone(),
            "timeed_pred_end": zeros.clone(),
        }
    )
    return features


def _build_empty_timecodec_non_tensor_features(*, batch_size: int) -> dict[str, list[Any]]:
    return {
        "cis_decoded_segments": [[] for _ in range(batch_size)],
        "timeple_decoded_segments": [[] for _ in range(batch_size)],
        "timeed_decoded_segments": [[] for _ in range(batch_size)],
    }


def _as_1d_object_array(values: list[Any]) -> np.ndarray:
    array = np.empty(len(values), dtype=object)
    for idx, value in enumerate(values):
        array[idx] = value
    return array


def _maybe_to_float_scalar(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _count_aligned_generated_timespans(sample_infos: list[dict[str, Any]]) -> int:
    aligned_count = 0
    for sample_info in sample_infos:
        candidate_positions = sample_info.get("candidate_positions")
        candidate_count = int(candidate_positions.numel()) if isinstance(candidate_positions, torch.Tensor) else 0
        aligned_count += min(
            candidate_count,
            len(sample_info.get("target_starts", [])),
            len(sample_info.get("target_ends", [])),
        )
    return aligned_count


def _build_clean_cis_aux_metrics_from_forward(
    *,
    output: Any,
    aligned_count: int,
) -> tuple[Optional[torch.Tensor], dict[str, float]]:
    metrics: dict[str, float] = {
        "cis_aux_valid": 0.0,
        "cis_aux_loss": 0.0,
    }
    cis_loss = getattr(output, "cis_loss", None)
    if not isinstance(cis_loss, torch.Tensor):
        return None, metrics

    cis_loss = cis_loss.float()
    metrics.update(
        {
            "cis_aux_valid": 1.0,
            "cis_aux_loss": float(cis_loss.detach().item()),
            "cis_aux_aligned_count": float(aligned_count),
        }
    )

    cis_loss_details = getattr(output, "cis_loss_details", None)
    if isinstance(cis_loss_details, dict):
        detail_mapping = {
            "cis_aux_embedding_mse_loss": "embedding_mse_loss",
            "cis_aux_embedding_cosine_loss": "embedding_cosine_loss",
            "cis_aux_decoded_iou_loss": "codec_decoded_iou_loss",
            "cis_aux_decoded_iou": "codec_decoded_iou",
            "cis_aux_interval_l1": "codec_interval_l1",
        }
        for metric_key, detail_key in detail_mapping.items():
            scalar = _maybe_to_float_scalar(cis_loss_details.get(detail_key))
            if scalar is not None:
                metrics[metric_key] = scalar

        cosine_loss = _maybe_to_float_scalar(cis_loss_details.get("embedding_cosine_loss"))
        if cosine_loss is not None:
            metrics["cis_aux_embedding_cosine"] = max(-1.0, min(1.0, 1.0 - cosine_loss))

    return cis_loss, metrics


def _populate_timecodec_features_from_forward_details(
    aux_features: dict[str, torch.Tensor],
    *,
    time_loss_details: Any,
    sample_idx: int,
) -> bool:
    if not isinstance(time_loss_details, dict):
        return False

    populated = False
    detail_mapping = {
        "timecodec_tpfd_total_loss": "tpfd_total_loss",
        "timecodec_tpfd_global_loss": "tpfd_global_loss",
        "timecodec_tpfd_start_loss": "tpfd_start_loss",
        "timecodec_tpfd_end_loss": "tpfd_end_loss",
    }
    for aux_key, detail_key in detail_mapping.items():
        scalar = _maybe_to_float_scalar(time_loss_details.get(detail_key))
        if scalar is not None:
            aux_features[aux_key][sample_idx] = scalar
            populated = True

    embedding_cosine_loss = _maybe_to_float_scalar(time_loss_details.get("embedding_cosine_loss"))
    if embedding_cosine_loss is not None:
        aux_features["timecodec_embedding_cosine"][sample_idx] = max(-1.0, min(1.0, 1.0 - embedding_cosine_loss))
        populated = True

    return populated


def _populate_cis_features_from_forward_details(
    aux_features: dict[str, torch.Tensor],
    *,
    cis_loss_details: Any,
    sample_idx: int,
) -> bool:
    if not isinstance(cis_loss_details, dict):
        return False

    populated = False
    detail_mapping = {
        "cis_total_loss": ("codec_total_loss", "full_path_decode_loss"),
        "cis_type_loss": ("codec_type_loss",),
        "cis_point_loss": ("codec_point_loss",),
        "timeple_loss": ("codec_span_loss",),
        "cis_interval_l1": ("codec_interval_l1",),
        "cis_point_l1": ("codec_point_l1",),
        "timeple_giou_loss": ("codec_span_giou_loss",),
        "timeple_iou": ("codec_span_iou", "span_iou"),
        "cis_mae_start": ("codec_mae_start",),
        "cis_mae_end": ("codec_mae_end",),
        "cis_mae_total": ("codec_mae_total", "cis_mae_total"),
        "cis_embedding_mse_loss": ("embedding_mse_loss",),
        "cis_reencoding_loss": ("reencoding_loss",),
        "cis_base_path_decode_loss": ("base_path_decode_loss",),
        "cis_base_vs_full_gap": ("base_vs_full_gap",),
    }

    embedding_populated = False
    for aux_key, detail_keys in detail_mapping.items():
        scalar = None
        for detail_key in detail_keys:
            scalar = _maybe_to_float_scalar(cis_loss_details.get(detail_key))
            if scalar is not None:
                break
        if scalar is None:
            continue
        aux_features[aux_key][sample_idx] = scalar
        if aux_key == "cis_embedding_mse_loss":
            embedding_populated = True
        populated = True

    embedding_cosine_loss = _maybe_to_float_scalar(cis_loss_details.get("embedding_cosine_loss"))
    if embedding_cosine_loss is not None:
        aux_features["cis_embedding_cosine"][sample_idx] = max(-1.0, min(1.0, 1.0 - embedding_cosine_loss))
        embedding_populated = True
        populated = True
    if embedding_populated:
        aux_features["cis_embedding_valid"][sample_idx] = 1.0

    return populated


def _populate_timeple_features_from_forward_details(
    aux_features: dict[str, torch.Tensor],
    *,
    timeple_loss_details: Any,
    sample_idx: int,
) -> bool:
    if not isinstance(timeple_loss_details, dict):
        return False

    populated = False
    detail_mapping = {
        "timeple_total_loss": ("codec_total_loss", "full_path_decode_loss", "timeple_loss"),
        "timeple_dfl_loss": ("codec_dfl_loss", "codec_span_loss"),
        "timeple_iou_loss": ("codec_iou_loss",),
        "timeple_span_loss": ("codec_span_loss",),
        "timeple_interval_l1": ("codec_interval_l1",),
        "timeple_boundary_loss": ("codec_boundary_loss",),
        "timeple_span_giou_loss": ("codec_span_giou_loss",),
        "timeple_span_iou": ("codec_span_iou", "span_iou"),
        "timeple_mae_start": ("codec_mae_start",),
        "timeple_mae_end": ("codec_mae_end",),
        "timeple_mae_total": ("codec_mae_total", "timeple_mae_total"),
    }
    for aux_key, detail_keys in detail_mapping.items():
        scalar = None
        for detail_key in detail_keys:
            scalar = _maybe_to_float_scalar(timeple_loss_details.get(detail_key))
            if scalar is not None:
                break
        if scalar is None:
            continue
        aux_features[aux_key][sample_idx] = scalar
        populated = True

    return populated


def _populate_timeple_features_from_forward_details(
    aux_features: dict[str, torch.Tensor],
    *,
    timeple_loss_details: Any,
    sample_idx: int,
) -> bool:
    if not isinstance(timeple_loss_details, dict):
        return False

    populated = False
    detail_mapping = {
        "timeple_total_loss": (
            "codec_total_loss",
            "full_path_decode_loss",
            "timeple_loss",
        ),
        "timeple_dfl_loss": ("codec_dfl_loss", "codec_span_loss"),
        "timeple_iou_loss": ("codec_iou_loss",),
        "timeple_span_loss": ("codec_span_loss",),
        "timeple_interval_l1": ("codec_interval_l1",),
        "timeple_boundary_loss": ("codec_boundary_loss",),
        "timeple_span_giou_loss": ("codec_span_giou_loss",),
        "timeple_span_iou": ("codec_span_iou", "span_iou"),
        "timeple_mae_start": ("codec_mae_start",),
        "timeple_mae_end": ("codec_mae_end",),
        "timeple_mae_total": ("codec_mae_total", "timeple_mae_total"),
    }
    for aux_key, detail_keys in detail_mapping.items():
        scalar = None
        for detail_key in detail_keys:
            scalar = _maybe_to_float_scalar(timeple_loss_details.get(detail_key))
            if scalar is not None:
                break
        if scalar is None:
            continue
        aux_features[aux_key][sample_idx] = scalar
        populated = True

    return populated


def _populate_timeed_features_from_forward_details(
    aux_features: dict[str, torch.Tensor],
    *,
    timeed_loss_details: Any,
    sample_idx: int,
) -> bool:
    if not isinstance(timeed_loss_details, dict):
        return False

    populated = False
    detail_mapping = {
        "timeed_loss_total": ("loss_total",),
        "timeed_timespan_loss_total": ("timespan_loss_total",),
        "timeed_timespan_loss_dfl": ("timespan_loss_dfl",),
        "timeed_timespan_loss_giou": ("timespan_loss_giou",),
        "timeed_span_iou": ("timespan_span_iou_expectation", "timespan_span_iou"),
        "timeed_span_iou_argmax": ("timespan_span_iou_argmax",),
        "timeed_mae_start": ("timespan_mae_start",),
        "timeed_mae_end": ("timespan_mae_end",),
        "timeed_mae_total": ("timespan_mae_total",),
        "timeed_distribution_entropy": ("timespan_distribution_entropy",),
        "timeed_invalid_span_rate": ("timespan_invalid_span_rate",),
    }
    for aux_key, detail_keys in detail_mapping.items():
        scalar = None
        for detail_key in detail_keys:
            scalar = _maybe_to_float_scalar(timeed_loss_details.get(detail_key))
            if scalar is not None:
                break
        if scalar is None:
            continue
        aux_features[aux_key][sample_idx] = scalar
        populated = True

    return populated


def _extract_timespan_labels_per_sample(micro_batch: dict[str, Any]) -> tuple[list[list[float]], list[list[float]]]:
    def _to_list(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    timespan_labels = _to_list(micro_batch.get("timespan_labels"))
    if isinstance(timespan_labels, list) and len(timespan_labels) > 0 and isinstance(timespan_labels[0], dict):
        starts_per_sample = [list(item.get("start", [])) for item in timespan_labels]
        ends_per_sample = [list(item.get("end", [])) for item in timespan_labels]
        return starts_per_sample, ends_per_sample
    if isinstance(timespan_labels, dict):
        return _normalize_nested_sequences(timespan_labels.get("start")), _normalize_nested_sequences(
            timespan_labels.get("end")
        )
    return [], []


def _extract_timespan_video_durations_per_sample(micro_batch: dict[str, Any]) -> list[list[float]]:
    def _to_list(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    timespan_labels = _to_list(micro_batch.get("timespan_labels"))
    if isinstance(timespan_labels, list) and len(timespan_labels) > 0 and isinstance(timespan_labels[0], dict):
        durations_per_sample: list[list[float]] = []
        for item in timespan_labels:
            sample_duration = item.get("video_duration", [])
            if isinstance(sample_duration, np.ndarray):
                sample_duration = sample_duration.tolist()
            elif hasattr(sample_duration, "tolist") and not isinstance(sample_duration, (list, tuple, dict)):
                sample_duration = sample_duration.tolist()
            if isinstance(sample_duration, tuple):
                sample_duration = list(sample_duration)
            if isinstance(sample_duration, list):
                durations_per_sample.append([float(value) for value in sample_duration])
            elif sample_duration is None:
                durations_per_sample.append([])
            else:
                durations_per_sample.append([float(sample_duration)])
        return durations_per_sample
    if isinstance(timespan_labels, dict) and "video_duration" in timespan_labels:
        return _normalize_nested_sequences(timespan_labels.get("video_duration"))

    explicit_durations = _to_list(micro_batch.get("timespan_video_durations"))
    if explicit_durations is not None:
        return _normalize_nested_sequences(explicit_durations)

    return []


def _resolve_decode_video_durations(
    *,
    target_durations: list[float],
    target_ends: list[float],
    prediction_count: int,
) -> list[float]:
    if prediction_count <= 0:
        return []
    if len(target_durations) > 0:
        return [float(target_durations[0])] * prediction_count
    if len(target_ends) > 0:
        return [max(float(max(target_ends)), 1.0)] * prediction_count
    return []


def _get_valid_response_length(
    response_mask: Optional[torch.Tensor],
    *,
    sample_idx: int,
    response_length: int,
) -> int:
    if response_mask is None:
        return response_length
    return int(response_mask[sample_idx].sum().item())


def _build_generated_timespan_supervision(
    *,
    micro_batch: dict[str, Any],
    responses: torch.Tensor,
    response_mask: Optional[torch.Tensor],
    timespan_token_id: int,
    response_offset: int,
    aux_features: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    batch_size, response_length = responses.shape
    starts_per_sample, ends_per_sample = _extract_timespan_labels_per_sample(micro_batch)
    durations_per_sample = _extract_timespan_video_durations_per_sample(micro_batch)

    generated_positions: list[list[int]] = []
    generated_starts: list[list[float]] = []
    generated_ends: list[list[float]] = []
    generated_durations: list[list[float]] = []
    sample_infos: list[dict[str, Any]] = []

    for sample_idx in range(batch_size):
        valid_response_len = _get_valid_response_length(
            response_mask,
            sample_idx=sample_idx,
            response_length=response_length,
        )
        if valid_response_len <= 0:
            generated_positions.append([])
            generated_starts.append([])
            generated_ends.append([])
            generated_durations.append([])
            continue

        candidate_positions = torch.nonzero(
            responses[sample_idx, :valid_response_len] == timespan_token_id,
            as_tuple=False,
        ).flatten()
        aux_features["timecodec_timespan_count"][sample_idx] = float(candidate_positions.numel())
        aux_features["cis_timespan_count"][sample_idx] = float(candidate_positions.numel())
        aux_features["timeple_timespan_count"][sample_idx] = float(candidate_positions.numel())
        aux_features["timeed_timespan_count"][sample_idx] = float(candidate_positions.numel())

        target_starts = starts_per_sample[sample_idx] if sample_idx < len(starts_per_sample) else []
        target_ends = ends_per_sample[sample_idx] if sample_idx < len(ends_per_sample) else []
        target_durations = durations_per_sample[sample_idx] if sample_idx < len(durations_per_sample) else []
        target_count = min(len(target_starts), len(target_ends))
        matched_count = min(int(candidate_positions.numel()), target_count)
        if "timeple_debug_has_timespan_labels" in aux_features:
            aux_features["timeple_debug_has_timespan_labels"][sample_idx] = 1.0 if target_count > 0 else 0.0
            aux_features["timeple_debug_target_count"][sample_idx] = float(target_count)
            aux_features["timeple_debug_generated_position_count"][sample_idx] = float(matched_count)

        aligned_positions = [
            int(response_offset + int(rel_position.item())) for rel_position in candidate_positions[:matched_count]
        ]
        aligned_starts = [float(target_starts[target_idx]) for target_idx in range(matched_count)]
        aligned_ends = [float(target_ends[target_idx]) for target_idx in range(matched_count)]
        aligned_durations = (
            [
                float(target_durations[min(target_idx, len(target_durations) - 1)])
                for target_idx in range(matched_count)
            ]
            if len(target_durations) > 0
            else []
        )

        generated_positions.append(aligned_positions)
        generated_starts.append(aligned_starts)
        generated_ends.append(aligned_ends)
        generated_durations.append(aligned_durations)

        if candidate_positions.numel() == 0 or target_count == 0:
            continue

        sample_infos.append(
            {
                "sample_idx": sample_idx,
                "candidate_positions": candidate_positions,
                "target_starts": target_starts,
                "target_ends": target_ends,
                "target_durations": target_durations,
                "target_count": target_count,
            }
        )

    generated_time_codec_kwargs: dict[str, Any] = {}
    if any(len(sample_positions) > 0 for sample_positions in generated_positions):
        generated_time_codec_kwargs["timespan_positions"] = generated_positions
        generated_time_codec_kwargs["timespan_labels"] = {
            "start": generated_starts,
            "end": generated_ends,
        }
        if any(len(sample_durations) > 0 for sample_durations in generated_durations):
            generated_time_codec_kwargs["timespan_labels"]["video_duration"] = generated_durations
            generated_time_codec_kwargs["timespan_video_durations"] = generated_durations

    return generated_time_codec_kwargs, sample_infos


def _slice_time_codec_kwargs_for_sample(time_codec_kwargs: dict[str, Any], sample_idx: int) -> dict[str, Any]:
    sliced_kwargs: dict[str, Any] = {
        key: value
        for key, value in time_codec_kwargs.items()
        if not key.endswith("_labels") and not key.endswith("_positions") and not key.endswith("_video_durations")
    }
    for prefix in ("timestamp", "timespan"):
        labels = time_codec_kwargs.get(f"{prefix}_labels")
        if isinstance(labels, dict):
            starts_per_sample = _normalize_nested_sequences(labels.get("start"))
            ends_per_sample = _normalize_nested_sequences(labels.get("end"))
            sliced_label_dict = {
                "start": [starts_per_sample[sample_idx] if sample_idx < len(starts_per_sample) else []],
                "end": [ends_per_sample[sample_idx] if sample_idx < len(ends_per_sample) else []],
            }
            durations_per_sample = _normalize_nested_sequences(
                labels.get("video_duration", time_codec_kwargs.get(f"{prefix}_video_durations"))
            )
            if len(durations_per_sample) > 0:
                sliced_label_dict["video_duration"] = [
                    durations_per_sample[sample_idx] if sample_idx < len(durations_per_sample) else []
                ]
                sliced_kwargs[f"{prefix}_video_durations"] = sliced_label_dict["video_duration"]
            sliced_kwargs[f"{prefix}_labels"] = sliced_label_dict

        positions = time_codec_kwargs.get(f"{prefix}_positions")
        if positions is None:
            continue
        positions_per_sample = _normalize_nested_sequences(positions)
        sliced_kwargs[f"{prefix}_positions"] = [
            positions_per_sample[sample_idx] if sample_idx < len(positions_per_sample) else []
        ]

    return sliced_kwargs


def _slice_multi_modal_inputs_for_sample(multi_modal_inputs: dict[str, Any], sample_idx: int) -> dict[str, Any]:
    sliced_inputs: dict[str, Any] = {}
    for key, value in multi_modal_inputs.items():
        if isinstance(value, torch.Tensor):
            sliced_inputs[key] = value[sample_idx : sample_idx + 1]
        else:
            sliced_inputs[key] = value
    return sliced_inputs


def _codec_param_context(actor_module: nn.Module):
    if isinstance(actor_module, FSDP):
        # Decoding only touches root-level CIS/time codec adapters, not transformer
        # layer parameters. Recursing here unshards the full 8B actor and can OOM.
        return FSDP.summon_full_params(actor_module, writeback=False, recurse=False)
    return nullcontext()


def _csdo_forward_supports_outputs(model: nn.Module) -> bool:
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False
    return "compute_csdo_outputs" in parameters and "csdo_positions" in parameters


def _tr_spd_forward_supports_outputs(model: nn.Module) -> bool:
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False
    return "compute_tr_spd_outputs" in parameters and "tr_spd_positions" in parameters


def _decode_timecodec_predictions_from_hidden_states(
    *,
    actor_module: nn.Module,
    model: nn.Module,
    hidden_states: torch.Tensor,
    time_token_positions: list[int],
    video_durations: Optional[list[float]] = None,
    summon_full_params: bool = True,
) -> list[tuple[float, float]]:
    if len(time_token_positions) == 0:
        return []

    context = _codec_param_context(actor_module) if summon_full_params else nullcontext()
    with context:
        if _is_timeple_model(model):
            if video_durations is None or len(video_durations) == 0:
                return []
            pred_starts, pred_ends = model.decode_timeple_from_hidden_states(
                hidden_states=hidden_states,
                timeple_token_positions=time_token_positions,
                video_duration_sec=video_durations,
                hard=True,
            )
        elif _is_cis_model(model):
            if video_durations is None or len(video_durations) == 0:
                return []
            pred_starts, pred_ends = model.decode_cis_from_hidden_states(
                hidden_states=hidden_states,
                cis_token_positions=time_token_positions,
                video_duration_sec=video_durations,
                hard=True,
            )
        elif _is_timeed_model(model):
            if video_durations is None or len(video_durations) == 0:
                return []
            pred_starts, pred_ends = model.decode_timeed_from_hidden_states(
                hidden_states=hidden_states,
                timespan_token_positions=time_token_positions,
                video_duration_sec=video_durations,
                hard=False,
            )
        else:
            pred_starts, pred_ends = model.decode_time_from_hidden_states(
                hidden_states=hidden_states,
                time_token_positions=time_token_positions,
                use_calibration=True,
            )

    predictions: list[tuple[float, float]] = []
    count = min(pred_starts.numel(), pred_ends.numel())
    for idx in range(count):
        predictions.append((float(pred_starts[idx].item()), float(pred_ends[idx].item())))
    return predictions


def _timeed_canonical_cell_to_seconds(
    model: nn.Module,
    *,
    u_indices: torch.Tensor,
    v_indices: torch.Tensor,
    video_durations: list[float],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    from timeed.models import canonical_square_to_interval, relative_to_seconds

    decoder = getattr(model, "timeed_decoder", None)
    timeed_config = getattr(model, "timeed_config", None)
    num_u_bins = int(getattr(decoder, "num_u_bins", getattr(timeed_config, "num_u_bins", 32)))
    num_v_bins = int(getattr(decoder, "num_v_bins", getattr(timeed_config, "num_v_bins", 32)))
    duration_tau = float(getattr(decoder, "duration_tau", getattr(timeed_config, "duration_tau", 0.03)))
    u_rel = u_indices.to(device=device, dtype=torch.float32) / float(max(num_u_bins - 1, 1))
    v_rel = v_indices.to(device=device, dtype=torch.float32) / float(max(num_v_bins - 1, 1))
    start_rel, end_rel, _ = canonical_square_to_interval(u_rel, v_rel, tau=duration_tau)
    durations = torch.as_tensor(video_durations, device=device, dtype=torch.float32).reshape(-1).clamp_min(1e-6)
    if durations.numel() == 0:
        durations = torch.ones_like(start_rel)
    elif durations.numel() == 1 and start_rel.numel() > 1:
        durations = durations.repeat(start_rel.numel())
    elif durations.numel() != start_rel.numel():
        durations = durations[:1].repeat(start_rel.numel())
    return relative_to_seconds(start_rel, end_rel, durations)


def _build_timeed_span_sample_positions_from_responses(
    *,
    input_ids: torch.Tensor,
    responses: torch.Tensor,
    response_mask: Optional[torch.Tensor],
    response_length: int,
    timespan_token_id: int,
) -> torch.Tensor:
    batch_size = input_ids.size(0)
    response_offset = input_ids.size(1) - response_length
    positions = torch.full((batch_size,), -1, device=input_ids.device, dtype=torch.long)
    for sample_idx in range(batch_size):
        valid_response_len = _get_valid_response_length(
            response_mask,
            sample_idx=sample_idx,
            response_length=response_length,
        )
        if valid_response_len <= 0:
            continue
        candidate_positions = torch.nonzero(
            responses[sample_idx, :valid_response_len] == timespan_token_id,
            as_tuple=False,
        ).flatten()
        if candidate_positions.numel() > 0:
            positions[sample_idx] = int(response_offset + int(candidate_positions[0].item()))
    return positions


def _compute_timeed_span_logits_for_responses(
    *,
    model: nn.Module,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    response_length: int,
    responses: torch.Tensor,
    response_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if not _is_timeed_model(model):
        return torch.zeros(input_ids.size(0), 1, 1, device=input_ids.device, dtype=torch.float32)
    decoder = getattr(model, "timeed_decoder", None)
    timespan_token_id = getattr(model, "timespan_token_id", None)
    if decoder is None or timespan_token_id is None:
        return torch.zeros(input_ids.size(0), 1, 1, device=input_ids.device, dtype=torch.float32)

    num_u_bins = int(getattr(decoder, "num_u_bins", getattr(getattr(model, "timeed_config", None), "num_u_bins", 32)))
    num_v_bins = int(getattr(decoder, "num_v_bins", getattr(getattr(model, "timeed_config", None), "num_v_bins", 32)))
    batch_size = input_ids.shape[0]
    response_offset = input_ids.size(1) - response_length
    span_logits = torch.zeros(batch_size, num_u_bins, num_v_bins, device=input_ids.device, dtype=torch.float32)
    for sample_idx in range(batch_size):
        valid_response_len = _get_valid_response_length(
            response_mask,
            sample_idx=sample_idx,
            response_length=response_length,
        )
        if valid_response_len <= 0:
            continue
        candidate_positions = torch.nonzero(
            responses[sample_idx, :valid_response_len] == timespan_token_id,
            as_tuple=False,
        ).flatten()
        if candidate_positions.numel() == 0:
            continue
        position = int(response_offset + int(candidate_positions[0].item()))
        sample_hidden = hidden_states[sample_idx] if hidden_states.dim() == 3 else hidden_states
        decoder_out = decoder(sample_hidden[position : position + 1, :])
        if getattr(decoder_out, "span_logits", None) is not None:
            span_logits[sample_idx] = decoder_out.span_logits.float().reshape(num_u_bins, num_v_bins)
    return span_logits


def _timeed_span_valid_mask_from_responses(
    *,
    input_ids: torch.Tensor,
    responses: torch.Tensor,
    response_mask: Optional[torch.Tensor],
    response_length: int,
    timespan_token_id: Optional[int],
) -> torch.Tensor:
    if timespan_token_id is None:
        return torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.float32)
    positions = _build_timeed_span_sample_positions_from_responses(
        input_ids=input_ids,
        responses=responses,
        response_mask=response_mask,
        response_length=response_length,
        timespan_token_id=int(timespan_token_id),
    )
    return (positions >= 0).float()


def _timeed_span_logits_or_fallback(
    *,
    model: nn.Module,
    output: Any,
    input_ids: torch.Tensor,
    response_length: int,
    responses: torch.Tensor,
    response_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    model_span_logits = getattr(output, "timeed_span_sample_logits", None)
    if isinstance(model_span_logits, torch.Tensor) and model_span_logits.dim() == 3:
        return model_span_logits.float()

    hidden_states = _extract_last_hidden_state_from_output(output)
    if hidden_states is None:
        return None
    return _compute_timeed_span_logits_for_responses(
        model=model,
        hidden_states=hidden_states,
        input_ids=input_ids,
        response_length=response_length,
        responses=responses,
        response_mask=response_mask,
    )


def _csdo_positions_from_responses(
    *,
    input_ids: torch.Tensor,
    responses: torch.Tensor,
    response_mask: Optional[torch.Tensor],
    response_length: int,
    timespan_token_id: Optional[int],
) -> torch.Tensor:
    batch_size = input_ids.size(0)
    response_offset = input_ids.size(1) - response_length
    positions = torch.full((batch_size,), -1, device=input_ids.device, dtype=torch.long)
    if timespan_token_id is None:
        return positions

    for sample_idx in range(batch_size):
        valid_response_len = _get_valid_response_length(
            response_mask,
            sample_idx=sample_idx,
            response_length=response_length,
        )
        if valid_response_len <= 0:
            continue
        candidate_positions = torch.nonzero(
            responses[sample_idx, :valid_response_len] == int(timespan_token_id),
            as_tuple=False,
        ).flatten()
        if candidate_positions.numel() == 1:
            positions[sample_idx] = int(response_offset + int(candidate_positions[0].item()))
    return positions


def _empty_csdo_logits_features(
    *,
    model: nn.Module,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    codec = getattr(model, "timeple_codec", None)
    decoder = getattr(codec, "decoder", None)
    if decoder is None or not hasattr(decoder, "span_head"):
        raise ValueError("CIS Counterfactual Span Distribution Optimization (CSDO) requires model.timeple_codec.decoder.span_head.")
    num_u_bins = int(getattr(decoder, "num_u_bins"))
    num_v_bins = int(getattr(decoder, "num_v_bins"))
    span_head = getattr(decoder, "span_head")
    feature_dim = int(getattr(span_head, "in_features"))
    dtype = next(span_head.parameters()).dtype
    logits = torch.zeros(batch_size, num_u_bins, num_v_bins, device=device, dtype=dtype)
    features = torch.zeros(batch_size, feature_dim, device=device, dtype=dtype)
    valid = torch.zeros(batch_size, device=device, dtype=torch.float32)
    return logits, features, valid


def _csdo_logits_features_from_hidden_states(
    *,
    model: nn.Module,
    hidden_states: Optional[torch.Tensor],
    positions: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, features, valid = _empty_csdo_logits_features(
        model=model,
        batch_size=batch_size,
        device=device,
    )
    if hidden_states is None:
        return logits, features, valid

    codec = getattr(model, "timeple_codec", None)
    decoder = getattr(codec, "decoder", None)
    if codec is None or decoder is None:
        raise ValueError("CIS Counterfactual Span Distribution Optimization (CSDO) requires a TimePLE codec decoder.")

    valid = (positions.reshape(-1).to(device=device) >= 0).float()
    valid_indices = torch.nonzero(valid > 0, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return logits, features, valid

    if hidden_states.dim() == 3:
        timespan_hidden = hidden_states[
            valid_indices,
            positions[valid_indices].to(device=hidden_states.device, dtype=torch.long),
            :,
        ]
    elif hidden_states.dim() == 2:
        timespan_hidden = hidden_states[positions[valid_indices].to(device=hidden_states.device, dtype=torch.long), :]
    else:
        raise ValueError(f"CIS Counterfactual Span Distribution Optimization (CSDO) hidden_states must be [B, S, H] or [S, H], got {tuple(hidden_states.shape)}.")

    if getattr(model, "use_timeple_interface_adapter", False):
        timespan_embedding = model.timeple_interface_adapter.forward_output(
            timespan_hidden,
            compute_diagnostics=False,
        ).adapted
    else:
        timespan_embedding = timespan_hidden

    trunk_dtype = next(decoder.span_head.parameters()).dtype
    valid_features = decoder.trunk(timespan_embedding.to(trunk_dtype))
    valid_logits = decoder.span_head(valid_features).view(
        -1,
        int(getattr(decoder, "num_u_bins")),
        int(getattr(decoder, "num_v_bins")),
    )
    features[valid_indices] = valid_features
    logits[valid_indices] = valid_logits
    return logits, features, valid


def _csdo_logits_features_from_output(
    *,
    model: nn.Module,
    output: Any,
    input_ids: torch.Tensor,
    responses: torch.Tensor,
    response_mask: Optional[torch.Tensor],
    response_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = _csdo_positions_from_responses(
        input_ids=input_ids,
        responses=responses,
        response_mask=response_mask,
        response_length=response_length,
        timespan_token_id=getattr(model, "timespan_token_id", None),
    )
    return _csdo_logits_features_from_hidden_states(
        model=model,
        hidden_states=_extract_last_hidden_state_from_output(output),
        positions=positions,
        batch_size=input_ids.size(0),
        device=input_ids.device,
    )


def _csdo_logits_features_from_forward_output(
    *,
    actor_module: nn.Module,
    model: nn.Module,
    output: Any,
    input_ids: torch.Tensor,
    responses: torch.Tensor,
    response_mask: Optional[torch.Tensor],
    response_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = getattr(output, "csdo_logits", None)
    features = getattr(output, "csdo_features", None)
    valid = getattr(output, "csdo_valid", None)
    if isinstance(logits, torch.Tensor) and isinstance(features, torch.Tensor) and isinstance(valid, torch.Tensor):
        return logits, features, valid

    if isinstance(actor_module, FSDP):
        raise RuntimeError(
            "TimePLE Counterfactual Span Distribution Optimization (CSDO) under FSDP requires model forward to return "
            "csdo_logits/features/valid. Sync patches/timeple before training."
        )

    return _csdo_logits_features_from_output(
        model=model,
        output=output,
        input_ids=input_ids,
        responses=responses,
        response_mask=response_mask,
        response_length=response_length,
    )


def _tr_spd_positions_from_responses(
    *,
    input_ids: torch.Tensor,
    responses: torch.Tensor,
    response_mask: Optional[torch.Tensor],
    response_length: int,
    timespan_token_id: Optional[int],
) -> torch.Tensor:
    return _csdo_positions_from_responses(
        input_ids=input_ids,
        responses=responses,
        response_mask=response_mask,
        response_length=response_length,
        timespan_token_id=timespan_token_id,
    )


def _tr_spd_logits_features_from_hidden_states(
    *,
    model: nn.Module,
    hidden_states: Optional[torch.Tensor],
    positions: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if hidden_states is None:
        codec = getattr(model, "timeple_codec", None)
        decoder = getattr(codec, "decoder", None)
        if decoder is None or not hasattr(decoder, "span_head"):
            raise ValueError("CIS Trust-Region Span Posterior Distillation (TR-SPD) requires model.timeple_codec.decoder.span_head.")
        num_u_bins = int(getattr(decoder, "num_u_bins"))
        num_v_bins = int(getattr(decoder, "num_v_bins"))
        span_head = getattr(decoder, "span_head")
        feature_dim = int(getattr(span_head, "in_features"))
        dtype = next(span_head.parameters()).dtype
        return (
            torch.zeros(batch_size, num_u_bins, num_v_bins, device=device, dtype=dtype),
            torch.zeros(batch_size, feature_dim, device=device, dtype=dtype),
            torch.zeros(batch_size, device=device, dtype=torch.float32),
        )

    codec = getattr(model, "timeple_codec", None)
    decoder = getattr(codec, "decoder", None)
    if codec is None or decoder is None:
        raise ValueError("CIS Trust-Region Span Posterior Distillation (TR-SPD) requires a TimePLE codec decoder.")

    logits, features, valid = _csdo_logits_features_from_hidden_states(
        model=model,
        hidden_states=hidden_states,
        positions=positions,
        batch_size=batch_size,
        device=device,
    )
    return logits, features, valid


def _tr_spd_logits_features_from_output(
    *,
    model: nn.Module,
    output: Any,
    input_ids: torch.Tensor,
    responses: torch.Tensor,
    response_mask: Optional[torch.Tensor],
    response_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = _tr_spd_positions_from_responses(
        input_ids=input_ids,
        responses=responses,
        response_mask=response_mask,
        response_length=response_length,
        timespan_token_id=getattr(model, "timespan_token_id", None),
    )
    return _tr_spd_logits_features_from_hidden_states(
        model=model,
        hidden_states=_extract_last_hidden_state_from_output(output),
        positions=positions,
        batch_size=input_ids.size(0),
        device=input_ids.device,
    )


def _tr_spd_logits_features_from_forward_output(
    *,
    actor_module: nn.Module,
    model: nn.Module,
    output: Any,
    input_ids: torch.Tensor,
    responses: torch.Tensor,
    response_mask: Optional[torch.Tensor],
    response_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = getattr(output, "tr_spd_logits", None)
    features = getattr(output, "tr_spd_features", None)
    valid = getattr(output, "tr_spd_valid", None)
    if isinstance(logits, torch.Tensor) and isinstance(features, torch.Tensor) and isinstance(valid, torch.Tensor):
        return logits, features, valid

    if isinstance(actor_module, FSDP):
        raise RuntimeError(
            "TimePLE Trust-Region Span Posterior Distillation (TR-SPD) under FSDP requires model forward to return "
            "tr_spd_logits/features/valid. Sync patches/timeple before training."
        )

    return _tr_spd_logits_features_from_output(
        model=model,
        output=output,
        input_ids=input_ids,
        responses=responses,
        response_mask=response_mask,
        response_length=response_length,
    )


def _span_flat_log_probs(span_logits: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(span_logits.float().flatten(1), dim=-1)


def _gather_span_cell_log_probs(
    span_logits: torch.Tensor,
    u_idx: torch.Tensor,
    v_idx: torch.Tensor,
) -> torch.Tensor:
    flat_log_probs = _span_flat_log_probs(span_logits)
    num_v_bins = int(span_logits.shape[-1])
    u_idx = u_idx.long().reshape(-1).to(device=span_logits.device)
    v_idx = v_idx.long().reshape(-1).to(device=span_logits.device)
    valid = (u_idx >= 0) & (v_idx >= 0) & (u_idx < int(span_logits.shape[-2])) & (v_idx < num_v_bins)
    flat_idx = u_idx.clamp_min(0).clamp_max(max(int(span_logits.shape[-2]) - 1, 0)) * num_v_bins
    flat_idx = flat_idx + v_idx.clamp_min(0).clamp_max(max(num_v_bins - 1, 0))
    gathered = torch.zeros(span_logits.size(0), device=span_logits.device, dtype=torch.float32)
    if bool(valid.any().item()):
        gathered[valid] = flat_log_probs[valid, flat_idx[valid]]
    return gathered


def _compute_exact_span_kl_loss(
    current_logits: Optional[torch.Tensor],
    ref_logits: Optional[torch.Tensor],
    sample_mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    if current_logits is None or ref_logits is None or current_logits.shape != ref_logits.shape:
        if isinstance(current_logits, torch.Tensor):
            device = current_logits.device
        elif isinstance(ref_logits, torch.Tensor):
            device = ref_logits.device
        else:
            device = torch.device("cpu")
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return zero, {"timeed_span_exact_kl": 0.0, "timeed_span_kl_valid_fraction": 0.0}

    logp = _span_flat_log_probs(current_logits)
    logp_ref = _span_flat_log_probs(ref_logits.to(device=current_logits.device))
    p = logp.exp()
    per_sample_kl = (p * (logp - logp_ref)).sum(dim=-1)
    if sample_mask is None:
        mask = torch.ones_like(per_sample_kl, dtype=torch.float32)
    else:
        mask = (sample_mask.float().reshape(-1).to(device=current_logits.device) > 0).float()
    valid_count = mask.sum().clamp_min(1.0)
    kl_loss = (per_sample_kl * mask).sum() / valid_count
    return kl_loss, {
        "timeed_span_exact_kl": float(kl_loss.detach().item()),
        "timeed_span_kl_valid_fraction": float(mask.mean().detach().item()),
    }


def _csdo_zero_metrics() -> dict[str, float]:
    return {
        "csdo_loss": 0.0,
        "csdo_span_kl": 0.0,
        "csdo_reward_old": 0.0,
        "csdo_reward_q": 0.0,
        "csdo_reward_cur": 0.0,
        "csdo_adv_mean": 0.0,
        "csdo_adv_std": 0.0,
        "csdo_q_entropy": 0.0,
        "csdo_old_entropy": 0.0,
        "csdo_cur_entropy": 0.0,
        "csdo_q_kl_old": 0.0,
        "csdo_cur_kl_old": 0.0,
        "csdo_valid_frac": 0.0,
        "csdo_adv_valid_frac": 0.0,
        "csdo_expect_shift_norm": 0.0,
        "csdo_ref_kl_enabled": 0.0,
    }


def _csdo_cell_coordinates(
    decoder: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    u_centers = getattr(decoder, "u_centers", None)
    v_centers = getattr(decoder, "v_centers", None)
    if not isinstance(u_centers, torch.Tensor) or not isinstance(v_centers, torch.Tensor):
        raise ValueError("CIS Counterfactual Span Distribution Optimization (CSDO) requires decoder.u_centers and decoder.v_centers buffers.")
    num_u_bins = int(getattr(decoder, "num_u_bins", u_centers.numel()))
    num_v_bins = int(getattr(decoder, "num_v_bins", v_centers.numel()))
    flat_u = u_centers.to(device=device, dtype=dtype).reshape(-1).repeat_interleave(num_v_bins)
    flat_v = v_centers.to(device=device, dtype=dtype).reshape(-1).repeat(num_u_bins)
    return torch.stack([flat_u, flat_v], dim=-1)


def _resolve_csdo_duration_tensor(
    *,
    micro_batch: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts_per_sample, ends_per_sample = _extract_timespan_labels_per_sample(micro_batch)
    durations_per_sample = _extract_timespan_video_durations_per_sample(micro_batch)
    durations: list[float] = []
    target_valid: list[float] = []
    for sample_idx in range(batch_size):
        target_starts = starts_per_sample[sample_idx] if sample_idx < len(starts_per_sample) else []
        target_ends = ends_per_sample[sample_idx] if sample_idx < len(ends_per_sample) else []
        target_count = min(len(target_starts), len(target_ends))
        if target_count <= 0:
            durations.append(1.0)
            target_valid.append(0.0)
            continue
        target_durations = durations_per_sample[sample_idx] if sample_idx < len(durations_per_sample) else []
        resolved = _resolve_decode_video_durations(
            target_durations=target_durations,
            target_ends=target_ends[:target_count],
            prediction_count=1,
        )
        durations.append(float(resolved[0]) if len(resolved) > 0 else max(max(target_ends[:target_count]), 1.0))
        target_valid.append(1.0)
    return (
        torch.as_tensor(durations, device=device, dtype=torch.float32).clamp_min(1e-6),
        torch.as_tensor(target_valid, device=device, dtype=torch.float32),
    )


def _csdo_reward_from_seconds(
    *,
    pred_start_sec: torch.Tensor,
    pred_end_sec: torch.Tensor,
    micro_batch: dict[str, Any],
    durations: torch.Tensor,
    reward_type: str,
    boundary_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts_per_sample, ends_per_sample = _extract_timespan_labels_per_sample(micro_batch)
    device = pred_start_sec.device
    reward = torch.zeros_like(pred_start_sec, dtype=torch.float32)
    valid = torch.zeros(pred_start_sec.shape[0], device=device, dtype=torch.float32)
    reward_type = str(reward_type)
    if reward_type not in {"iou", "iou_boundary"}:
        raise ValueError(f"Unsupported csdo_reward_type: {reward_type!r}.")

    for sample_idx in range(pred_start_sec.shape[0]):
        target_starts = starts_per_sample[sample_idx] if sample_idx < len(starts_per_sample) else []
        target_ends = ends_per_sample[sample_idx] if sample_idx < len(ends_per_sample) else []
        target_count = min(len(target_starts), len(target_ends))
        if target_count <= 0:
            continue

        pred_start_flat = pred_start_sec[sample_idx].float().reshape(1, -1)
        pred_end_flat = pred_end_sec[sample_idx].float().reshape(1, -1)
        target_start = torch.as_tensor(
            target_starts[:target_count],
            device=device,
            dtype=torch.float32,
        ).reshape(-1, 1)
        target_end = torch.as_tensor(
            target_ends[:target_count],
            device=device,
            dtype=torch.float32,
        ).reshape(-1, 1)
        candidate_reward = _tensor_interval_iou(pred_start_flat, pred_end_flat, target_start, target_end)
        if reward_type == "iou_boundary" and float(boundary_weight) > 0.0:
            duration = durations[sample_idx].float().clamp_min(1e-6)
            boundary = (
                (pred_start_flat - target_start).abs()
                + (pred_end_flat - target_end).abs()
            ) / duration
            candidate_reward = candidate_reward - float(boundary_weight) * boundary
        best_reward = candidate_reward.amax(dim=0).reshape_as(pred_start_sec[sample_idx])
        reward[sample_idx] = best_reward.to(reward)
        valid[sample_idx] = 1.0

    return reward, valid


def _compute_csdo_reward(
    *,
    codec: nn.Module,
    features: torch.Tensor,
    uv: torch.Tensor,
    micro_batch: dict[str, Any],
    durations: torch.Tensor,
    reward_type: str,
    boundary_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(codec, "decode_uv_with_features_seconds"):
        raise ValueError("CIS Counterfactual Span Distribution Optimization (CSDO) requires codec.decode_uv_with_features_seconds for decoder parity.")
    pred_start_sec, pred_end_sec, _ = codec.decode_uv_with_features_seconds(
        features=features,
        uv=uv,
        video_duration_sec=durations,
        hard=True,
    )
    return _csdo_reward_from_seconds(
        pred_start_sec=pred_start_sec,
        pred_end_sec=pred_end_sec,
        micro_batch=micro_batch,
        durations=durations,
        reward_type=reward_type,
        boundary_weight=boundary_weight,
    )


def _masked_mean_float(values: torch.Tensor, mask: torch.Tensor) -> float:
    valid_count = mask.sum().clamp_min(1.0)
    return float(((values.float().reshape(-1) * mask.float().reshape(-1)).sum() / valid_count).detach().item())


def _compute_csdo_loss(
    *,
    current_logits: torch.Tensor,
    current_features: torch.Tensor,
    old_logits: torch.Tensor,
    old_features: torch.Tensor,
    ref_logits: Optional[torch.Tensor],
    micro_batch: dict[str, Any],
    sample_mask: torch.Tensor,
    codec: nn.Module,
    eta: float,
    tau: float,
    adv_norm: bool,
    adv_clip: float,
    min_adv_std: float,
    reward_type: str,
    boundary_weight: float,
    use_ref_kl: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if current_logits is None or old_logits is None or current_features is None or old_features is None:
        raise ValueError("CIS Counterfactual Span Distribution Optimization (CSDO) requires current/old logits and current/old decoder features.")
    if use_ref_kl and ref_logits is None:
        raise ValueError("csdo_ref_logits are required when csdo_use_ref_kl=True.")
    if current_logits.shape != old_logits.shape:
        raise ValueError(f"CIS Counterfactual Span Distribution Optimization (CSDO) current/old logits shape mismatch: {current_logits.shape} vs {old_logits.shape}.")
    if current_logits.dim() != 3:
        raise ValueError(f"CIS Counterfactual Span Distribution Optimization (CSDO) logits must have shape [B, U, V], got {tuple(current_logits.shape)}.")
    if current_features.shape != old_features.shape or current_features.size(0) != current_logits.size(0):
        raise ValueError(
            "CIS Counterfactual Span Distribution Optimization (CSDO) feature shape mismatch: "
            f"current={tuple(current_features.shape)}, old={tuple(old_features.shape)}, logits={tuple(current_logits.shape)}."
        )
    if use_ref_kl and ref_logits.shape != current_logits.shape:
        raise ValueError(f"CIS Counterfactual Span Distribution Optimization (CSDO) ref logits shape mismatch: {ref_logits.shape} vs {current_logits.shape}.")

    device = current_logits.device
    zero_loss = current_logits.sum() * 0.0
    zero_kl = current_logits.sum() * 0.0
    metrics = _csdo_zero_metrics()
    metrics["csdo_ref_kl_enabled"] = 1.0 if use_ref_kl else 0.0

    batch_size = current_logits.size(0)
    sample_mask = (sample_mask.float().reshape(-1).to(device=device) > 0).float()
    if sample_mask.numel() != batch_size:
        raise ValueError(f"CIS Counterfactual Span Distribution Optimization (CSDO) sample_mask size mismatch: {sample_mask.numel()} vs {batch_size}.")
    durations, target_valid = _resolve_csdo_duration_tensor(
        micro_batch=micro_batch,
        batch_size=batch_size,
        device=device,
    )
    mask = sample_mask * target_valid
    metrics["csdo_valid_frac"] = float(mask.mean().detach().item()) if batch_size > 0 else 0.0
    if mask.sum().item() <= 0.0:
        return zero_loss, zero_kl, metrics

    old_logits = old_logits.to(device=device).float().detach()
    old_features = old_features.to(device=device).detach()
    current_features_for_metrics = current_features.detach()
    ref_logits_for_kl = ref_logits.to(device=device).float() if ref_logits is not None else None

    current_flat_logp = F.log_softmax(current_logits.float().flatten(1), dim=-1)
    current_p = current_flat_logp.exp()
    old_logp = F.log_softmax(old_logits.flatten(1), dim=-1).detach()
    old_p = old_logp.exp()
    coords = _csdo_cell_coordinates(
        codec.decoder,
        device=device,
        dtype=current_flat_logp.dtype,
    )

    with torch.no_grad():
        mu_old = old_p @ coords
        reward_old, reward_valid = _compute_csdo_reward(
            codec=codec,
            features=old_features,
            uv=mu_old,
            micro_batch=micro_batch,
            durations=durations,
            reward_type=reward_type,
            boundary_weight=boundary_weight,
        )
        mask = mask * reward_valid
        if mask.sum().item() <= 0.0:
            metrics["csdo_valid_frac"] = float(mask.mean().detach().item()) if batch_size > 0 else 0.0
            return zero_loss, zero_kl, metrics

        mu_plus = (1.0 - float(eta)) * mu_old.unsqueeze(1) + float(eta) * coords.unsqueeze(0)
        reward_plus, _ = _compute_csdo_reward(
            codec=codec,
            features=old_features,
            uv=mu_plus,
            micro_batch=micro_batch,
            durations=durations,
            reward_type=reward_type,
            boundary_weight=boundary_weight,
        )
        advantages = reward_plus - reward_old.unsqueeze(-1)
        raw_adv_mean = advantages.mean(dim=-1)
        raw_adv_std = advantages.std(dim=-1, unbiased=False)
        adv_signal_mask = (raw_adv_std > float(min_adv_std)).float()
        if adv_norm:
            centered = advantages - advantages.mean(dim=-1, keepdim=True)
            normalized = centered / raw_adv_std.clamp_min(float(min_adv_std)).unsqueeze(-1)
            advantages_for_q = torch.where(
                adv_signal_mask.reshape(-1, 1) > 0,
                normalized,
                torch.zeros_like(normalized),
            )
        else:
            advantages_for_q = advantages
        if float(adv_clip) > 0.0:
            advantages_for_q = advantages_for_q.clamp(min=-float(adv_clip), max=float(adv_clip))

        q = F.softmax(old_logp + advantages_for_q / max(float(tau), 1e-6), dim=-1).detach()

    espo_loss_per_sample = -(q * current_flat_logp).sum(dim=-1)
    valid_count = mask.sum().clamp_min(1.0)
    espo_loss = (espo_loss_per_sample * mask).sum() / valid_count

    if use_ref_kl:
        ref_logp = F.log_softmax(ref_logits_for_kl.flatten(1), dim=-1)
        per_sample_kl = (current_p * (current_flat_logp - ref_logp)).sum(dim=-1)
        span_kl_loss = (per_sample_kl * mask).sum() / valid_count
    else:
        span_kl_loss = zero_kl

    with torch.no_grad():
        mu_q = q @ coords
        reward_q, _ = _compute_csdo_reward(
            codec=codec,
            features=old_features,
            uv=mu_q,
            micro_batch=micro_batch,
            durations=durations,
            reward_type=reward_type,
            boundary_weight=boundary_weight,
        )
        mu_cur = current_p.detach() @ coords
        reward_cur, _ = _compute_csdo_reward(
            codec=codec,
            features=current_features_for_metrics,
            uv=mu_cur,
            micro_batch=micro_batch,
            durations=durations,
            reward_type=reward_type,
            boundary_weight=boundary_weight,
        )

        old_entropy = -(old_p * old_logp).sum(dim=-1)
        q_log = q.clamp_min(1e-12).log()
        q_entropy = -(q * q_log).sum(dim=-1)
        cur_entropy = -(current_p.detach() * current_flat_logp.detach()).sum(dim=-1)
        q_kl_old = (q * (q_log - old_logp)).sum(dim=-1)
        cur_kl_old = (current_p.detach() * (current_flat_logp.detach() - old_logp)).sum(dim=-1)
        expect_shift_norm = (mu_q - mu_old).norm(dim=-1)
        used_adv_mean = advantages_for_q.mean(dim=-1)
        used_adv_std = advantages_for_q.std(dim=-1, unbiased=False)
        adv_valid_mask = mask * adv_signal_mask

        metrics.update(
            {
                "csdo_loss": float(espo_loss.detach().item()),
                "csdo_span_kl": float(span_kl_loss.detach().item()),
                "csdo_reward_old": _masked_mean_float(reward_old, mask),
                "csdo_reward_q": _masked_mean_float(reward_q, mask),
                "csdo_reward_cur": _masked_mean_float(reward_cur, mask),
                "csdo_adv_mean": _masked_mean_float(used_adv_mean, mask),
                "csdo_adv_std": _masked_mean_float(used_adv_std, mask),
                "csdo_q_entropy": _masked_mean_float(q_entropy, mask),
                "csdo_old_entropy": _masked_mean_float(old_entropy, mask),
                "csdo_cur_entropy": _masked_mean_float(cur_entropy, mask),
                "csdo_q_kl_old": _masked_mean_float(q_kl_old, mask),
                "csdo_cur_kl_old": _masked_mean_float(cur_kl_old, mask),
                "csdo_valid_frac": float(mask.mean().detach().item()) if batch_size > 0 else 0.0,
                "csdo_adv_valid_frac": float(adv_valid_mask.mean().detach().item()) if batch_size > 0 else 0.0,
                "csdo_expect_shift_norm": _masked_mean_float(expect_shift_norm, mask),
                "csdo_ref_kl_enabled": 1.0 if use_ref_kl else 0.0,
            }
        )
    return espo_loss, span_kl_loss, metrics


def _compute_timeed_cell_reward_maps(
    *,
    model: nn.Module,
    span_logits: Optional[torch.Tensor],
    micro_batch: dict[str, Any],
    sample_mask: Optional[torch.Tensor],
    iou_weight: float,
    boundary_weight: float,
    boundary_tau: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if span_logits is None or span_logits.dim() != 3:
        device = span_logits.device if isinstance(span_logits, torch.Tensor) else torch.device("cpu")
        empty = torch.zeros(0, 1, 1, device=device, dtype=torch.float32)
        return empty, torch.zeros(0, device=device, dtype=torch.float32), {
            "timeed_span_cell_valid_fraction": 0.0,
            "timeed_span_cell_reward_mean": 0.0,
            "timeed_span_cell_reward_max": 0.0,
        }

    device = span_logits.device
    batch_size, num_u_bins, num_v_bins = span_logits.shape
    reward_maps = torch.zeros(batch_size, num_u_bins, num_v_bins, device=device, dtype=torch.float32)
    valid_mask = torch.zeros(batch_size, device=device, dtype=torch.float32)
    if sample_mask is None:
        response_mask = torch.ones(batch_size, device=device, dtype=torch.float32)
    else:
        response_mask = (sample_mask.float().reshape(-1).to(device=device) > 0).float()

    starts_per_sample, ends_per_sample = _extract_timespan_labels_per_sample(micro_batch)
    durations_per_sample = _extract_timespan_video_durations_per_sample(micro_batch)
    flat_u = torch.arange(num_u_bins, device=device, dtype=torch.long).repeat_interleave(num_v_bins)
    flat_v = torch.arange(num_v_bins, device=device, dtype=torch.long).repeat(num_u_bins)
    flat_count = int(flat_u.numel())
    iou_weight = max(float(iou_weight), 0.0)
    boundary_weight = max(float(boundary_weight), 0.0)
    boundary_tau = max(float(boundary_tau), 1e-6)

    for sample_idx in range(batch_size):
        if response_mask[sample_idx].item() <= 0.0:
            continue
        target_starts = starts_per_sample[sample_idx] if sample_idx < len(starts_per_sample) else []
        target_ends = ends_per_sample[sample_idx] if sample_idx < len(ends_per_sample) else []
        target_durations = durations_per_sample[sample_idx] if sample_idx < len(durations_per_sample) else []
        target_count = min(len(target_starts), len(target_ends))
        if target_count <= 0:
            continue

        duration = _resolve_decode_video_durations(
            target_durations=target_durations,
            target_ends=target_ends,
            prediction_count=1,
        )
        if len(duration) == 0:
            duration = [max(max(float(value) for value in target_ends[:target_count]), 1.0)]
        pred_start, pred_end = _timeed_canonical_cell_to_seconds(
            model,
            u_indices=flat_u,
            v_indices=flat_v,
            video_durations=[float(duration[0])] * flat_count,
            device=device,
        )

        target_start = torch.as_tensor(target_starts[:target_count], device=device, dtype=torch.float32).reshape(-1, 1)
        target_end = torch.as_tensor(target_ends[:target_count], device=device, dtype=torch.float32).reshape(-1, 1)
        pred_start_2d = pred_start.reshape(1, -1)
        pred_end_2d = pred_end.reshape(1, -1)
        iou = _tensor_interval_iou(pred_start_2d, pred_end_2d, target_start, target_end)
        boundary = torch.exp(
            -(
                torch.abs(pred_start_2d - target_start)
                + torch.abs(pred_end_2d - target_end)
            )
            / boundary_tau
        )
        reward_flat = (iou_weight * iou + boundary_weight * boundary).amax(dim=0).clamp(0.0, 1.0)
        reward_maps[sample_idx] = reward_flat.reshape(num_u_bins, num_v_bins)
        valid_mask[sample_idx] = 1.0

    if valid_mask.sum().item() <= 0.0:
        return reward_maps, valid_mask, {
            "timeed_span_cell_valid_fraction": float(valid_mask.mean().detach().item()) if batch_size > 0 else 0.0,
            "timeed_span_cell_reward_mean": 0.0,
            "timeed_span_cell_reward_max": 0.0,
        }

    valid_rewards = reward_maps[valid_mask > 0]
    return reward_maps, valid_mask, {
        "timeed_span_cell_valid_fraction": float(valid_mask.mean().detach().item()),
        "timeed_span_cell_reward_mean": float(valid_rewards.mean().detach().item()),
        "timeed_span_cell_reward_max": float(valid_rewards.max().detach().item()),
    }


def _compute_exact_span_grpo_loss(
    *,
    old_logits: Optional[torch.Tensor],
    current_logits: Optional[torch.Tensor],
    reward_maps: torch.Tensor,
    sample_mask: Optional[torch.Tensor],
    clip_ratio_low: float,
    clip_ratio_high: float,
    advantage_eps: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if current_logits is None or old_logits is None or current_logits.shape != old_logits.shape:
        if isinstance(current_logits, torch.Tensor):
            device = current_logits.device
        elif isinstance(old_logits, torch.Tensor):
            device = old_logits.device
        else:
            device = torch.device("cpu")
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return zero, {
            "span_valid_fraction": 0.0,
            "span_expected_reward_old": 0.0,
            "span_expected_reward_new": 0.0,
        }

    old_logits = old_logits.to(device=current_logits.device).float()
    reward_maps = reward_maps.to(device=current_logits.device).float()
    if reward_maps.shape != current_logits.shape:
        zero = torch.zeros((), device=current_logits.device, dtype=torch.float32)
        return zero, {
            "span_valid_fraction": 0.0,
            "span_expected_reward_old": 0.0,
            "span_expected_reward_new": 0.0,
        }

    old_logp = _span_flat_log_probs(old_logits).detach()
    new_logp = _span_flat_log_probs(current_logits)
    p_old = old_logp.exp()
    p_new = new_logp.exp()
    rewards = reward_maps.flatten(1).detach()
    baseline = (p_old * rewards).sum(dim=-1, keepdim=True)
    variance = (p_old * (rewards - baseline).pow(2)).sum(dim=-1, keepdim=True)
    std = variance.clamp_min(0.0).sqrt().clamp_min(float(advantage_eps))
    advantages = (rewards - baseline) / std

    log_ratio = torch.clamp(new_logp - old_logp, -20.0, 20.0)
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(min=1.0 - float(clip_ratio_low), max=1.0 + float(clip_ratio_high))
    objective = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    per_sample_loss = -(p_old * objective).sum(dim=-1)

    if sample_mask is None:
        mask = torch.ones_like(per_sample_loss, dtype=torch.float32)
    else:
        mask = (sample_mask.float().reshape(-1).to(device=current_logits.device) > 0).float()
    valid_count = mask.sum().clamp_min(1.0)
    exact_span_grpo_loss = (per_sample_loss * mask).sum() / valid_count

    expected_old = baseline.reshape(-1)
    expected_new = (p_new * rewards).sum(dim=-1)
    approx_kl = (p_old * (old_logp - new_logp)).sum(dim=-1)
    entropy_new = -(p_new * new_logp).sum(dim=-1)
    weighted_clipfrac = (
        p_old
        * ((ratio > (1.0 + float(clip_ratio_high))) | (ratio < (1.0 - float(clip_ratio_low)))).float()
    ).sum(dim=-1)
    return exact_span_grpo_loss, {
        "span_valid_fraction": float(mask.mean().detach().item()),
        "span_expected_reward_old": float(((expected_old * mask).sum() / valid_count).detach().item()),
        "span_expected_reward_new": float(((expected_new * mask).sum() / valid_count).detach().item()),
        "span_reward_std_old": float(((std.reshape(-1) * mask).sum() / valid_count).detach().item()),
        "span_reward_advantage_mean": float(
            (((p_old * advantages).sum(dim=-1) * mask).sum() / valid_count).detach().item()
        ),
        "span_policy_approx_kl": float(((approx_kl * mask).sum() / valid_count).detach().item()),
        "span_entropy": float(((entropy_new * mask).sum() / valid_count).detach().item()),
        "span_pg_clipfrac": float(((weighted_clipfrac * mask).sum() / valid_count).detach().item()),
    }


def _compute_reward_map_span_preference_loss(
    *,
    current_logits: Optional[torch.Tensor],
    ref_logits: Optional[torch.Tensor],
    old_logits: Optional[torch.Tensor],
    reward_maps: torch.Tensor,
    sample_mask: Optional[torch.Tensor],
    beta: float,
    reward_gap_delta: float,
    negative_reward_threshold: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if (
        current_logits is None
        or ref_logits is None
        or old_logits is None
        or current_logits.shape != ref_logits.shape
        or current_logits.shape != old_logits.shape
        or reward_maps.shape != current_logits.shape
    ):
        device = current_logits.device if isinstance(current_logits, torch.Tensor) else torch.device("cpu")
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return zero, {"timeed_span_pref_valid_fraction": 0.0, "timeed_span_pref_loss": 0.0}

    device = current_logits.device
    rewards = reward_maps.to(device=device).float().flatten(1).detach()
    old_logp = _span_flat_log_probs(old_logits.to(device=device).float()).detach()
    p_old = old_logp.exp()
    new_logp = _span_flat_log_probs(current_logits)
    ref_logp = _span_flat_log_probs(ref_logits.to(device=device).float())

    chosen_idx = torch.argmax(rewards, dim=-1)
    negative_mask = rewards <= float(negative_reward_threshold)
    negative_scores = p_old.masked_fill(~negative_mask, -1.0)
    has_negative = negative_mask.any(dim=-1)
    hard_rejected_idx = torch.argmax(negative_scores, dim=-1)
    fallback_rejected_idx = torch.argmin(rewards, dim=-1)
    rejected_idx = torch.where(has_negative, hard_rejected_idx, fallback_rejected_idx)

    row_idx = torch.arange(rewards.size(0), device=device, dtype=torch.long)
    chosen_reward = rewards[row_idx, chosen_idx]
    rejected_reward = rewards[row_idx, rejected_idx]
    reward_gap = chosen_reward - rejected_reward
    if sample_mask is None:
        mask = torch.ones_like(reward_gap, dtype=torch.float32)
    else:
        mask = (sample_mask.float().reshape(-1).to(device=device) > 0).float()
    mask = mask * (reward_gap > float(reward_gap_delta)).float()
    valid_count = mask.sum().clamp_min(1.0)

    chosen_new = new_logp[row_idx, chosen_idx]
    rejected_new = new_logp[row_idx, rejected_idx]
    chosen_ref = ref_logp[row_idx, chosen_idx]
    rejected_ref = ref_logp[row_idx, rejected_idx]
    current_margin = chosen_new - rejected_new
    ref_margin = chosen_ref - rejected_ref
    logits = float(beta) * (current_margin - ref_margin)
    pref_loss_vec = -F.logsigmoid(logits)
    pref_loss = (pref_loss_vec * mask).sum() / valid_count
    return pref_loss, {
        "timeed_span_pref_valid_fraction": float(mask.mean().detach().item()),
        "timeed_span_pref_loss": float(pref_loss.detach().item()),
        "timeed_span_pref_margin": float(
            ((current_margin - ref_margin) * mask).sum().detach().item() / float(valid_count.item())
        ),
        "timeed_span_pref_reward_gap": float(((reward_gap * mask).sum() / valid_count).detach().item()),
        "timeed_span_pref_chosen_reward": float(((chosen_reward * mask).sum() / valid_count).detach().item()),
        "timeed_span_pref_rejected_reward": float(((rejected_reward * mask).sum() / valid_count).detach().item()),
    }


def _compute_span_preference_loss(
    *,
    current_logits: Optional[torch.Tensor],
    ref_logits: Optional[torch.Tensor],
    chosen_u_idx: torch.Tensor,
    chosen_v_idx: torch.Tensor,
    rejected_u_idx: torch.Tensor,
    rejected_v_idx: torch.Tensor,
    pref_mask: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if current_logits is None or ref_logits is None or current_logits.shape != ref_logits.shape:
        if isinstance(current_logits, torch.Tensor):
            device = current_logits.device
        elif isinstance(ref_logits, torch.Tensor):
            device = ref_logits.device
        else:
            device = torch.device("cpu")
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return zero, {"timeed_span_pref_valid_fraction": 0.0, "timeed_span_pref_loss": 0.0}

    chosen_logp = _gather_span_cell_log_probs(current_logits, chosen_u_idx, chosen_v_idx)
    rejected_logp = _gather_span_cell_log_probs(current_logits, rejected_u_idx, rejected_v_idx)
    ref_logits = ref_logits.to(device=current_logits.device)
    chosen_ref_logp = _gather_span_cell_log_probs(ref_logits, chosen_u_idx, chosen_v_idx)
    rejected_ref_logp = _gather_span_cell_log_probs(ref_logits, rejected_u_idx, rejected_v_idx)
    mask = (pref_mask.float().reshape(-1).to(device=current_logits.device) > 0).float()
    valid_count = mask.sum().clamp_min(1.0)
    current_margin = chosen_logp - rejected_logp
    ref_margin = chosen_ref_logp - rejected_ref_logp
    logits = float(beta) * (current_margin - ref_margin)
    loss_vec = -F.logsigmoid(logits)
    pref_loss = (loss_vec * mask).sum() / valid_count
    return pref_loss, {
        "timeed_span_pref_valid_fraction": float(mask.mean().detach().item()),
        "timeed_span_pref_loss": float(pref_loss.detach().item()),
        "timeed_span_pref_margin": float(
            ((current_margin - ref_margin) * mask).sum().detach().item() / float(valid_count.item())
        ),
    }


def _extract_last_hidden_state_from_output(output: Any) -> Optional[torch.Tensor]:
    last_hidden_state = getattr(output, "last_hidden_state", None)
    if isinstance(last_hidden_state, torch.Tensor) and last_hidden_state.shape[-1] > 0:
        return last_hidden_state

    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is None:
        return None
    if isinstance(hidden_states, (tuple, list)):
        if len(hidden_states) == 0:
            return None
        candidate = hidden_states[-1]
        if isinstance(candidate, torch.Tensor) and candidate.shape[-1] > 0:
            return candidate
        return None
    if isinstance(hidden_states, torch.Tensor):
        if hidden_states.shape[-1] <= 0:
            return None
        return hidden_states
    return None


def _resolve_timecodec_decode_hidden_states(
    *,
    output: Any,
    actor_module: nn.Module,
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    multi_modal_inputs: dict[str, Any],
    time_codec_kwargs: dict[str, Any],
) -> tuple[Optional[torch.Tensor], float]:
    last_hidden_state = _extract_last_hidden_state_from_output(output)
    if last_hidden_state is not None:
        return last_hidden_state, 1.0

    return None, 0.0


def _select_best_decoded_prediction(
    *,
    decoded_predictions: list[tuple[float, float]],
    target_starts: list[float],
    target_ends: list[float],
    prefix: str = "timecodec",
) -> Optional[dict[str, float]]:
    target_count = min(len(target_starts), len(target_ends))
    if len(decoded_predictions) == 0 or target_count == 0:
        return None

    best_prediction: Optional[dict[str, float]] = None
    iou_key = f"{prefix}_decoded_iou"
    pred_start_key = f"{prefix}_pred_start"
    pred_end_key = f"{prefix}_pred_end"
    for pred_start, pred_end in decoded_predictions:
        for target_idx in range(target_count):
            decoded_iou = _segment_iou(
                pred_start,
                pred_end,
                float(target_starts[target_idx]),
                float(target_ends[target_idx]),
            )
            if best_prediction is None or decoded_iou > best_prediction[iou_key]:
                best_prediction = {
                    iou_key: float(decoded_iou),
                    pred_start_key: float(pred_start),
                    pred_end_key: float(pred_end),
                }
    return best_prediction


def _tensor_interval_iou(
    pred_start: torch.Tensor,
    pred_end: torch.Tensor,
    target_start: torch.Tensor,
    target_end: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred_left = torch.minimum(pred_start, pred_end)
    pred_right = torch.maximum(pred_start, pred_end)
    target_left = torch.minimum(target_start, target_end)
    target_right = torch.maximum(target_start, target_end)

    intersection = (torch.minimum(pred_right, target_right) - torch.maximum(pred_left, target_left)).clamp(min=0.0)
    union = (pred_right - pred_left) + (target_right - target_left) - intersection
    return intersection / union.clamp_min(eps)


def _assign_aux_scalar(
    aux_features: dict[str, torch.Tensor],
    *,
    key: str,
    sample_idx: int,
    value: Any,
) -> bool:
    scalar = _maybe_to_float_scalar(value)
    if scalar is None or key not in aux_features:
        return False
    aux_features[key][sample_idx] = scalar
    return True


def _populate_cis_sample_features_from_hidden_states(
    aux_features: dict[str, torch.Tensor],
    aux_non_tensors: dict[str, list[Any]],
    *,
    model: nn.Module,
    sample_hidden_states: torch.Tensor,
    sample_info: dict[str, Any],
    response_offset: int,
    decode_source: float,
) -> None:
    if sample_hidden_states.dim() != 2:
        return

    candidate_positions = sample_info.get("candidate_positions")
    if not isinstance(candidate_positions, torch.Tensor):
        return

    target_starts = sample_info.get("target_starts", [])
    target_ends = sample_info.get("target_ends", [])
    target_durations = sample_info.get("target_durations", [])
    matched_count = min(int(candidate_positions.numel()), len(target_starts), len(target_ends))
    if matched_count <= 0:
        return

    resolved_durations = _resolve_decode_video_durations(
        target_durations=target_durations,
        target_ends=target_ends,
        prediction_count=matched_count,
    )
    if len(resolved_durations) == 0:
        return

    positions: list[int] = []
    starts: list[float] = []
    ends: list[float] = []
    durations: list[float] = []
    seqlen = sample_hidden_states.size(0)
    for target_idx in range(matched_count):
        position = response_offset + int(candidate_positions[target_idx].item())
        if position < 0 or position >= seqlen:
            continue
        positions.append(position)
        starts.append(float(target_starts[target_idx]))
        ends.append(float(target_ends[target_idx]))
        durations.append(float(resolved_durations[target_idx]))

    if len(positions) == 0:
        return

    device = sample_hidden_states.device
    position_tensor = torch.as_tensor(positions, device=device, dtype=torch.long)
    target_start = torch.as_tensor(starts, device=device, dtype=torch.float32)
    target_end = torch.as_tensor(ends, device=device, dtype=torch.float32)
    duration_tensor = torch.as_tensor(durations, device=device, dtype=torch.float32).clamp_min(1e-6)

    timespan_hidden = sample_hidden_states.index_select(0, position_tensor)
    if getattr(model, "use_cis_interface_adapter", False):
        timespan_embedding = model.cis_interface_adapter.forward_output(
            timespan_hidden,
            compute_diagnostics=False,
        ).adapted.float()
    else:
        timespan_embedding = timespan_hidden.float()

    target_embedding = model.cis_codec.encode(target_start, target_end, duration_tensor).float()
    embedding_mse_loss = (timespan_embedding - target_embedding).pow(2).mean(dim=-1)
    pred_norm = F.normalize(timespan_embedding, dim=-1)
    target_norm = F.normalize(target_embedding, dim=-1)
    embedding_cosine = (pred_norm * target_norm).sum(dim=-1)

    decoded = model.cis_codec.decode_relative(timespan_embedding)
    codec_losses = model.cis_codec.compute_loss_from_decoded(
        decoded,
        target_start_sec=target_start,
        target_end_sec=target_end,
        video_duration_sec=duration_tensor,
        reduction="mean",
    )

    detail_mapping = {
        "cis_total_loss": "total_loss",
        "cis_type_loss": "type_loss",
        "cis_point_loss": "point_loss",
        "timeple_loss": "span_loss",
        "cis_interval_l1": "interval_l1",
        "cis_point_l1": "point_l1",
        "timeple_giou_loss": "span_giou_loss",
        "timeple_iou": "span_iou",
        "cis_mae_start": "mae_start",
        "cis_mae_end": "mae_end",
        "cis_mae_total": "mae_total",
    }
    populated = False
    for aux_key, detail_key in detail_mapping.items():
        populated = _assign_aux_scalar(
            aux_features,
            key=aux_key,
            sample_idx=sample_idx,
            value=codec_losses.get(detail_key),
        ) or populated

    aux_features["cis_embedding_mse_loss"][sample_idx] = float(embedding_mse_loss.detach().mean().item())
    aux_features["cis_embedding_cosine"][sample_idx] = max(
        -1.0,
        min(1.0, float(embedding_cosine.detach().mean().item())),
    )
    aux_features["cis_embedding_valid"][sample_idx] = 1.0
    populated = True

    pred_start_sec = codec_losses.get("pred_start_sec")
    pred_end_sec = codec_losses.get("pred_end_sec")
    decoded_predictions: list[tuple[float, float]] = []
    if isinstance(pred_start_sec, torch.Tensor) and isinstance(pred_end_sec, torch.Tensor):
        pred_count = min(pred_start_sec.numel(), pred_end_sec.numel())
        for pred_idx in range(pred_count):
            decoded_predictions.append(
                (
                    float(pred_start_sec.flatten()[pred_idx].item()),
                    float(pred_end_sec.flatten()[pred_idx].item()),
                )
            )

    best_decoded_prediction = _select_best_decoded_prediction(
        decoded_predictions=decoded_predictions,
        target_starts=target_starts,
        target_ends=target_ends,
        prefix="cis",
    )
    if best_decoded_prediction is not None:
        for key, value in best_decoded_prediction.items():
            aux_features[key][sample_idx] = value
        aux_non_tensors["cis_decoded_segments"][sample_idx] = [
            [float(pred_start), float(pred_end)] for pred_start, pred_end in decoded_predictions
        ]
        aux_features["cis_decoded_valid"][sample_idx] = 1.0
        aux_features["cis_decode_source"][sample_idx] = decode_source

    if populated:
        aux_features["cis_metrics_valid"][sample_idx] = 1.0


def _populate_timeple_sample_features_from_hidden_states(
    aux_features: dict[str, torch.Tensor],
    aux_non_tensors: dict[str, list[Any]],
    *,
    model: nn.Module,
    sample_hidden_states: torch.Tensor,
    sample_info: dict[str, Any],
    response_offset: int,
    decode_source: float,
) -> None:
    sample_idx = int(sample_info.get("sample_idx", 0))
    aux_features["timeple_debug_decode_entered"][sample_idx] = 1.0
    if sample_hidden_states.dim() != 2:
        return

    candidate_positions = sample_info.get("candidate_positions")
    if not isinstance(candidate_positions, torch.Tensor):
        return

    target_starts = sample_info.get("target_starts", [])
    target_ends = sample_info.get("target_ends", [])
    target_durations = sample_info.get("target_durations", [])
    matched_count = min(int(candidate_positions.numel()), len(target_starts), len(target_ends))
    if matched_count <= 0:
        return

    resolved_durations = _resolve_decode_video_durations(
        target_durations=target_durations,
        target_ends=target_ends,
        prediction_count=matched_count,
    )
    if len(resolved_durations) == 0:
        return

    positions: list[int] = []
    starts: list[float] = []
    ends: list[float] = []
    durations: list[float] = []
    seqlen = sample_hidden_states.size(0)
    for target_idx in range(matched_count):
        position = response_offset + int(candidate_positions[target_idx].item())
        if position < 0 or position >= seqlen:
            continue
        positions.append(position)
        starts.append(float(target_starts[target_idx]))
        ends.append(float(target_ends[target_idx]))
        durations.append(float(resolved_durations[target_idx]))

    if len(positions) == 0:
        return
    aux_features["timeple_debug_decode_position_count"][sample_idx] = float(len(positions))

    device = sample_hidden_states.device
    position_tensor = torch.as_tensor(positions, device=device, dtype=torch.long)
    target_start = torch.as_tensor(starts, device=device, dtype=torch.float32)
    target_end = torch.as_tensor(ends, device=device, dtype=torch.float32)
    duration_tensor = torch.as_tensor(durations, device=device, dtype=torch.float32).clamp_min(1e-6)

    timespan_hidden = sample_hidden_states.index_select(0, position_tensor)
    if getattr(model, "use_timeple_interface_adapter", False):
        timespan_embedding = model.timeple_interface_adapter.forward_output(
            timespan_hidden,
            compute_diagnostics=False,
        ).adapted.float()
    else:
        timespan_embedding = timespan_hidden.float()

    decoded = model.timeple_codec.decode_relative(timespan_embedding)
    codec_losses = model.timeple_codec.compute_loss_from_decoded(
        decoded,
        target_start_sec=target_start,
        target_end_sec=target_end,
        video_duration_sec=duration_tensor,
        reduction="mean",
    )

    detail_mapping = {
        "timeple_total_loss": "total_loss",
        "timeple_dfl_loss": "dfl_loss",
        "timeple_iou_loss": "iou_loss",
        "timeple_span_loss": "span_loss",
        "timeple_interval_l1": "interval_l1",
        "timeple_boundary_loss": "boundary_loss",
        "timeple_span_giou_loss": "span_giou_loss",
        "timeple_span_iou": "span_iou",
        "timeple_mae_start": "mae_start",
        "timeple_mae_end": "mae_end",
        "timeple_mae_total": "mae_total",
    }
    populated = False
    for aux_key, detail_key in detail_mapping.items():
        populated = _assign_aux_scalar(
            aux_features,
            key=aux_key,
            sample_idx=sample_idx,
            value=codec_losses.get(detail_key),
        ) or populated

    pred_start_sec = codec_losses.get("pred_start_sec")
    pred_end_sec = codec_losses.get("pred_end_sec")
    decoded_predictions: list[tuple[float, float]] = []
    if isinstance(pred_start_sec, torch.Tensor) and isinstance(pred_end_sec, torch.Tensor):
        pred_count = min(pred_start_sec.numel(), pred_end_sec.numel())
        for pred_idx in range(pred_count):
            decoded_predictions.append(
                (
                    float(pred_start_sec.flatten()[pred_idx].item()),
                    float(pred_end_sec.flatten()[pred_idx].item()),
                )
            )
    aux_features["timeple_debug_decode_prediction_count"][sample_idx] = float(len(decoded_predictions))

    best_decoded_prediction = _select_best_decoded_prediction(
        decoded_predictions=decoded_predictions,
        target_starts=target_starts,
        target_ends=target_ends,
        prefix="timeple",
    )
    if best_decoded_prediction is not None:
        for key, value in best_decoded_prediction.items():
            aux_features[key][sample_idx] = value
        aux_non_tensors["timeple_decoded_segments"][sample_idx] = [
            [float(pred_start), float(pred_end)] for pred_start, pred_end in decoded_predictions
        ]
        aux_features["timeple_decoded_valid"][sample_idx] = 1.0
        aux_features["timeple_decode_source"][sample_idx] = decode_source
        aux_features["timeple_debug_decode_populated"][sample_idx] = 1.0

    if populated:
        aux_features["timeple_metrics_valid"][sample_idx] = 1.0


def _populate_timeple_sample_features_from_hidden_states(
    aux_features: dict[str, torch.Tensor],
    aux_non_tensors: dict[str, list[Any]],
    *,
    model: nn.Module,
    sample_hidden_states: torch.Tensor,
    sample_info: dict[str, Any],
    response_offset: int,
    decode_source: float,
) -> None:
    sample_idx = int(sample_info.get("sample_idx", 0))
    aux_features["timeple_debug_decode_entered"][sample_idx] = 1.0
    if sample_hidden_states.dim() != 2:
        return

    candidate_positions = sample_info.get("candidate_positions")
    if not isinstance(candidate_positions, torch.Tensor):
        return

    target_starts = sample_info.get("target_starts", [])
    target_ends = sample_info.get("target_ends", [])
    target_durations = sample_info.get("target_durations", [])
    matched_count = min(int(candidate_positions.numel()), len(target_starts), len(target_ends))
    if matched_count <= 0:
        return

    resolved_durations = _resolve_decode_video_durations(
        target_durations=target_durations,
        target_ends=target_ends,
        prediction_count=matched_count,
    )
    if len(resolved_durations) == 0:
        return

    positions: list[int] = []
    starts: list[float] = []
    ends: list[float] = []
    durations: list[float] = []
    seqlen = sample_hidden_states.size(0)
    for target_idx in range(matched_count):
        position = response_offset + int(candidate_positions[target_idx].item())
        if position < 0 or position >= seqlen:
            continue
        positions.append(position)
        starts.append(float(target_starts[target_idx]))
        ends.append(float(target_ends[target_idx]))
        durations.append(float(resolved_durations[target_idx]))

    if len(positions) == 0:
        return
    aux_features["timeple_debug_decode_position_count"][sample_idx] = float(len(positions))

    device = sample_hidden_states.device
    position_tensor = torch.as_tensor(positions, device=device, dtype=torch.long)
    target_start = torch.as_tensor(starts, device=device, dtype=torch.float32)
    target_end = torch.as_tensor(ends, device=device, dtype=torch.float32)
    duration_tensor = torch.as_tensor(durations, device=device, dtype=torch.float32).clamp_min(1e-6)

    timespan_hidden = sample_hidden_states.index_select(0, position_tensor)
    if getattr(model, "use_timeple_interface_adapter", False):
        timespan_embedding = model.timeple_interface_adapter.forward_output(
            timespan_hidden,
            compute_diagnostics=False,
        ).adapted.float()
    else:
        timespan_embedding = timespan_hidden.float()

    decoded = model.timeple_codec.decode_relative(timespan_embedding)
    codec_losses = model.timeple_codec.compute_loss_from_decoded(
        decoded,
        target_start_sec=target_start,
        target_end_sec=target_end,
        video_duration_sec=duration_tensor,
        reduction="mean",
    )

    detail_mapping = {
        "timeple_total_loss": "total_loss",
        "timeple_dfl_loss": "dfl_loss",
        "timeple_iou_loss": "iou_loss",
        "timeple_span_loss": "span_loss",
        "timeple_interval_l1": "interval_l1",
        "timeple_boundary_loss": "boundary_loss",
        "timeple_span_giou_loss": "span_giou_loss",
        "timeple_span_iou": "span_iou",
        "timeple_mae_start": "mae_start",
        "timeple_mae_end": "mae_end",
        "timeple_mae_total": "mae_total",
    }
    populated = False
    for aux_key, detail_key in detail_mapping.items():
        populated = _assign_aux_scalar(
            aux_features,
            key=aux_key,
            sample_idx=sample_idx,
            value=codec_losses.get(detail_key),
        ) or populated

    pred_start_sec = codec_losses.get("pred_start_sec")
    pred_end_sec = codec_losses.get("pred_end_sec")
    decoded_predictions: list[tuple[float, float]] = []
    if isinstance(pred_start_sec, torch.Tensor) and isinstance(pred_end_sec, torch.Tensor):
        pred_count = min(pred_start_sec.numel(), pred_end_sec.numel())
        for pred_idx in range(pred_count):
            decoded_predictions.append(
                (
                    float(pred_start_sec.flatten()[pred_idx].item()),
                    float(pred_end_sec.flatten()[pred_idx].item()),
                )
            )
    aux_features["timeple_debug_decode_prediction_count"][sample_idx] = float(len(decoded_predictions))

    best_decoded_prediction = _select_best_decoded_prediction(
        decoded_predictions=decoded_predictions,
        target_starts=target_starts,
        target_ends=target_ends,
        prefix="timeple",
    )
    if best_decoded_prediction is not None:
        for key, value in best_decoded_prediction.items():
            aux_features[key][sample_idx] = value
        aux_non_tensors["timeple_decoded_segments"][sample_idx] = [
            [float(pred_start), float(pred_end)] for pred_start, pred_end in decoded_predictions
        ]
        aux_features["timeple_decoded_valid"][sample_idx] = 1.0
        aux_features["timeple_decode_source"][sample_idx] = decode_source
        aux_features["timeple_debug_decode_populated"][sample_idx] = 1.0

    if populated:
        aux_features["timeple_metrics_valid"][sample_idx] = 1.0


def _populate_aux_features_from_batch_output(
    aux_features: dict[str, torch.Tensor],
    aux_non_tensors: dict[str, list[Any]],
    *,
    actor_module: nn.Module,
    model: nn.Module,
    output: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    multi_modal_inputs: dict[str, Any],
    time_codec_kwargs: dict[str, Any],
    sample_infos: list[dict[str, Any]],
    response_offset: int,
    is_timecodec_model: bool,
    force_param_context: bool,
) -> None:
    hidden_states, decode_source = _resolve_timecodec_decode_hidden_states(
        output=output,
        actor_module=actor_module,
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        multi_modal_inputs=multi_modal_inputs,
        time_codec_kwargs=time_codec_kwargs,
    )
    is_timeple_model = _is_timeple_model(model)
    if is_timeple_model:
        debug_sample_count = float(len(sample_infos))
        has_hidden = 1.0 if hidden_states is not None else 0.0
        for sample_info in sample_infos:
            sample_idx = int(sample_info["sample_idx"])
            aux_features["timeple_debug_sample_info_count"][sample_idx] = debug_sample_count
            aux_features["timeple_debug_hidden_available"][sample_idx] = has_hidden

    context = _codec_param_context(actor_module) if force_param_context else nullcontext()
    with context:
        if len(sample_infos) == 0 or hidden_states is None:
            return

        if len(sample_infos) == 1:
            sample_idx = int(sample_infos[0]["sample_idx"])
            if is_timecodec_model:
                populated = _populate_timecodec_features_from_forward_details(
                    aux_features,
                    time_loss_details=getattr(output, "time_loss_details", None),
                    sample_idx=sample_idx,
                )
                if populated:
                    aux_features["timecodec_metrics_valid"][sample_idx] = 1.0
            elif is_timeple_model:
                populated = _populate_timeple_features_from_forward_details(
                    aux_features,
                    timeple_loss_details=getattr(output, "timeple_loss_details", None),
                    sample_idx=sample_idx,
                )
                if populated:
                    aux_features["timeple_metrics_valid"][sample_idx] = 1.0
            elif _is_timeed_model(model):
                populated = _populate_timeed_features_from_forward_details(
                    aux_features,
                    timeed_loss_details=getattr(output, "timeed_loss_details", None),
                    sample_idx=sample_idx,
                )
                if populated:
                    aux_features["timeed_metrics_valid"][sample_idx] = 1.0

        for sample_info in sample_infos:
            sample_idx = int(sample_info["sample_idx"])
            if hidden_states.dim() == 3:
                sample_hidden_states = hidden_states[sample_idx]
            else:
                sample_hidden_states = hidden_states

            if _is_cis_model(model):
                _populate_cis_sample_features_from_hidden_states(
                    aux_features,
                    aux_non_tensors,
                    model=model,
                    sample_hidden_states=sample_hidden_states,
                    sample_info=sample_info,
                    response_offset=response_offset,
                    decode_source=decode_source,
                )
                continue

            if is_timeple_model:
                _populate_timeple_sample_features_from_hidden_states(
                    aux_features,
                    aux_non_tensors,
                    model=model,
                    sample_hidden_states=sample_hidden_states,
                    sample_info=sample_info,
                    response_offset=response_offset,
                    decode_source=decode_source,
                )
                continue

            if _is_timeple_model(model):
                _populate_timeple_sample_features_from_hidden_states(
                    aux_features,
                    aux_non_tensors,
                    model=model,
                    sample_hidden_states=sample_hidden_states,
                    sample_info=sample_info,
                    response_offset=response_offset,
                    decode_source=decode_source,
                )
                continue

            is_timeed_model = _is_timeed_model(model)
            prediction_prefix = "timeed" if is_timeed_model else "timecodec"
            decoded_predictions = _decode_timecodec_predictions_from_hidden_states(
                actor_module=actor_module,
                model=model,
                hidden_states=sample_hidden_states,
                time_token_positions=[
                    int(response_offset + int(rel_position.item()))
                    for rel_position in sample_info["candidate_positions"]
                ],
                video_durations=_resolve_decode_video_durations(
                    target_durations=sample_info.get("target_durations", []),
                    target_ends=sample_info["target_ends"],
                    prediction_count=len(sample_info["candidate_positions"]),
                ),
                summon_full_params=False,
            )
            best_decoded_prediction = _select_best_decoded_prediction(
                decoded_predictions=decoded_predictions,
                target_starts=sample_info["target_starts"],
                target_ends=sample_info["target_ends"],
                prefix=prediction_prefix,
            )
            if best_decoded_prediction is None:
                continue
            for key, value in best_decoded_prediction.items():
                aux_features[key][sample_idx] = value
            if is_timeed_model:
                aux_non_tensors["timeed_decoded_segments"][sample_idx] = [
                    [float(pred_start), float(pred_end)] for pred_start, pred_end in decoded_predictions
                ]
                aux_features["timeed_decoded_valid"][sample_idx] = 1.0
                aux_features["timeed_decode_source"][sample_idx] = decode_source
            else:
                aux_features["timecodec_decoded_valid"][sample_idx] = 1.0
                aux_features["timecodec_decode_source"][sample_idx] = decode_source


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    def _forward_micro_batch(self, micro_batch: dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}
        time_codec_kwargs = _build_time_codec_kwargs(micro_batch)

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            if len(time_codec_kwargs) != 0:
                if self.config.ulysses_size > 1:
                    raise NotImplementedError(
                        "padding_free with time-codec labels is not supported when ulysses_size > 1."
                    )
                time_codec_kwargs = _flatten_time_codec_kwargs_for_padding_free(
                    time_codec_kwargs,
                    batch_size=batch_size,
                    seqlen=seqlen,
                    indices=indices,
                )

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                **time_codec_kwargs,
                use_cache=False,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                **time_codec_kwargs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

        return log_probs

    def _forward_micro_batch_with_timeed_span_log_probs(
        self,
        micro_batch: dict[str, Any],
        temperature: float,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, dict[str, float]]:
        if self.config.padding_free:
            raise NotImplementedError("TimeED span-policy GRPO currently requires actor.padding_free=false.")

        input_ids = micro_batch["input_ids"]
        batch_size = input_ids.size(0)
        device = input_ids.device
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_mask = micro_batch.get("response_mask")
        response_length = responses.size(-1)
        if position_ids.dim() == 3:
            position_ids = position_ids.transpose(0, 1)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}
        time_codec_kwargs = _build_time_codec_kwargs(micro_batch)
        model = _unwrap_model(self.actor_module)
        timespan_token_id = getattr(model, "timespan_token_id", None)
        if _is_timeed_model(model):
            time_codec_kwargs = {
                **time_codec_kwargs,
                "compute_timeed_loss": False,
                "timeed_anchor_loss_weight": 0.0,
            }
        aux_features = _build_empty_timecodec_features(batch_size=batch_size, device=device)
        if timespan_token_id is not None:
            generated_time_codec_kwargs, _ = _build_generated_timespan_supervision(
                micro_batch=micro_batch,
                responses=responses,
                response_mask=response_mask,
                timespan_token_id=int(timespan_token_id),
                response_offset=input_ids.size(1) - response_length,
                aux_features=aux_features,
            )
            if len(generated_time_codec_kwargs) != 0:
                time_codec_kwargs = {
                    **time_codec_kwargs,
                    **generated_time_codec_kwargs,
                    "compute_timeed_loss": float(getattr(self.config, "timeed_span_aux_loss_weight", 0.0)) > 0.0,
                    "timeed_dfl_weight": float(getattr(self.config, "timeed_span_aux_dfl_weight", 1.0)),
                    "timeed_giou_weight": float(getattr(self.config, "timeed_span_aux_giou_weight", 1.0)),
                    "timeed_anchor_loss_weight": 0.0,
                    "timeed_loss_weight": 1.0,
                }
        if _timeed_forward_supports_span_policy(model) and timespan_token_id is not None:
            time_codec_kwargs = {
                **time_codec_kwargs,
                "timeed_span_sample_positions": _build_timeed_span_sample_positions_from_responses(
                    input_ids=input_ids,
                    responses=responses,
                    response_mask=response_mask,
                    response_length=response_length,
                    timespan_token_id=int(timespan_token_id),
                ),
            }

        output = self.actor_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **multi_modal_inputs,
            **time_codec_kwargs,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        logits: torch.Tensor = output.logits
        logits.div_(temperature)
        logits = logits[:, -response_length - 1 : -1, :]
        log_probs = self.log_probs_from_logits(logits, responses)

        span_logits = _timeed_span_logits_or_fallback(
            model=model,
            output=output,
            input_ids=input_ids,
            response_length=response_length,
            responses=responses,
            response_mask=response_mask,
        )
        span_valid = _timeed_span_valid_mask_from_responses(
            input_ids=input_ids,
            responses=responses,
            response_mask=response_mask,
            response_length=response_length,
            timespan_token_id=timespan_token_id,
        )

        timeed_aux_loss = torch.zeros((), device=device, dtype=torch.float32)
        timeed_aux_metrics: dict[str, float] = {
            "timeed_span_aux_valid": 0.0,
            "timeed_span_aux_loss": 0.0,
        }
        output_timeed_loss = getattr(output, "timeed_loss", None)
        timeed_loss_details = getattr(output, "timeed_loss_details", None)
        if isinstance(timeed_loss_details, dict) and isinstance(
            timeed_loss_details.get("timespan_loss_total"), torch.Tensor
        ):
            output_timeed_loss = timeed_loss_details["timespan_loss_total"]
        if isinstance(output_timeed_loss, torch.Tensor):
            timeed_aux_loss = output_timeed_loss.float()
            timeed_aux_metrics["timeed_span_aux_valid"] = 1.0
            timeed_aux_metrics["timeed_span_aux_loss"] = float(timeed_aux_loss.detach().item())
        if isinstance(timeed_loss_details, dict):
            for metric_key, detail_key in (
                ("timeed_span_aux_dfl", "timespan_loss_dfl"),
                ("timeed_span_aux_giou", "timespan_loss_giou"),
                ("timeed_span_aux_total", "timespan_loss_total"),
                ("timeed_span_aux_iou", "timespan_span_iou"),
                ("timeed_span_aux_iou_argmax", "timespan_span_iou_argmax"),
            ):
                scalar = _maybe_to_float_scalar(timeed_loss_details.get(detail_key))
                if scalar is not None:
                    timeed_aux_metrics[metric_key] = scalar

        return log_probs, span_logits, span_valid, timeed_aux_loss, timeed_aux_metrics

    def _forward_micro_batch_with_csdo(
        self,
        micro_batch: dict[str, Any],
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config.padding_free:
            raise NotImplementedError("TimePLE Counterfactual Span Distribution Optimization (CSDO) requires actor.padding_free=false.")

        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_mask = micro_batch.get("response_mask")
        response_length = responses.size(-1)
        if position_ids.dim() == 3:
            position_ids = position_ids.transpose(0, 1)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        model = _unwrap_model(self.actor_module)
        if not _is_timeple_model(model):
            raise ValueError("csdo_enabled=True requires qwen3_vl_timeple.")

        csdo_positions = _csdo_positions_from_responses(
            input_ids=input_ids,
            responses=responses,
            response_mask=response_mask,
            response_length=response_length,
            timespan_token_id=getattr(model, "timespan_token_id", None),
        )
        csdo_forward_kwargs: dict[str, Any] = {}
        if _csdo_forward_supports_outputs(model):
            csdo_forward_kwargs = {
                "compute_csdo_outputs": True,
                "csdo_positions": csdo_positions,
            }
        elif isinstance(self.actor_module, FSDP):
            raise RuntimeError(
                "TimePLE Counterfactual Span Distribution Optimization (CSDO) under FSDP requires a synced model forward patch with "
                "compute_csdo_outputs/csdo_positions support."
            )

        time_codec_kwargs = {
            **_build_time_codec_kwargs(micro_batch),
            "compute_timeple_loss": False,
        }
        output = self.actor_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **multi_modal_inputs,
            **time_codec_kwargs,
            **csdo_forward_kwargs,
            use_cache=False,
            output_hidden_states=not bool(csdo_forward_kwargs),
            return_dict=True,
        )
        logits: torch.Tensor = output.logits
        logits.div_(temperature)
        logits = logits[:, -response_length - 1 : -1, :]
        log_probs = self.log_probs_from_logits(logits, responses)
        csdo_current_logits, csdo_current_features, csdo_valid = (
            _csdo_logits_features_from_forward_output(
                actor_module=self.actor_module,
                model=model,
                output=output,
                input_ids=input_ids,
                responses=responses,
                response_mask=response_mask,
                response_length=response_length,
            )
        )
        return (
            log_probs,
            csdo_current_logits,
            csdo_current_features,
            csdo_valid,
        )

    def _forward_micro_batch_with_tr_spd(
        self,
        micro_batch: dict[str, Any],
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config.padding_free:
            raise NotImplementedError("TimePLE Trust-Region Span Posterior Distillation (TR-SPD) requires actor.padding_free=false.")

        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_mask = micro_batch.get("response_mask")
        response_length = responses.size(-1)
        if position_ids.dim() == 3:
            position_ids = position_ids.transpose(0, 1)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        model = _unwrap_model(self.actor_module)
        if not _is_timeple_model(model):
            raise ValueError("tr_spd_enabled=True requires qwen3_vl_timeple.")

        tr_spd_positions = _tr_spd_positions_from_responses(
            input_ids=input_ids,
            responses=responses,
            response_mask=response_mask,
            response_length=response_length,
            timespan_token_id=getattr(model, "timespan_token_id", None),
        )
        tr_spd_forward_kwargs: dict[str, Any] = {}
        if _tr_spd_forward_supports_outputs(model):
            tr_spd_forward_kwargs = {
                "compute_tr_spd_outputs": True,
                "tr_spd_positions": tr_spd_positions,
            }
        elif isinstance(self.actor_module, FSDP):
            raise RuntimeError(
                "TimePLE Trust-Region Span Posterior Distillation (TR-SPD) under FSDP requires a synced model forward patch with "
                "compute_tr_spd_outputs/tr_spd_positions support."
            )

        time_codec_kwargs = {
            **_build_time_codec_kwargs(micro_batch),
            "compute_timeple_loss": False,
        }
        output = self.actor_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **multi_modal_inputs,
            **time_codec_kwargs,
            **tr_spd_forward_kwargs,
            use_cache=False,
            output_hidden_states=not bool(tr_spd_forward_kwargs),
            return_dict=True,
        )
        logits: torch.Tensor = output.logits
        logits.div_(temperature)
        logits = logits[:, -response_length - 1 : -1, :]
        log_probs = self.log_probs_from_logits(logits, responses)
        tr_spd_current_logits, tr_spd_current_features, tr_spd_valid = (
            _tr_spd_logits_features_from_forward_output(
                actor_module=self.actor_module,
                model=model,
                output=output,
                input_ids=input_ids,
                responses=responses,
                response_mask=response_mask,
                response_length=response_length,
            )
        )
        return (
            log_probs,
            tr_spd_current_logits,
            tr_spd_current_features,
            tr_spd_valid,
        )

    def _forward_micro_batch_with_cis_aux_loss(
        self,
        micro_batch: dict[str, Any],
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_mask = micro_batch.get("response_mask")
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        model = _unwrap_model(self.actor_module)
        time_codec_kwargs = _build_time_codec_kwargs(micro_batch)
        cis_aux_metrics: dict[str, float] = {}
        cis_aux_requested = False
        sample_infos: list[dict[str, Any]] = []

        if _is_cis_model(model):
            timespan_token_id = getattr(model, "timespan_token_id", None)
            if timespan_token_id is not None:
                aux_features = _build_empty_timecodec_features(batch_size=batch_size, device=input_ids.device)
                generated_time_codec_kwargs, sample_infos = _build_generated_timespan_supervision(
                    micro_batch=micro_batch,
                    responses=responses,
                    response_mask=response_mask,
                    timespan_token_id=timespan_token_id,
                    response_offset=input_ids.size(1) - response_length,
                    aux_features=aux_features,
                )
                if len(generated_time_codec_kwargs) != 0:
                    cis_aux_decode_weight = max(float(self.config.cis_aux_decode_loss_weight), 0.0)
                    cis_aux_embedding_weight = max(float(self.config.cis_aux_embedding_loss_weight), 0.0)
                    cis_aux_embedding_cosine_weight = max(
                        float(self.config.cis_aux_embedding_cosine_loss_weight),
                        0.0,
                    )
                    cis_aux_component_weight = (
                        cis_aux_decode_weight + cis_aux_embedding_weight + cis_aux_embedding_cosine_weight
                    )
                    if cis_aux_component_weight > 0.0:
                        cis_aux_decode_weight /= cis_aux_component_weight
                        cis_aux_embedding_weight /= cis_aux_component_weight
                        cis_aux_embedding_cosine_weight /= cis_aux_component_weight
                    cis_aux_requested = True
                    time_codec_kwargs = {
                        **time_codec_kwargs,
                        **generated_time_codec_kwargs,
                        "compute_cis_loss": not self.config.padding_free and cis_aux_component_weight > 0.0,
                        "cis_decode_loss_weight": cis_aux_decode_weight,
                        "cis_codec_recon_loss_weight": 0.0,
                        "cis_embedding_loss_weight": cis_aux_embedding_weight,
                        "cis_embedding_cosine_loss_weight": cis_aux_embedding_cosine_weight,
                        "cis_reencoding_loss_weight": 0.0,
                        "cis_decode_loss_mode": "soft_iou",
                        "cis_adapter_regularization_weight": 0.0,
                        "cis_aux_loss_only": True,
                    }

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            if len(time_codec_kwargs) != 0:
                if self.config.ulysses_size > 1:
                    raise NotImplementedError(
                        "padding_free with time-codec labels is not supported when ulysses_size > 1."
                    )
                time_codec_kwargs = _flatten_time_codec_kwargs_for_padding_free(
                    time_codec_kwargs,
                    batch_size=batch_size,
                    seqlen=seqlen,
                    indices=indices,
                )

            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                **time_codec_kwargs,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            logits_rmpad = output.logits.squeeze(0)
            logits_rmpad.div_(temperature)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            if self.config.ulysses_size > 1:
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                **time_codec_kwargs,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]
            log_probs = self.log_probs_from_logits(logits, responses)

        cis_aux_loss = torch.zeros((), device=input_ids.device, dtype=torch.float32)
        cis_aux_metrics = {
            "cis_aux_valid": 0.0,
            "cis_aux_loss": 0.0,
        }
        if cis_aux_requested and not self.config.padding_free:
            forward_cis_loss, forward_cis_metrics = _build_clean_cis_aux_metrics_from_forward(
                output=output,
                aligned_count=_count_aligned_generated_timespans(sample_infos),
            )
            if forward_cis_loss is not None:
                cis_aux_loss = forward_cis_loss
                cis_aux_metrics = forward_cis_metrics

        return log_probs, cis_aux_loss, cis_aux_metrics

    def _compute_clean_cis_aux_loss(
        self,
        *,
        model: nn.Module,
        output: Any,
        sample_infos: list[dict[str, Any]],
        response_offset: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        zero = torch.zeros((), device=device, dtype=torch.float32)
        metrics: dict[str, float] = {
            "cis_aux_valid": 0.0,
            "cis_aux_loss": 0.0,
        }
        if len(sample_infos) == 0 or not _is_cis_model(model):
            return zero, metrics

        hidden_states = _extract_last_hidden_state_from_output(output)
        if hidden_states is None:
            metrics["cis_aux_invalid_hidden"] = 1.0
            return zero, metrics

        flat_batch_indices: list[int] = []
        flat_positions: list[int] = []
        flat_starts: list[float] = []
        flat_ends: list[float] = []
        flat_durations: list[float] = []
        for sample_info in sample_infos:
            sample_idx = int(sample_info["sample_idx"])
            candidate_positions = sample_info["candidate_positions"]
            target_starts = sample_info["target_starts"]
            target_ends = sample_info["target_ends"]
            target_durations = sample_info.get("target_durations", [])
            matched_count = min(int(candidate_positions.numel()), len(target_starts), len(target_ends))
            if matched_count <= 0:
                continue

            resolved_durations = _resolve_decode_video_durations(
                target_durations=target_durations,
                target_ends=target_ends,
                prediction_count=matched_count,
            )
            for target_idx in range(matched_count):
                flat_batch_indices.append(sample_idx)
                flat_positions.append(response_offset + int(candidate_positions[target_idx].item()))
                flat_starts.append(float(target_starts[target_idx]))
                flat_ends.append(float(target_ends[target_idx]))
                flat_durations.append(float(resolved_durations[target_idx]))

        if len(flat_positions) == 0:
            return zero, metrics

        batch_indices = torch.as_tensor(flat_batch_indices, device=device, dtype=torch.long)
        positions = torch.as_tensor(flat_positions, device=device, dtype=torch.long)
        target_start = torch.as_tensor(flat_starts, device=device, dtype=torch.float32)
        target_end = torch.as_tensor(flat_ends, device=device, dtype=torch.float32)
        durations = torch.as_tensor(flat_durations, device=device, dtype=torch.float32).clamp_min(1e-6)

        if hidden_states.dim() == 2:
            timespan_hidden = hidden_states[positions, :]
        else:
            timespan_hidden = hidden_states[batch_indices, positions, :]
        if timespan_hidden.shape[-1] <= 0:
            metrics["cis_aux_invalid_hidden"] = 1.0
            return zero, metrics

        if getattr(model, "use_cis_interface_adapter", False):
            timespan_embedding = model.cis_interface_adapter.forward_output(timespan_hidden).adapted.float()
        else:
            timespan_embedding = timespan_hidden.float()

        with torch.no_grad():
            target_embedding = model.cis_codec.encode(target_start, target_end, durations).float()

        component_losses: list[tuple[torch.Tensor, float]] = []
        embedding_mse_loss = (timespan_embedding - target_embedding).pow(2).mean(dim=-1)
        if self.config.cis_aux_embedding_loss_weight > 0.0:
            component_losses.append((embedding_mse_loss.mean(), float(self.config.cis_aux_embedding_loss_weight)))

        pred_norm = F.normalize(timespan_embedding, dim=-1)
        target_norm = F.normalize(target_embedding, dim=-1)
        embedding_cosine = (pred_norm * target_norm).sum(dim=-1)
        embedding_cosine_loss = 1.0 - embedding_cosine
        if self.config.cis_aux_embedding_cosine_loss_weight > 0.0:
            component_losses.append(
                (embedding_cosine_loss.mean(), float(self.config.cis_aux_embedding_cosine_loss_weight))
            )

        decoded = model.cis_codec.decode_relative(timespan_embedding)
        target_start_rel, target_end_rel, _ = model.cis_codec._to_relative(target_start, target_end, durations)
        decoded_iou = _tensor_interval_iou(
            decoded.pred_start_rel.float(),
            decoded.pred_end_rel.float(),
            target_start_rel.float(),
            target_end_rel.float(),
        )
        decoded_iou_loss = 1.0 - decoded_iou
        if self.config.cis_aux_decode_loss_weight > 0.0:
            component_losses.append((decoded_iou_loss.mean(), float(self.config.cis_aux_decode_loss_weight)))

        total_weight = sum(max(weight, 0.0) for _, weight in component_losses)
        if total_weight <= 0.0:
            return zero, metrics

        aux_loss = sum(loss * max(weight, 0.0) for loss, weight in component_losses) / total_weight
        interval_l1 = (
            (decoded.pred_start_rel.float() - target_start_rel.float()).abs()
            + (decoded.pred_end_rel.float() - target_end_rel.float()).abs()
        ) * 0.5

        metrics.update(
            {
                "cis_aux_valid": 1.0,
                "cis_aux_loss": float(aux_loss.detach().item()),
                "cis_aux_aligned_count": float(len(flat_positions)),
                "cis_aux_embedding_mse_loss": float(embedding_mse_loss.detach().mean().item()),
                "cis_aux_embedding_cosine_loss": float(embedding_cosine_loss.detach().mean().item()),
                "cis_aux_embedding_cosine": float(embedding_cosine.detach().mean().item()),
                "cis_aux_decoded_iou_loss": float(decoded_iou_loss.detach().mean().item()),
                "cis_aux_decoded_iou": float(decoded_iou.detach().mean().item()),
                "cis_aux_interval_l1": float(interval_l1.detach().mean().item()),
            }
        )
        return aux_loss, metrics

    def _forward_micro_batch_with_aux(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, list[Any]]]:
        input_ids = micro_batch["input_ids"]
        batch_size, _ = input_ids.shape
        device = input_ids.device
        aux_features = _build_empty_timecodec_features(batch_size=batch_size, device=device)
        aux_non_tensors = _build_empty_timecodec_non_tensor_features(batch_size=batch_size)

        model = _unwrap_model(self.actor_module)
        is_timecodec_model = _is_timecodec_model(model)
        is_cis_codec_model = _is_cis_model(model)
        is_timeple_model = _is_timeple_model(model)
        is_timeed_model = _is_timeed_model(model)
        csdo_enabled = bool(getattr(self.config, "csdo_enabled", False))
        tr_spd_enabled = bool(getattr(self.config, "tr_spd_enabled", False))
        if csdo_enabled and self.config.padding_free:
            raise NotImplementedError("TimePLE Counterfactual Span Distribution Optimization (CSDO) requires actor.padding_free=false.")
        if csdo_enabled and not is_timeple_model:
            raise ValueError("csdo_enabled=True requires qwen3_vl_timeple.")
        if tr_spd_enabled and self.config.padding_free:
            raise NotImplementedError("TimePLE Trust-Region Span Posterior Distillation (TR-SPD) requires actor.padding_free=false.")
        if tr_spd_enabled and not is_timeple_model:
            raise ValueError("tr_spd_enabled=True requires qwen3_vl_timeple.")
        if is_timeple_model:
            aux_features["timeple_debug_is_model"].fill_(1.0)
        if (
            not is_timecodec_model
            and not is_cis_codec_model
            and not is_timeple_model
            and not is_timeed_model
        ):
            return self._forward_micro_batch(micro_batch, temperature=temperature), aux_features, aux_non_tensors

        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_mask = micro_batch.get("response_mask")
        response_length = responses.size(-1)
        if position_ids.dim() == 3:
            position_ids = position_ids.transpose(0, 1)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        time_codec_kwargs = _build_time_codec_kwargs(micro_batch)
        timespan_token_id = getattr(model, "timespan_token_id", None)
        response_offset = input_ids.size(1) - response_length
        sample_infos: list[dict[str, Any]] = []
        if is_timeple_model and timespan_token_id is not None:
            aux_features["timeple_debug_has_timespan_token_id"].fill_(1.0)
        if timespan_token_id is not None:
            generated_time_codec_kwargs, sample_infos = _build_generated_timespan_supervision(
                micro_batch=micro_batch,
                responses=responses,
                response_mask=response_mask,
                timespan_token_id=timespan_token_id,
                response_offset=response_offset,
                aux_features=aux_features,
            )
            if len(generated_time_codec_kwargs) != 0:
                time_codec_kwargs = {**time_codec_kwargs, **generated_time_codec_kwargs}
                if (
                    (is_timeple_model or is_timeple_model)
                    and "timespan_positions" in generated_time_codec_kwargs
                ):
                    aux_features["timeple_debug_forward_has_timespan_positions"].fill_(1.0)
                if is_cis_codec_model:
                    time_codec_kwargs = {
                        **time_codec_kwargs,
                        "compute_cis_loss": False,
                    }
                elif is_timeple_model:
                    time_codec_kwargs = {
                        **time_codec_kwargs,
                        "compute_timeple_loss": True,
                        "timeple_decode_loss_weight": 1.0,
                        "timeple_dfl_loss_weight": 1.0,
                        "timeple_iou_loss_weight": 1.0,
                    }
                elif is_timeple_model:
                    time_codec_kwargs = {
                        **time_codec_kwargs,
                        "compute_timeple_loss": not (
                            csdo_enabled or tr_spd_enabled
                        ),
                        "timeple_decode_loss_weight": 1.0,
                        "timeple_dfl_loss_weight": 1.0,
                        "timeple_iou_loss_weight": 1.0,
                    }
                elif is_timeed_model:
                    time_codec_kwargs = {
                        **time_codec_kwargs,
                        "compute_timeed_loss": True,
                    }
        if (
            bool(getattr(self.config, "timeed_span_grpo_enabled", False))
            and is_timeed_model
            and _timeed_forward_supports_span_policy(model)
            and timespan_token_id is not None
        ):
            time_codec_kwargs = {
                **time_codec_kwargs,
                "timeed_span_sample_positions": _build_timeed_span_sample_positions_from_responses(
                    input_ids=input_ids,
                    responses=responses,
                    response_mask=response_mask,
                    response_length=response_length,
                    timespan_token_id=int(timespan_token_id),
                ),
            }
        if csdo_enabled and is_timeple_model:
            time_codec_kwargs = {
                **time_codec_kwargs,
                "compute_timeple_loss": False,
            }
        if tr_spd_enabled and is_timeple_model:
            time_codec_kwargs = {
                **time_codec_kwargs,
                "compute_timeple_loss": False,
            }
        csdo_forward_kwargs: dict[str, Any] = {}
        if csdo_enabled and is_timeple_model:
            csdo_positions = _csdo_positions_from_responses(
                input_ids=input_ids,
                responses=responses,
                response_mask=response_mask,
                response_length=response_length,
                timespan_token_id=timespan_token_id,
            )
            if _csdo_forward_supports_outputs(model):
                csdo_forward_kwargs = {
                    "compute_csdo_outputs": True,
                    "csdo_positions": csdo_positions,
                }
            elif isinstance(self.actor_module, FSDP):
                raise RuntimeError(
                    "TimePLE Counterfactual Span Distribution Optimization (CSDO) under FSDP requires a synced model forward patch with "
                    "compute_csdo_outputs/csdo_positions support."
                )
        tr_spd_forward_kwargs: dict[str, Any] = {}
        if tr_spd_enabled and is_timeple_model:
            tr_spd_positions = _tr_spd_positions_from_responses(
                input_ids=input_ids,
                responses=responses,
                response_mask=response_mask,
                response_length=response_length,
                timespan_token_id=timespan_token_id,
            )
            if _tr_spd_forward_supports_outputs(model):
                tr_spd_forward_kwargs = {
                    "compute_tr_spd_outputs": True,
                    "tr_spd_positions": tr_spd_positions,
                }
            elif isinstance(self.actor_module, FSDP):
                raise RuntimeError(
                    "TimePLE Trust-Region Span Posterior Distillation (TR-SPD) under FSDP requires a synced model forward patch with "
                    "compute_tr_spd_outputs/tr_spd_positions support."
                )

        if self.config.padding_free:
            log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            if is_timecodec_model:
                codec_module = getattr(model, "time_codec", None)
            elif is_cis_codec_model:
                codec_module = getattr(model, "cis_codec", None)
            elif is_timeple_model:
                codec_module = getattr(model, "timeple_codec", None)
            elif is_timeple_model:
                codec_module = getattr(model, "timeple_codec", None)
            else:
                codec_module = getattr(model, "timeed_decoder", None)
            if (is_timeple_model or is_timeple_model) and codec_module is not None:
                aux_features["timeple_debug_has_codec_module"].fill_(1.0)
            if timespan_token_id is None or codec_module is None or len(sample_infos) == 0:
                return log_probs, aux_features, aux_non_tensors

            if isinstance(self.actor_module, FSDP):
                return log_probs, aux_features, aux_non_tensors

            for sample_info in sample_infos:
                sample_idx = sample_info["sample_idx"]
                single_position_ids = (
                    position_ids[:, sample_idx : sample_idx + 1, :]
                    if position_ids.dim() == 3
                    else position_ids[sample_idx : sample_idx + 1]
                )
                single_metrics, single_non_tensors = self._compute_single_sample_timecodec_aux(
                    model=model,
                    input_ids=input_ids[sample_idx : sample_idx + 1],
                    attention_mask=attention_mask[sample_idx : sample_idx + 1],
                    position_ids=single_position_ids,
                    multi_modal_inputs=_slice_multi_modal_inputs_for_sample(multi_modal_inputs, sample_idx),
                    time_codec_kwargs=_slice_time_codec_kwargs_for_sample(time_codec_kwargs, sample_idx),
                    target_starts=sample_info["target_starts"],
                    target_ends=sample_info["target_ends"],
                    target_durations=sample_info.get("target_durations", []),
                )
                for key, value in single_metrics.items():
                    aux_features[key][sample_idx] = value
                for key, value in single_non_tensors.items():
                    aux_non_tensors[key][sample_idx] = value

            return log_probs, aux_features, aux_non_tensors

        output = self.actor_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **multi_modal_inputs,
            **time_codec_kwargs,
            **csdo_forward_kwargs,
            **tr_spd_forward_kwargs,
            use_cache=False,
            output_hidden_states=(
                (is_timeple_model or is_timeple_model)
                and not bool(csdo_forward_kwargs)
                and not bool(tr_spd_forward_kwargs)
            ),
            return_dict=True,
        )
        logits: torch.Tensor = output.logits
        logits.div_(temperature)
        logits = logits[:, -response_length - 1 : -1, :]
        log_probs = self.log_probs_from_logits(logits, responses)

        if is_timecodec_model:
            codec_module = getattr(model, "time_codec", None)
        elif is_cis_codec_model:
            codec_module = getattr(model, "cis_codec", None)
        elif is_timeple_model:
            codec_module = getattr(model, "timeple_codec", None)
        elif is_timeple_model:
            codec_module = getattr(model, "timeple_codec", None)
        else:
            codec_module = getattr(model, "timeed_decoder", None)
        if (is_timeple_model or is_timeple_model) and codec_module is not None:
            aux_features["timeple_debug_has_codec_module"].fill_(1.0)
        if csdo_enabled and (timespan_token_id is None or codec_module is None):
            raise ValueError("CIS Counterfactual Span Distribution Optimization (CSDO) requires timespan_token_id and timeple_codec.")
        if tr_spd_enabled and (timespan_token_id is None or codec_module is None):
            raise ValueError("CIS Trust-Region Span Posterior Distillation (TR-SPD) requires timespan_token_id and timeple_codec.")
        if timespan_token_id is None or codec_module is None:
            return log_probs, aux_features, aux_non_tensors

        _populate_aux_features_from_batch_output(
            aux_features,
            aux_non_tensors,
            actor_module=self.actor_module,
            model=model,
            output=output,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            multi_modal_inputs=multi_modal_inputs,
            time_codec_kwargs=time_codec_kwargs,
            sample_infos=sample_infos,
            response_offset=response_offset,
            is_timecodec_model=is_timecodec_model,
            force_param_context=isinstance(self.actor_module, FSDP),
        )

        if csdo_enabled and is_timeple_model:
            csdo_old_logits, csdo_old_features, csdo_valid = (
                _csdo_logits_features_from_forward_output(
                    actor_module=self.actor_module,
                    model=model,
                    output=output,
                    input_ids=input_ids,
                    responses=responses,
                    response_mask=response_mask,
                    response_length=response_length,
                )
            )
            aux_features["csdo_old_logits"] = csdo_old_logits.detach().float()
            aux_features["csdo_old_features"] = csdo_old_features.detach().float()
            aux_features["csdo_valid"] = csdo_valid.detach().float()

        if tr_spd_enabled and is_timeple_model:
            tr_spd_old_logits, tr_spd_old_features, tr_spd_valid = (
                _tr_spd_logits_features_from_forward_output(
                    actor_module=self.actor_module,
                    model=model,
                    output=output,
                    input_ids=input_ids,
                    responses=responses,
                    response_mask=response_mask,
                    response_length=response_length,
                )
            )
            aux_features["tr_spd_old_logits"] = tr_spd_old_logits.detach().float()
            aux_features["tr_spd_old_features"] = tr_spd_old_features.detach().float()
            aux_features["tr_spd_valid"] = tr_spd_valid.detach().float()

        if bool(getattr(self.config, "timeed_span_grpo_enabled", False)) and is_timeed_model:
            span_logits = _timeed_span_logits_or_fallback(
                model=model,
                output=output,
                input_ids=input_ids,
                response_length=response_length,
                responses=responses,
                response_mask=response_mask,
            )
            if isinstance(span_logits, torch.Tensor):
                aux_features["timeed_span_old_logits"] = span_logits.detach().float()
                aux_features["timeed_span_valid"] = _timeed_span_valid_mask_from_responses(
                    input_ids=input_ids,
                    responses=responses,
                    response_mask=response_mask,
                    response_length=response_length,
                    timespan_token_id=timespan_token_id,
                )

        return log_probs, aux_features, aux_non_tensors

    def _compute_single_sample_timecodec_aux(
        self,
        *,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        multi_modal_inputs: dict[str, Any],
        time_codec_kwargs: dict[str, Any],
        target_starts: list[float],
        target_ends: list[float],
        target_durations: list[float],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        sample_aux = _build_empty_timecodec_features(batch_size=1, device=input_ids.device)
        sample_non_tensors = _build_empty_timecodec_non_tensor_features(batch_size=1)
        if len(target_starts) == 0 or len(target_ends) == 0:
            return (
                {key: float(value[0].item()) for key, value in sample_aux.items()},
                {key: value[0] for key, value in sample_non_tensors.items()},
            )

        forward_time_codec_kwargs = time_codec_kwargs
        forward_position_ids = position_ids
        if self.config.padding_free:
            batch_size, seqlen = input_ids.shape
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
            forward_time_codec_kwargs = _flatten_time_codec_kwargs_for_padding_free(
                time_codec_kwargs,
                batch_size=batch_size,
                seqlen=seqlen,
                indices=indices,
            )
            if position_ids.dim() == 3:
                forward_position_ids = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )
            else:
                forward_position_ids = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=forward_position_ids,
                **multi_modal_inputs,
                **forward_time_codec_kwargs,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=forward_position_ids,
                **multi_modal_inputs,
                **forward_time_codec_kwargs,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )

        is_timecodec_model = _is_timecodec_model(model)
        is_timeple_model = _is_timeple_model(model)
        is_timeed_model = _is_timeed_model(model)
        if is_timecodec_model:
            populated = _populate_timecodec_features_from_forward_details(
                sample_aux,
                time_loss_details=getattr(output, "time_loss_details", None),
                sample_idx=0,
            )
        elif is_timeple_model:
            populated = _populate_timeple_features_from_forward_details(
                sample_aux,
                timeple_loss_details=getattr(output, "timeple_loss_details", None),
                sample_idx=0,
            )
        elif is_timeple_model:
            populated = _populate_timeple_features_from_forward_details(
                sample_aux,
                timeple_loss_details=getattr(
                    output,
                    "timeple_loss_details",
                    None,
                ),
                sample_idx=0,
            )
        elif is_timeed_model:
            populated = _populate_timeed_features_from_forward_details(
                sample_aux,
                timeed_loss_details=getattr(output, "timeed_loss_details", None),
                sample_idx=0,
            )
        else:
            populated = _populate_cis_features_from_forward_details(
                sample_aux,
                cis_loss_details=getattr(output, "cis_loss_details", None),
                sample_idx=0,
            )
        if populated:
            if is_timecodec_model:
                sample_aux["timecodec_metrics_valid"][0] = 1.0
            elif is_timeple_model or is_timeple_model:
                sample_aux["timeple_metrics_valid"][0] = 1.0
            elif is_timeed_model:
                sample_aux["timeed_metrics_valid"][0] = 1.0
            else:
                sample_aux["cis_metrics_valid"][0] = 1.0

        positions_per_sample = _normalize_nested_sequences(forward_time_codec_kwargs.get("timespan_positions"))
        time_token_positions = [int(position) for position in positions_per_sample[0]] if len(positions_per_sample) > 0 else []
        if is_timecodec_model:
            sample_aux["timecodec_timespan_count"][0] = float(len(time_token_positions))
        elif is_timeple_model or is_timeple_model:
            sample_aux["timeple_timespan_count"][0] = float(len(time_token_positions))
        elif is_timeed_model:
            sample_aux["timeed_timespan_count"][0] = float(len(time_token_positions))
        else:
            sample_aux["cis_timespan_count"][0] = float(len(time_token_positions))

        decode_output = output
        decode_time_token_positions = time_token_positions
        if output.hidden_states is None and self.config.padding_free:
            decode_output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                **time_codec_kwargs,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )

        if decode_output is not output:
            decode_positions_per_sample = _normalize_nested_sequences(time_codec_kwargs.get("timespan_positions"))
            decode_time_token_positions = (
                [int(position) for position in decode_positions_per_sample[0]]
                if len(decode_positions_per_sample) > 0
                else []
            )

        if len(decode_time_token_positions) == 0:
            return (
                {key: float(value[0].item()) for key, value in sample_aux.items()},
                {key: value[0] for key, value in sample_non_tensors.items()},
            )

        hidden_states, decode_source = _resolve_timecodec_decode_hidden_states(
            output=decode_output,
            actor_module=self.actor_module,
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            multi_modal_inputs=multi_modal_inputs,
            time_codec_kwargs=time_codec_kwargs,
        )
        if hidden_states is None:
            return (
                {key: float(value[0].item()) for key, value in sample_aux.items()},
                {key: value[0] for key, value in sample_non_tensors.items()},
            )
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[0]

        decoded_predictions = _decode_timecodec_predictions_from_hidden_states(
            actor_module=self.actor_module,
            model=model,
            hidden_states=hidden_states,
            time_token_positions=decode_time_token_positions,
            video_durations=_resolve_decode_video_durations(
                target_durations=target_durations,
                target_ends=target_ends,
                prediction_count=len(decode_time_token_positions),
            ),
        )
        best_decoded_prediction = _select_best_decoded_prediction(
            decoded_predictions=decoded_predictions,
            target_starts=target_starts,
            target_ends=target_ends,
            prefix=(
                "timecodec"
                if is_timecodec_model
                else (
                    "timeple"
                    if is_timeple_model or is_timeple_model
                    else ("timeed" if is_timeed_model else "cis")
                )
            ),
        )
        if best_decoded_prediction is not None:
            for key, value in best_decoded_prediction.items():
                sample_aux[key][0] = value
            if is_timecodec_model:
                sample_aux["timecodec_decoded_valid"][0] = 1.0
                sample_aux["timecodec_decode_source"][0] = decode_source
            elif is_timeple_model or is_timeple_model:
                sample_non_tensors["timeple_decoded_segments"][0] = [
                    [float(pred_start), float(pred_end)] for pred_start, pred_end in decoded_predictions
                ]
                sample_aux["timeple_decoded_valid"][0] = 1.0
                sample_aux["timeple_decode_source"][0] = decode_source
            elif is_timeed_model:
                sample_non_tensors["timeed_decoded_segments"][0] = [
                    [float(pred_start), float(pred_end)] for pred_start, pred_end in decoded_predictions
                ]
                sample_aux["timeed_decoded_valid"][0] = 1.0
                sample_aux["timeed_decode_source"][0] = decode_source
            else:
                sample_non_tensors["cis_decoded_segments"][0] = [
                    [float(pred_start), float(pred_end)] for pred_start, pred_end in decoded_predictions
                ]
                sample_aux["cis_decoded_valid"][0] = 1.0
                sample_aux["cis_decode_source"][0] = decode_source

        return (
            {key: float(value[0].item()) for key, value in sample_aux.items()},
            {key: value[0] for key, value in sample_non_tensors.items()},
        )

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs", *TIME_CODEC_NON_TENSOR_KEYS]

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)

        return log_probs

    @torch.no_grad()
    def compute_log_prob_and_aux(
        self, data: DataProto
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, np.ndarray]]:
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
        non_tensor_select_keys = ["multi_modal_inputs", *TIME_CODEC_NON_TENSOR_KEYS]

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        log_probs_lst = []
        aux_tensors_lst: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
        aux_non_tensors_lst: defaultdict[str, list[Any]] = defaultdict(list)
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs, aux_tensors, aux_non_tensors = self._forward_micro_batch_with_aux(
                model_inputs, temperature=temperature
            )
            log_probs_lst.append(log_probs)
            for key, value in aux_tensors.items():
                aux_tensors_lst[key].append(value)
            for key, value in aux_non_tensors.items():
                aux_non_tensors_lst[key].extend(value)

        log_probs = torch.concat(log_probs_lst, dim=0)
        aux_tensors = {key: torch.concat(value, dim=0) for key, value in aux_tensors_lst.items()}
        aux_non_tensors = {key: _as_1d_object_array(value) for key, value in aux_non_tensors_lst.items()}

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            aux_tensors = {key: restore_dynamic_batch(value, batch_idx_list) for key, value in aux_tensors.items()}
            aux_non_tensors = {
                key: _as_1d_object_array(value[batch_idx_list].tolist()) for key, value in aux_non_tensors.items()
            }

        return log_probs, aux_tensors, aux_non_tensors

    @torch.no_grad()
    def compute_log_prob_and_timeed_span_ref(
        self, data: DataProto
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
        for key in ("timeed_span_valid",):
            if key in data.batch and key not in select_keys:
                select_keys.append(key)
        non_tensor_select_keys = ["multi_modal_inputs", *TIME_CODEC_NON_TENSOR_KEYS]

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        log_probs_lst = []
        span_ref_logits_lst = []
        span_ref_valid_lst = []
        csdo_ref_logits_lst = []
        use_timeed_span_ref = bool(getattr(self.config, "timeed_span_grpo_enabled", False))
        use_csdo_ref = bool(getattr(self.config, "csdo_enabled", False)) and bool(
            getattr(self.config, "csdo_use_ref_kl", True)
        )
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute ref log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            if use_timeed_span_ref:
                log_probs, span_logits, span_valid, _, _ = self._forward_micro_batch_with_timeed_span_log_probs(
                    model_inputs,
                    temperature=temperature,
                )
                if span_logits is not None:
                    span_ref_logits_lst.append(span_logits.detach())
                    span_ref_valid_lst.append(span_valid.detach())
            elif use_csdo_ref:
                log_probs, csdo_ref_logits, _, _ = self._forward_micro_batch_with_csdo(
                    model_inputs,
                    temperature=temperature,
                )
                csdo_ref_logits_lst.append(csdo_ref_logits.detach().float())
            else:
                log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        aux_tensors: dict[str, torch.Tensor] = {}
        if len(span_ref_logits_lst) > 0:
            aux_tensors["timeed_span_ref_logits"] = torch.concat(span_ref_logits_lst, dim=0)
        if len(span_ref_valid_lst) > 0:
            aux_tensors["timeed_span_ref_valid"] = torch.concat(span_ref_valid_lst, dim=0)
        if len(csdo_ref_logits_lst) > 0:
            aux_tensors["csdo_ref_logits"] = torch.concat(csdo_ref_logits_lst, dim=0)

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            aux_tensors = {key: restore_dynamic_batch(value, batch_idx_list) for key, value in aux_tensors.items()}

        return log_probs, aux_tensors

    def update_policy(self, data: DataProto) -> dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
        select_keys.extend(["old_log_probs", "ref_log_probs", "advantages"])
        for key in TIMEED_SPAN_POLICY_KEYS:
            if key in data.batch and key not in select_keys:
                select_keys.append(key)
        for key in CSDO_POLICY_KEYS:
            if key in data.batch and key not in select_keys:
                select_keys.append(key)
        for key in TR_SPD_POLICY_KEYS:
            if key in data.batch and key not in select_keys:
                select_keys.append(key)
        non_tensor_select_keys = ["multi_modal_inputs", *TIME_CODEC_NON_TENSOR_KEYS]

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                if self.config.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    use_timeed_span_grpo = (
                        bool(getattr(self.config, "timeed_span_grpo_enabled", False))
                        and "timeed_span_old_logits" in model_inputs
                        and "timeed_span_valid" in model_inputs
                    )
                    use_csdo = bool(getattr(self.config, "csdo_enabled", False))
                    use_tr_spd = bool(getattr(self.config, "tr_spd_enabled", False))
                    if use_csdo:
                        missing_espo_keys = [
                            key
                            for key in (
                                "csdo_old_logits",
                                "csdo_old_features",
                                "csdo_valid",
                            )
                            if key not in model_inputs
                        ]
                        if missing_espo_keys:
                            raise ValueError(
                                "CIS Counterfactual Span Distribution Optimization (CSDO) requires detached old-policy tensors from compute_log_prob_and_aux; "
                                f"missing: {missing_espo_keys}."
                            )
                    if use_tr_spd:
                        missing_spod_keys = [
                            key
                            for key in (
                                "tr_spd_old_logits",
                                "tr_spd_old_features",
                                "tr_spd_valid",
                            )
                            if key not in model_inputs
                        ]
                        if missing_spod_keys:
                            raise ValueError(
                                "CIS Trust-Region Span Posterior Distillation (TR-SPD) requires detached old-policy tensors from compute_log_prob_and_aux; "
                                f"missing: {missing_spod_keys}."
                            )

                    # all return: (bsz, response_length)
                    if use_csdo:
                        (
                            log_probs,
                            csdo_current_logits,
                            csdo_current_features,
                            csdo_current_valid,
                        ) = self._forward_micro_batch_with_csdo(
                            model_inputs,
                            temperature=temperature,
                        )
                        cis_aux_loss = torch.zeros((), device=log_probs.device, dtype=torch.float32)
                        cis_aux_metrics = {}
                        timeed_span_log_probs = None
                        timeed_span_logits = None
                        timeed_span_valid = None
                        timeed_aux_loss = torch.zeros((), device=log_probs.device, dtype=torch.float32)
                        timeed_aux_metrics = {}
                        tr_spd_current_logits = None
                        tr_spd_current_features = None
                        tr_spd_current_valid = None
                    elif use_tr_spd:
                        (
                            log_probs,
                            tr_spd_current_logits,
                            tr_spd_current_features,
                            tr_spd_current_valid,
                        ) = self._forward_micro_batch_with_tr_spd(
                            model_inputs,
                            temperature=temperature,
                        )
                        cis_aux_loss = torch.zeros((), device=log_probs.device, dtype=torch.float32)
                        cis_aux_metrics = {}
                        timeed_span_log_probs = None
                        timeed_span_logits = None
                        timeed_span_valid = None
                        timeed_aux_loss = torch.zeros((), device=log_probs.device, dtype=torch.float32)
                        timeed_aux_metrics = {}
                        csdo_current_logits = None
                        csdo_current_features = None
                        csdo_current_valid = None
                    elif use_timeed_span_grpo:
                        (
                            log_probs,
                            timeed_span_logits,
                            timeed_span_valid,
                            timeed_aux_loss,
                            timeed_aux_metrics,
                        ) = self._forward_micro_batch_with_timeed_span_log_probs(
                            model_inputs,
                            temperature=temperature,
                        )
                        cis_aux_loss = torch.zeros((), device=log_probs.device, dtype=torch.float32)
                        cis_aux_metrics = {}
                        csdo_current_logits = None
                        csdo_current_features = None
                        csdo_current_valid = None
                        tr_spd_current_logits = None
                        tr_spd_current_features = None
                        tr_spd_current_valid = None
                    elif self.config.cis_aux_loss_weight > 0.0:
                        log_probs, cis_aux_loss, cis_aux_metrics = self._forward_micro_batch_with_cis_aux_loss(
                            model_inputs,
                            temperature=temperature,
                        )
                        timeed_span_log_probs = None
                        timeed_span_logits = None
                        timeed_span_valid = None
                        csdo_current_logits = None
                        csdo_current_features = None
                        csdo_current_valid = None
                        tr_spd_current_logits = None
                        tr_spd_current_features = None
                        tr_spd_current_valid = None
                        timeed_aux_loss = torch.zeros((), device=log_probs.device, dtype=torch.float32)
                        timeed_aux_metrics = {}
                    else:
                        log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
                        cis_aux_loss = torch.zeros((), device=log_probs.device, dtype=torch.float32)
                        cis_aux_metrics = {}
                        timeed_span_log_probs = None
                        timeed_span_logits = None
                        timeed_span_valid = None
                        csdo_current_logits = None
                        csdo_current_features = None
                        csdo_current_valid = None
                        tr_spd_current_logits = None
                        tr_spd_current_features = None
                        tr_spd_current_valid = None
                        timeed_aux_loss = torch.zeros((), device=log_probs.device, dtype=torch.float32)
                        timeed_aux_metrics = {}

                    pg_loss, pg_metrics = compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        tau_positive=self.config.tau_positive,
                        tau_negative=self.config.tau_negative,
                        loss_type=self.config.loss_type,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )
                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        # compute kl loss
                        kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)
                        loss = pg_loss + kl_loss * self.config.kl_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef
                    else:
                        loss = pg_loss
                    if use_tr_spd:
                        text_pg_weight = float(getattr(self.config, "tr_spd_text_pg_loss_weight", 0.0))
                        if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                            loss = pg_loss * text_pg_weight + kl_loss * self.config.kl_coef
                        else:
                            loss = pg_loss * text_pg_weight

                    exact_span_grpo_loss = torch.zeros((), device=log_probs.device, dtype=torch.float32)
                    exact_span_grpo_metrics = {}
                    span_kl_loss = torch.zeros((), device=log_probs.device, dtype=torch.float32)
                    span_kl_metrics = {}
                    span_pref_loss = torch.zeros((), device=log_probs.device, dtype=torch.float32)
                    span_pref_metrics = {}
                    csdo_loss = torch.zeros((), device=log_probs.device, dtype=torch.float32)
                    csdo_kl_loss = torch.zeros((), device=log_probs.device, dtype=torch.float32)
                    csdo_metrics = {}
                    tr_spd_loss = torch.zeros((), device=log_probs.device, dtype=torch.float32)
                    tr_spd_metrics = {}
                    if use_timeed_span_grpo:
                        span_loss_mask = model_inputs.get("timeed_span_valid", timeed_span_valid)
                        if "timeed_text_response_rewards" in model_inputs:
                            text_format_mask = (
                                model_inputs["timeed_text_response_rewards"]
                                .float()
                                .reshape(-1)
                                .to(device=log_probs.device)
                                > 0
                            ).float()
                            span_loss_mask = span_loss_mask.float().reshape(-1).to(device=log_probs.device) * text_format_mask
                        timeed_reward_maps, timeed_reward_valid, timeed_reward_metrics = _compute_timeed_cell_reward_maps(
                            model=_unwrap_model(self.actor_module),
                            span_logits=timeed_span_logits,
                            micro_batch=model_inputs,
                            sample_mask=span_loss_mask,
                            iou_weight=float(getattr(self.config, "timeed_span_reward_iou_weight", 0.8)),
                            boundary_weight=float(getattr(self.config, "timeed_span_reward_boundary_weight", 0.2)),
                            boundary_tau=float(getattr(self.config, "timeed_span_reward_boundary_tau", 10.0)),
                        )
                        exact_span_grpo_loss, exact_span_grpo_metrics = _compute_exact_span_grpo_loss(
                            old_logits=model_inputs["timeed_span_old_logits"],
                            current_logits=timeed_span_logits,
                            reward_maps=timeed_reward_maps,
                            sample_mask=timeed_reward_valid,
                            clip_ratio_low=self.config.clip_ratio_low,
                            clip_ratio_high=self.config.clip_ratio_high,
                            advantage_eps=float(getattr(self.config, "timeed_span_advantage_eps", 1e-6)),
                        )
                        loss = loss + float(self.config.timeed_span_loss_weight) * exact_span_grpo_loss
                        if "timeed_span_ref_logits" in model_inputs and float(self.config.timeed_span_kl_coef) > 0.0:
                            span_kl_loss, span_kl_metrics = _compute_exact_span_kl_loss(
                                current_logits=timeed_span_logits,
                                ref_logits=model_inputs["timeed_span_ref_logits"],
                                sample_mask=timeed_reward_valid,
                            )
                            loss = loss + float(self.config.timeed_span_kl_coef) * span_kl_loss
                        if (
                            "timeed_span_ref_logits" in model_inputs
                            and float(self.config.timeed_span_pref_loss_weight) > 0.0
                        ):
                            span_pref_loss, span_pref_metrics = _compute_reward_map_span_preference_loss(
                                current_logits=timeed_span_logits,
                                ref_logits=model_inputs["timeed_span_ref_logits"],
                                old_logits=model_inputs["timeed_span_old_logits"],
                                reward_maps=timeed_reward_maps,
                                sample_mask=timeed_reward_valid,
                                beta=float(self.config.timeed_span_pref_beta),
                                reward_gap_delta=float(getattr(self.config, "timeed_span_pref_delta", 0.05)),
                                negative_reward_threshold=float(
                                    getattr(self.config, "timeed_span_pref_negative_reward_threshold", 0.2)
                                ),
                            )
                            loss = loss + float(self.config.timeed_span_pref_loss_weight) * span_pref_loss
                        if float(self.config.timeed_span_aux_loss_weight) > 0.0:
                            loss = loss + float(self.config.timeed_span_aux_loss_weight) * timeed_aux_loss
                    else:
                        timeed_reward_metrics = {}

                    if use_csdo:
                        csdo_old_valid = (
                            model_inputs["csdo_valid"].float().reshape(-1).to(device=log_probs.device) > 0
                        )
                        csdo_current_valid_mask = (
                            csdo_current_valid.float().reshape(-1).to(device=log_probs.device) > 0
                        )
                        if not torch.equal(csdo_old_valid, csdo_current_valid_mask):
                            raise ValueError(
                                "CIS Counterfactual Span Distribution Optimization (CSDO) old/current valid mask mismatch; old logits/features must match the "
                                "same generated <|TIMESPAN|> samples used by the current actor update."
                            )
                        csdo_sample_mask = csdo_old_valid.float()
                        csdo_param_context = (
                            _codec_param_context(self.actor_module)
                            if isinstance(self.actor_module, FSDP)
                            else nullcontext()
                        )
                        with csdo_param_context:
                            csdo_loss, csdo_kl_loss, csdo_metrics = _compute_csdo_loss(
                                current_logits=csdo_current_logits,
                                current_features=csdo_current_features,
                                old_logits=model_inputs["csdo_old_logits"],
                                old_features=model_inputs["csdo_old_features"],
                                ref_logits=model_inputs.get("csdo_ref_logits"),
                                micro_batch=model_inputs,
                                sample_mask=csdo_sample_mask,
                                codec=getattr(_unwrap_model(self.actor_module), "timeple_codec"),
                                eta=float(getattr(self.config, "csdo_eta", 0.05)),
                                tau=float(getattr(self.config, "csdo_tau", 0.5)),
                                adv_norm=bool(getattr(self.config, "csdo_adv_norm", True)),
                                adv_clip=float(getattr(self.config, "csdo_adv_clip", 5.0)),
                                min_adv_std=float(getattr(self.config, "csdo_min_adv_std", 1.0e-4)),
                                reward_type=str(getattr(self.config, "csdo_reward_type", "iou_boundary")),
                                boundary_weight=float(getattr(self.config, "csdo_boundary_weight", 0.1)),
                                use_ref_kl=bool(getattr(self.config, "csdo_use_ref_kl", True)),
                            )
                        loss = loss + float(self.config.csdo_loss_weight) * csdo_loss
                        loss = loss + float(self.config.csdo_span_kl_coef) * csdo_kl_loss

                    if use_tr_spd:
                        tr_spd_old_valid = (
                            model_inputs["tr_spd_valid"].float().reshape(-1).to(device=log_probs.device) > 0
                        )
                        tr_spd_current_valid_mask = (
                            tr_spd_current_valid.float().reshape(-1).to(device=log_probs.device) > 0
                        )
                        if not torch.equal(tr_spd_old_valid, tr_spd_current_valid_mask):
                            raise ValueError(
                                "CIS Trust-Region Span Posterior Distillation (TR-SPD) old/current valid mask mismatch; old logits/features must match the "
                                "same generated <|TIMESPAN|> samples used by the current actor update."
                            )
                        tr_spd_sample_mask = tr_spd_old_valid.float()
                        if "tr_spd_text_response_rewards" in model_inputs:
                            tr_spd_text_mask = (
                                model_inputs["tr_spd_text_response_rewards"]
                                .float()
                                .reshape(-1)
                                .to(device=log_probs.device)
                                > 0
                            ).float()
                            tr_spd_sample_mask = tr_spd_sample_mask * tr_spd_text_mask
                        tr_spd_param_context = (
                            _codec_param_context(self.actor_module)
                            if isinstance(self.actor_module, FSDP)
                            else nullcontext()
                        )
                        with tr_spd_param_context:
                            tr_spd_loss, tr_spd_metrics = compute_tr_spd_loss(
                                current_logits=tr_spd_current_logits,
                                current_features=tr_spd_current_features,
                                old_logits=model_inputs["tr_spd_old_logits"],
                                old_features=model_inputs["tr_spd_old_features"],
                                micro_batch=model_inputs,
                                sample_mask=tr_spd_sample_mask,
                                codec=getattr(_unwrap_model(self.actor_module), "timeple_codec"),
                                tau_candidates=getattr(self.config, "tr_spd_tau_candidates", (1.0, 1.5, 2.0)),
                                support_mode=str(getattr(self.config, "tr_spd_support_mode", "none")),
                                accept_delta=float(getattr(self.config, "tr_spd_accept_delta", 0.0)),
                                trust_region_kl_budget=getattr(
                                    self.config,
                                    "tr_spd_trust_region_kl_budget",
                                    None,
                                ),
                                rejected_retention_weight=float(
                                    getattr(self.config, "tr_spd_rejected_retention_weight", 0.1)
                                ),
                                use_improvement_weight=bool(
                                    getattr(self.config, "tr_spd_use_improvement_weight", True)
                                ),
                                improvement_gamma=float(
                                    getattr(self.config, "tr_spd_improvement_gamma", 0.5)
                                ),
                                improvement_scale=float(
                                    getattr(self.config, "tr_spd_improvement_scale", 0.05)
                                ),
                                improvement_max_extra_weight=float(
                                    getattr(self.config, "tr_spd_improvement_max_extra_weight", 2.0)
                                ),
                                reward_type=str(getattr(self.config, "tr_spd_reward_type", "iou")),
                                boundary_weight=float(getattr(self.config, "tr_spd_boundary_weight", 0.0)),
                            )
                        loss = loss + float(self.config.tr_spd_loss_weight) * tr_spd_loss

                    if self.config.cis_aux_loss_weight > 0.0:
                        loss = loss + float(self.config.cis_aux_loss_weight) * cis_aux_loss

                    loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                    loss.backward()

                    batch_metrics = {f"actor/{k}": v for k, v in pg_metrics.items()}
                    batch_metrics["actor/pg_loss"] = pg_loss.detach().item()
                    if use_timeed_span_grpo:
                        batch_metrics["actor/timeed_exact_span_grpo_loss"] = exact_span_grpo_loss.detach().item()
                        batch_metrics["actor/timeed_span_pg_loss"] = exact_span_grpo_loss.detach().item()
                        batch_metrics["actor/timeed_exact_span_grpo_loss_weight"] = float(
                            self.config.timeed_span_loss_weight
                        )
                        batch_metrics["actor/timeed_span_loss_weight"] = float(self.config.timeed_span_loss_weight)
                        batch_metrics["actor/timeed_exact_span_grpo_loss_weighted"] = (
                            float(self.config.timeed_span_loss_weight) * exact_span_grpo_loss.detach().item()
                        )
                        batch_metrics["actor/timeed_span_loss_weighted"] = (
                            float(self.config.timeed_span_loss_weight) * exact_span_grpo_loss.detach().item()
                        )
                        batch_metrics["actor/timeed_span_kl_loss"] = span_kl_loss.detach().item()
                        batch_metrics["actor/timeed_span_kl_coef"] = float(self.config.timeed_span_kl_coef)
                        batch_metrics["actor/timeed_span_kl_loss_weighted"] = (
                            float(self.config.timeed_span_kl_coef) * span_kl_loss.detach().item()
                        )
                        batch_metrics["actor/timeed_span_pref_loss"] = span_pref_loss.detach().item()
                        batch_metrics["actor/timeed_span_pref_loss_weight"] = float(
                            self.config.timeed_span_pref_loss_weight
                        )
                        batch_metrics["actor/timeed_span_pref_loss_weighted"] = (
                            float(self.config.timeed_span_pref_loss_weight) * span_pref_loss.detach().item()
                        )
                        batch_metrics["actor/timeed_span_pref_beta"] = float(self.config.timeed_span_pref_beta)
                        batch_metrics["actor/timeed_span_aux_loss_weight"] = float(
                            self.config.timeed_span_aux_loss_weight
                        )
                        batch_metrics["actor/timeed_span_aux_loss_weighted"] = (
                            float(self.config.timeed_span_aux_loss_weight) * timeed_aux_loss.detach().item()
                        )
                        for key, value in exact_span_grpo_metrics.items():
                            batch_metrics[f"actor/{key}"] = value
                        for key, value in timeed_reward_metrics.items():
                            batch_metrics[f"actor/{key}"] = value
                        for key, value in span_kl_metrics.items():
                            batch_metrics[f"actor/{key}"] = value
                        for key, value in span_pref_metrics.items():
                            batch_metrics[f"actor/{key}"] = value
                        for key, value in timeed_aux_metrics.items():
                            batch_metrics[f"actor/{key}"] = value
                    if use_csdo:
                        batch_metrics["actor/csdo_loss"] = csdo_loss.detach().item()
                        batch_metrics["actor/csdo_loss_weight"] = float(
                            self.config.csdo_loss_weight
                        )
                        batch_metrics["actor/csdo_loss_weighted"] = (
                            float(self.config.csdo_loss_weight) * csdo_loss.detach().item()
                        )
                        batch_metrics["actor/csdo_span_kl"] = csdo_kl_loss.detach().item()
                        batch_metrics["actor/csdo_span_kl_coef"] = float(
                            self.config.csdo_span_kl_coef
                        )
                        batch_metrics["actor/csdo_span_kl_weighted"] = (
                            float(self.config.csdo_span_kl_coef) * csdo_kl_loss.detach().item()
                        )
                        for key, value in csdo_metrics.items():
                            batch_metrics[f"actor/{key}"] = value
                    if use_tr_spd:
                        batch_metrics["actor/tr_spd_loss"] = tr_spd_loss.detach().item()
                        batch_metrics["actor/tr_spd_loss_weight"] = float(
                            self.config.tr_spd_loss_weight
                        )
                        batch_metrics["actor/tr_spd_loss_weighted"] = (
                            float(self.config.tr_spd_loss_weight) * tr_spd_loss.detach().item()
                        )
                        batch_metrics["actor/tr_spd_text_pg_loss_weight"] = float(
                            getattr(self.config, "tr_spd_text_pg_loss_weight", 0.0)
                        )
                        for key, value in tr_spd_metrics.items():
                            batch_metrics[f"actor/{key}"] = value
                    if self.config.cis_aux_loss_weight > 0.0:
                        batch_metrics["actor/cis_aux_loss_weight"] = float(self.config.cis_aux_loss_weight)
                        batch_metrics["actor/cis_aux_loss_weighted"] = (
                            float(self.config.cis_aux_loss_weight) * cis_aux_loss.detach().item()
                        )
                        for key, value in cis_aux_metrics.items():
                            batch_metrics[f"actor/{key}"] = value
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
