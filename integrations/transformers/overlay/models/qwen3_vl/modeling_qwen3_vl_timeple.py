# coding=utf-8
"""Qwen3-VL model with span-only TimePLE integration."""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from ...modeling_outputs import ModelOutput
from ...utils import logging

from timeple.models import TimePLEInterfaceAdapter, TimePLECodec

from .configuration_qwen3_vl_timeple import Qwen3VLTimePLEConfig
from .modeling_qwen3_vl import Qwen3VLForConditionalGeneration

logger = logging.get_logger(__name__)


@dataclass
class Qwen3VLOutputWithTimePLECodec(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    timeple_loss: Optional[torch.FloatTensor] = None
    timeple_predictions: Optional[Dict[str, List[float]]] = None
    timeple_loss_details: Optional[Dict[str, torch.FloatTensor]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    csdo_logits: Optional[torch.FloatTensor] = None
    csdo_features: Optional[torch.FloatTensor] = None
    csdo_valid: Optional[torch.FloatTensor] = None
    tr_spd_logits: Optional[torch.FloatTensor] = None
    tr_spd_features: Optional[torch.FloatTensor] = None
    tr_spd_valid: Optional[torch.FloatTensor] = None


class Qwen3VLForConditionalGenerationWithTimePLECodec(Qwen3VLForConditionalGeneration):
    config_class = Qwen3VLTimePLEConfig

    def __init__(self, config: Qwen3VLTimePLEConfig):
        super().__init__(config)

        self.use_timeple_codec = config.use_timeple_codec
        self.use_timeple_interface_adapter = bool(getattr(config, "use_timeple_interface_adapter", False))
        self.default_video_duration_sec = float(getattr(config, "default_video_duration_sec", 1.0))

        if self.use_timeple_codec:
            self.timeple_codec = TimePLECodec(config.timeple_codec_config)
            self.timestamp_token_id = config.timestamp_token_id
            self.timespan_token_id = config.timespan_token_id

            codec_dim = config.get_timeple_codec_output_dim()
            hidden_size = config.text_config.hidden_size
            if codec_dim != hidden_size:
                raise ValueError(
                    f"TimePLE output dimension ({codec_dim}) must match model hidden size ({hidden_size})."
                )
            if self.use_timeple_interface_adapter:
                self.timeple_interface_adapter = TimePLEInterfaceAdapter(
                    hidden_size,
                    config=getattr(config, "timeple_interface_adapter", None),
                )

        if config.freeze_vision:
            for param in self._get_vision_module().parameters():
                param.requires_grad = False

        if config.freeze_language:
            for param in self._get_language_module().parameters():
                param.requires_grad = False

    def _get_vision_module(self):
        if hasattr(self, "visual"):
            return self.visual
        if hasattr(self, "model") and hasattr(self.model, "visual"):
            return self.model.visual
        raise AttributeError("Unable to locate Qwen3-VL vision module for freeze_vision.")

    def _get_language_module(self):
        if hasattr(self, "model") and hasattr(self.model, "language_model"):
            return self.model.language_model
        if hasattr(self, "language_model"):
            return self.language_model
        raise AttributeError("Unable to locate Qwen3-VL language module for freeze_language.")

    def _prepare_duration_tensor(
        self,
        sample_duration,
        target_count: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        durations = torch.as_tensor(sample_duration, device=device, dtype=dtype)
        if durations.ndim == 0:
            return durations.repeat(target_count)
        durations = durations.reshape(-1)
        if durations.numel() == 1:
            return durations.repeat(target_count)
        if durations.numel() != target_count:
            return durations[:1].repeat(target_count)
        return durations

    def _resolve_sample_durations(
        self,
        labels: Optional[Dict[str, List[List[float]]]],
        provided_durations: Optional[List[List[float]]],
        batch_idx: int,
        target_count: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        sample_duration = None
        if provided_durations is not None and batch_idx < len(provided_durations):
            sample_duration = provided_durations[batch_idx]
        elif labels is not None and "video_duration" in labels and batch_idx < len(labels["video_duration"]):
            sample_duration = labels["video_duration"][batch_idx]

        if sample_duration is None:
            fallback = self.default_video_duration_sec
            if labels is not None and batch_idx < len(labels.get("end", [])) and labels["end"][batch_idx]:
                fallback = max(float(max(labels["end"][batch_idx])), self.default_video_duration_sec)
            sample_duration = fallback

        return self._prepare_duration_tensor(
            sample_duration,
            target_count,
            device=device,
            dtype=dtype,
        ).clamp_min(1e-6)

    def _get_adapter_regularization_weights(self) -> Dict[str, float]:
        if not self.use_timeple_interface_adapter or not hasattr(self, "timeple_interface_adapter"):
            return {
                "base_decode_aux_weight": 0.0,
                "input_residual_norm_weight": 0.0,
                "output_residual_norm_weight": 0.0,
                "output_bridge_weight": 0.0,
            }
        return self.timeple_interface_adapter.get_regularization_weights()

    def _compute_timeple_policy_outputs(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_timeple_codec:
            raise RuntimeError("TimePLE is not enabled.")
        decoder = self.timeple_codec.decoder
        span_head = decoder.span_head
        batch_size, seq_len = hidden_states.shape[:2]
        num_u_bins = int(decoder.num_u_bins)
        num_v_bins = int(decoder.num_v_bins)
        feature_dim = int(span_head.in_features)
        trunk_dtype = next(span_head.parameters()).dtype

        logits = hidden_states.new_zeros((batch_size, num_u_bins, num_v_bins), dtype=trunk_dtype)
        features = hidden_states.new_zeros((batch_size, feature_dim), dtype=trunk_dtype)
        positions = positions.reshape(-1).to(device=hidden_states.device, dtype=torch.long)
        if positions.numel() != batch_size:
            raise ValueError(f"positions must have shape [B], got {tuple(positions.shape)}.")

        valid = (positions >= 0) & (positions < seq_len)
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            return logits, features, valid.float()

        timespan_hidden = hidden_states[valid_indices, positions[valid_indices], :]
        if self.use_timeple_interface_adapter:
            timespan_embedding = self.timeple_interface_adapter.forward_output(
                timespan_hidden,
                compute_diagnostics=False,
            ).adapted
        else:
            timespan_embedding = timespan_hidden

        valid_features = decoder.trunk(timespan_embedding.to(trunk_dtype))
        valid_logits = span_head(valid_features).view(-1, num_u_bins, num_v_bins)
        features = features.index_copy(0, valid_indices, valid_features)
        logits = logits.index_copy(0, valid_indices, valid_logits)
        return logits, features, valid.float()

    def _compute_csdo_outputs(
        self,
        hidden_states: torch.Tensor,
        csdo_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._compute_timeple_policy_outputs(hidden_states, csdo_positions)

    def _compute_tr_spd_outputs(
        self,
        hidden_states: torch.Tensor,
        tr_spd_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._compute_timeple_policy_outputs(hidden_states, tr_spd_positions)

    def _reduce_codec_loss_slice(
        self,
        losses: Dict[str, torch.Tensor],
        start_idx: int,
        end_idx: int,
    ) -> Dict[str, torch.Tensor]:
        reduced: Dict[str, torch.Tensor] = {}

        def take(name: str):
            value = losses.get(name)
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return value[start_idx:end_idx]
            return value

        def mean_value(name: str):
            value = take(name)
            if not isinstance(value, torch.Tensor):
                return value
            return value.mean() if value.ndim > 0 else value

        for name in (
            "total_loss",
            "dfl_loss",
            "iou_loss",
            "interval_l1",
            "boundary_loss",
            "boundary_weight",
            "span_giou_loss",
            "span_iou",
            "mae_start",
            "mae_end",
            "mae_total",
        ):
            if name in losses:
                reduced[name] = mean_value(name)
        for name in ("pred_start_sec", "pred_end_sec"):
            if name in losses:
                reduced[name] = take(name)

        return reduced

    @staticmethod
    def _validate_batch_time_inputs(
        name: str,
        batch_size: int,
        labels: Dict[str, List[List[float]]],
        positions: List[List[int]],
    ) -> None:
        start_list = labels.get("start", [])
        end_list = labels.get("end", [])
        if len(start_list) != batch_size or len(end_list) != batch_size or len(positions) != batch_size:
            raise ValueError(
                f"{name} batch alignment mismatch: "
                f"batch_size={batch_size}, start={len(start_list)}, end={len(end_list)}, positions={len(positions)}"
            )

    @staticmethod
    def _validate_sample_time_alignment(
        name: str,
        batch_idx: int,
        start_times: torch.Tensor,
        end_times: torch.Tensor,
        positions: List[int],
    ) -> None:
        label_count = start_times.numel()
        if end_times.numel() != label_count:
            raise ValueError(
                f"{name} start/end count mismatch at batch_idx={batch_idx}: "
                f"start={label_count}, end={end_times.numel()}"
            )
        if len(positions) != label_count:
            raise ValueError(
                f"{name} token/label count mismatch at batch_idx={batch_idx}: "
                f"positions={len(positions)}, labels={label_count}"
            )

    def _collect_flat_time_batch(
        self,
        name: str,
        batch_size: int,
        labels: Dict[str, List[List[float]]],
        positions_list: List[List[int]],
        provided_durations: Optional[List[List[float]]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Dict[str, object]]:
        self._validate_batch_time_inputs(name, batch_size, labels, positions_list)

        flat_start_times: List[torch.Tensor] = []
        flat_end_times: List[torch.Tensor] = []
        flat_durations: List[torch.Tensor] = []
        flat_batch_indices: List[int] = []
        flat_positions: List[int] = []
        sample_slices: Dict[int, Tuple[int, int]] = {}
        offset = 0

        for batch_idx in range(batch_size):
            start_times = torch.as_tensor(labels["start"][batch_idx], device=device, dtype=dtype)
            end_times = torch.as_tensor(labels["end"][batch_idx], device=device, dtype=dtype)
            positions = positions_list[batch_idx]
            self._validate_sample_time_alignment(name, batch_idx, start_times, end_times, positions)
            if start_times.numel() == 0:
                continue

            count = start_times.numel()
            durations = self._resolve_sample_durations(
                labels,
                provided_durations,
                batch_idx,
                count,
                device=device,
                dtype=dtype,
            )
            flat_start_times.append(start_times)
            flat_end_times.append(end_times)
            flat_durations.append(durations)
            flat_batch_indices.extend([batch_idx] * count)
            flat_positions.extend(positions)
            sample_slices[batch_idx] = (offset, offset + count)
            offset += count

        if not flat_start_times:
            return None

        return {
            "start_times": torch.cat(flat_start_times, dim=0),
            "end_times": torch.cat(flat_end_times, dim=0),
            "durations": torch.cat(flat_durations, dim=0),
            "batch_indices": torch.tensor(flat_batch_indices, device=device, dtype=torch.long),
            "positions": torch.tensor(flat_positions, device=device, dtype=torch.long),
            "sample_slices": sample_slices,
        }

    def _prepare_timestamp_runtime(
        self,
        inputs_embeds: torch.FloatTensor,
        timestamp_labels: Dict[str, List[List[float]]],
        timestamp_positions: List[List[int]],
        timestamp_video_durations: Optional[List[List[float]]],
    ) -> Optional[Dict[str, object]]:
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device
        flat_timestamp_batch = self._collect_flat_time_batch(
            "timestamp",
            batch_size,
            timestamp_labels,
            timestamp_positions,
            timestamp_video_durations,
            device=device,
            dtype=torch.float32,
        )
        if flat_timestamp_batch is None:
            return None

        batch_indices = flat_timestamp_batch["batch_indices"]
        positions = flat_timestamp_batch["positions"]
        timestamp_embeddings = self.timeple_codec.encode(
            flat_timestamp_batch["start_times"],
            flat_timestamp_batch["end_times"],
            flat_timestamp_batch["durations"],
        ).to(dtype=inputs_embeds.dtype)

        input_adapter_penalties: Dict[int, torch.Tensor] = {}
        input_adapter_ratios: List[torch.Tensor] = []
        if self.use_timeple_interface_adapter:
            anchor_embedding = inputs_embeds[batch_indices, positions, :].detach()
            input_adapter_output = self.timeple_interface_adapter.forward_input(
                timestamp_embeddings,
                anchor_embedding=anchor_embedding,
            )
            timestamp_embeddings = input_adapter_output.adapted.to(dtype=inputs_embeds.dtype)
            per_token_penalties = input_adapter_output.residual_penalty_per_token.float()
            per_token_ratios = input_adapter_output.residual_ratio_per_token.float()
            for batch_idx, (start_idx, end_idx) in flat_timestamp_batch["sample_slices"].items():
                input_adapter_penalties[batch_idx] = per_token_penalties[start_idx:end_idx].mean()
                input_adapter_ratios.append(per_token_ratios[start_idx:end_idx].mean())

        return {
            "batch_indices": batch_indices,
            "positions": positions,
            "timestamp_embeddings": timestamp_embeddings,
            "input_adapter_penalties": input_adapter_penalties,
            "input_adapter_ratios": input_adapter_ratios,
        }

    @staticmethod
    def _past_key_values_length(past_key_values: Optional[List[torch.FloatTensor]]) -> int:
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "get_seq_length"):
            return int(past_key_values.get_seq_length())
        try:
            first_layer = past_key_values[0]
            first_key = first_layer[0]
            return int(first_key.shape[-2])
        except Exception:
            return 0

    @staticmethod
    def _positions_fit_current_sequence(timestamp_positions: List[List[int]], seq_len: int) -> bool:
        for sample_positions in timestamp_positions:
            for position in sample_positions:
                if int(position) < 0 or int(position) >= seq_len:
                    return False
        return True

    @staticmethod
    def _apply_timestamp_runtime_to_inputs_embeds(
        inputs_embeds: torch.FloatTensor,
        timestamp_runtime: Dict[str, object],
    ) -> torch.FloatTensor:
        patched_inputs_embeds = inputs_embeds.clone()
        batch_indices = timestamp_runtime["batch_indices"]
        positions = timestamp_runtime["positions"]
        timestamp_embeddings = timestamp_runtime["timestamp_embeddings"].to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        patched_inputs_embeds[batch_indices, positions, :] = timestamp_embeddings
        return patched_inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        timestamp_labels: Optional[Dict[str, List[List[float]]]] = None,
        timestamp_positions: Optional[List[List[int]]] = None,
        timestamp_video_durations: Optional[List[List[float]]] = None,
        timespan_labels: Optional[Dict[str, List[List[float]]]] = None,
        timespan_positions: Optional[List[List[int]]] = None,
        timespan_video_durations: Optional[List[List[float]]] = None,
        compute_timeple_loss: bool = True,
        timeple_loss_weight: float = 1.0,
        timeple_decode_loss_weight: float = 1.0,
        timeple_dfl_loss_weight: Optional[float] = None,
        timeple_iou_loss_weight: Optional[float] = None,
        timeple_boundary_loss_weight: Optional[float] = None,
        timeple_codec_recon_loss_weight: float = 0.0,
        timeple_embedding_loss_weight: float = 0.0,
        timeple_embedding_cosine_loss_weight: float = 0.0,
        timeple_reencoding_loss_weight: float = 0.0,
        compute_csdo_outputs: bool = False,
        csdo_positions: Optional[torch.LongTensor] = None,
        compute_tr_spd_outputs: bool = False,
        tr_spd_positions: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen3VLOutputWithTimePLECodec]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is None and inputs_embeds is None:
            raise ValueError("TimePLE forward requires input_ids or inputs_embeds.")

        input_adapter_penalties: Dict[int, torch.Tensor] = {}
        input_adapter_ratios: List[torch.Tensor] = []
        timeple_timestamp_runtime = None
        past_key_values_length = self._past_key_values_length(past_key_values)
        should_apply_timestamp_runtime = (
            self.use_timeple_codec
            and timestamp_labels is not None
            and timestamp_positions is not None
            and past_key_values_length == 0
        )
        if should_apply_timestamp_runtime:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            if self._positions_fit_current_sequence(timestamp_positions, inputs_embeds.shape[1]):
                timeple_timestamp_runtime = self._prepare_timestamp_runtime(
                    inputs_embeds,
                    timestamp_labels,
                    timestamp_positions,
                    timestamp_video_durations,
                )
                if timeple_timestamp_runtime is not None:
                    input_adapter_penalties = timeple_timestamp_runtime["input_adapter_penalties"]
                    input_adapter_ratios = timeple_timestamp_runtime["input_adapter_ratios"]

        model_output_hidden_states = False if output_hidden_states is None else output_hidden_states
        model_output_attentions = False if output_attentions is None else output_attentions

        model_input_ids = input_ids
        model_inputs_embeds = inputs_embeds
        model_kwargs = dict(kwargs)
        if timeple_timestamp_runtime is not None:
            if hasattr(self.model, "origin_forward"):
                model_kwargs["timeple_timestamp_runtime"] = timeple_timestamp_runtime
            else:
                model_inputs_embeds = self._apply_timestamp_runtime_to_inputs_embeds(
                    inputs_embeds,
                    timeple_timestamp_runtime,
                )
                if position_ids is None and hasattr(self.model, "compute_3d_position_ids"):
                    position_ids = self.model.compute_3d_position_ids(
                        input_ids=input_ids,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        inputs_embeds=model_inputs_embeds,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        mm_token_type_ids=mm_token_type_ids,
                    )
                model_input_ids = None

        outputs = self.model(
            input_ids=model_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=model_inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            cache_position=cache_position,
            output_attentions=model_output_attentions,
            output_hidden_states=model_output_hidden_states,
            return_dict=True,
            **model_kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        lm_loss = None
        if labels is not None:
            lm_loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
            )

        timeple_loss = None
        timeple_loss_details = None
        timeple_predictions = None
        csdo_logits = None
        csdo_features = None
        csdo_valid = None
        tr_spd_logits = None
        tr_spd_features = None
        tr_spd_valid = None

        if compute_csdo_outputs:
            if csdo_positions is None:
                raise ValueError("compute_csdo_outputs=True requires csdo_positions.")
            csdo_logits, csdo_features, csdo_valid = self._compute_csdo_outputs(
                hidden_states=hidden_states,
                csdo_positions=csdo_positions,
            )
        if compute_tr_spd_outputs:
            if tr_spd_positions is None:
                raise ValueError("compute_tr_spd_outputs=True requires tr_spd_positions.")
            tr_spd_logits, tr_spd_features, tr_spd_valid = self._compute_tr_spd_outputs(
                hidden_states=hidden_states,
                tr_spd_positions=tr_spd_positions,
            )

        if self.use_timeple_codec and compute_timeple_loss and timespan_labels is not None and timespan_positions is not None:
            timeple_loss_details = {}
            all_timeple_losses = []
            # `outputs[0]` is already the final backbone hidden state. Reusing it here
            # avoids forcing the model to materialize all intermediate hidden states.
            last_hidden_states = hidden_states
            batch_size = last_hidden_states.shape[0]
            self._validate_batch_time_inputs("timespan", batch_size, timespan_labels, timespan_positions)
            flat_timespan_batch = self._collect_flat_time_batch(
                "timespan",
                batch_size,
                timespan_labels,
                timespan_positions,
                timespan_video_durations,
                device=last_hidden_states.device,
                dtype=torch.float32,
            )

            if flat_timespan_batch is not None:
                flat_batch_indices = flat_timespan_batch["batch_indices"]
                flat_positions = flat_timespan_batch["positions"]
                target_start = flat_timespan_batch["start_times"]
                target_end = flat_timespan_batch["end_times"]
                durations = flat_timespan_batch["durations"]

                timespan_hidden = last_hidden_states[flat_batch_indices, flat_positions, :]
                if self.use_timeple_interface_adapter:
                    output_adapter_output = self.timeple_interface_adapter.forward_output(
                        timespan_hidden,
                        compute_diagnostics=False,
                    )
                    timespan_hidden_proj = output_adapter_output.adapted.float()
                else:
                    timespan_hidden_proj = timespan_hidden.float()

                decoded_timespan = self.timeple_codec.decode_relative(timespan_hidden_proj)
                decode_losses = self.timeple_codec.compute_loss_from_decoded(
                    decoded_timespan,
                    target_start,
                    target_end,
                    durations,
                    reduction="none",
                )

                for batch_idx, (start_idx, end_idx) in flat_timespan_batch["sample_slices"].items():
                    sample_timeple_loss = torch.tensor(0.0, device=last_hidden_states.device, dtype=torch.float32)

                    decode_losses_slice = None
                    if decode_losses is not None:
                        decode_losses_slice = self._reduce_codec_loss_slice(decode_losses, start_idx, end_idx)
                        if (
                            timeple_dfl_loss_weight is None
                            and timeple_iou_loss_weight is None
                            and timeple_boundary_loss_weight is None
                        ):
                            decode_total_loss = decode_losses_slice["total_loss"]
                        else:
                            zero = decode_losses_slice["total_loss"].new_tensor(0.0)
                            decode_total_loss = (
                                float(timeple_dfl_loss_weight or 0.0) * decode_losses_slice.get("dfl_loss", zero)
                                + float(timeple_iou_loss_weight or 0.0) * decode_losses_slice.get("iou_loss", zero)
                                + float(timeple_boundary_loss_weight or 0.0) * decode_losses_slice.get("boundary_loss", zero)
                            )
                        sample_timeple_loss = sample_timeple_loss + timeple_decode_loss_weight * decode_total_loss

                    if batch_idx == 0 and decode_losses is not None and not self.training:
                        timeple_predictions = {
                            "start": decode_losses["pred_start_sec"][start_idx:end_idx].detach().cpu().tolist(),
                            "end": decode_losses["pred_end_sec"][start_idx:end_idx].detach().cpu().tolist(),
                        }

                    all_timeple_losses.append(sample_timeple_loss)
                    if batch_idx == 0:
                        if decode_losses_slice:
                            for k, v in decode_losses_slice.items():
                                if isinstance(v, torch.Tensor) and v.ndim == 0:
                                    timeple_loss_details[f"codec_{k}"] = v
                            if "mae_total" in decode_losses_slice:
                                timeple_loss_details["timeple_mae_total"] = decode_losses_slice["mae_total"]
                            if "span_iou" in decode_losses_slice:
                                timeple_loss_details["span_iou"] = decode_losses_slice["span_iou"]
                            timeple_loss_details["full_path_decode_loss"] = decode_losses_slice["total_loss"]
                        timeple_loss_details["timeple_loss"] = sample_timeple_loss

            if all_timeple_losses:
                timeple_loss = torch.stack(all_timeple_losses).mean()

            if timeple_loss_details is not None and self.use_timeple_interface_adapter:
                debug_state = self.timeple_interface_adapter.export_debug_state()
                for key, value in debug_state.items():
                    timeple_loss_details[key] = value.to(last_hidden_states.device)
                if input_adapter_ratios:
                    timeple_loss_details["input_residual_ratio"] = torch.stack(input_adapter_ratios).mean()

        total_loss = lm_loss
        if timeple_loss is not None and lm_loss is not None:
            total_loss = lm_loss + timeple_loss_weight * timeple_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return Qwen3VLOutputWithTimePLECodec(
            loss=total_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            timeple_loss=timeple_loss,
            timeple_predictions=timeple_predictions,
            timeple_loss_details=timeple_loss_details,
            last_hidden_state=hidden_states,
            csdo_logits=csdo_logits,
            csdo_features=csdo_features,
            csdo_valid=csdo_valid,
            tr_spd_logits=tr_spd_logits,
            tr_spd_features=tr_spd_features,
            tr_spd_valid=tr_spd_valid,
        )

    def decode_timeple_from_hidden_states(
        self,
        hidden_states: torch.Tensor,
        timeple_token_positions: List[int],
        video_duration_sec: Union[float, List[float], torch.Tensor],
        hard: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_timeple_codec:
            raise RuntimeError("TimePLE is not enabled.")
        timeple_hidden = hidden_states[timeple_token_positions, :]
        if self.use_timeple_interface_adapter:
            timeple_hidden = self.timeple_interface_adapter.forward_output(timeple_hidden).adapted
        timeple_hidden = timeple_hidden.float()
        return self.timeple_codec.decode(timeple_hidden, video_duration_sec=video_duration_sec, hard=hard)

    def format_timeple_output(
        self,
        start_sec: Union[float, torch.Tensor],
        end_sec: Union[float, torch.Tensor],
    ) -> str:
        if isinstance(start_sec, torch.Tensor):
            start_sec = start_sec.item()
        if isinstance(end_sec, torch.Tensor):
            end_sec = end_sec.item()

        def _format(seconds: float) -> str:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}:{secs:05.2f}"

        return f"from {_format(start_sec)} to {_format(end_sec)}"

    def generate_with_timeple_codec(
        self,
        input_ids: torch.LongTensor,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 128,
        **generate_kwargs,
    ) -> Dict:
        generated_ids = super().generate(
            input_ids=input_ids,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
        return {
            "generated_ids": generated_ids,
            "timeple_predictions": None,
        }

    def save_pretrained(self, save_directory: str, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        if self.use_timeple_codec:
            timeple_codec_path = os.path.join(save_directory, "timeple_codec.pth")
            torch.save(self.timeple_codec.state_dict(), timeple_codec_path)
            logger.info("TimePLE saved to %s", timeple_codec_path)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        config: Optional[Qwen3VLTimePLEConfig] = None,
        **kwargs,
    ):
        if config is None:
            timeple_codec_kwargs = {}
            allowed_keys = [
                "use_timeple_codec",
                "timestamp_token_id",
                "timespan_token_id",
                "timeple_codec_config",
                "use_timeple_interface_adapter",
                "timeple_interface_adapter",
                "freeze_vision",
                "freeze_language",
                "default_video_duration_sec",
            ]
            for key in allowed_keys:
                if key in kwargs:
                    timeple_codec_kwargs[key] = kwargs.pop(key)
            config = Qwen3VLTimePLEConfig.from_pretrained(
                pretrained_model_name_or_path,
                **timeple_codec_kwargs,
            )

        kwargs["config"] = config
        model = super(Qwen3VLForConditionalGenerationWithTimePLECodec, cls).from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )

        if hasattr(model, "timeple_codec") and config.use_timeple_codec:
            timeple_codec_path = os.path.join(pretrained_model_name_or_path, "timeple_codec.pth")
            if os.path.exists(timeple_codec_path):
                state_dict = torch.load(timeple_codec_path, map_location="cpu")
                model.timeple_codec.load_state_dict(state_dict)
                model_dtype = next(model.parameters()).dtype
                model.timeple_codec = model.timeple_codec.to(dtype=model_dtype)
                logger.info("TimePLE loaded from %s with dtype %s", timeple_codec_path, model_dtype)
            else:
                logger.warning("TimePLE weights not found at %s. Using random initialization.", timeple_codec_path)

        return model


__all__ = ["Qwen3VLForConditionalGenerationWithTimePLECodec", "Qwen3VLOutputWithTimePLECodec"]
