# coding=utf-8
"""Qwen3-VL model configuration with span-only TimePLE support."""

from copy import deepcopy
from typing import Optional

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig


class Qwen3VLTimePLEConfig(Qwen3VLConfig):
    """
    Extends ``Qwen3VLConfig`` with span-only Canonical Interval Square settings.
    """

    model_type = "qwen3_vl_timeple"

    def __init__(
        self,
        use_timeple_codec: bool = True,
        timestamp_token_id: int = 151669,
        timespan_token_id: int = 151670,
        timeple_codec_config: dict = None,
        use_timeple_interface_adapter: bool = False,
        timeple_interface_adapter: Optional[dict] = None,
        freeze_vision: bool = False,
        freeze_language: bool = False,
        default_video_duration_sec: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        hidden_size = self.text_config.hidden_size

        self.use_timeple_codec = use_timeple_codec
        self.timestamp_token_id = timestamp_token_id
        self.timespan_token_id = timespan_token_id
        self.use_timeple_interface_adapter = bool(use_timeple_interface_adapter)
        self.timeple_interface_adapter = deepcopy(timeple_interface_adapter) if timeple_interface_adapter is not None else None
        self.freeze_vision = freeze_vision
        self.freeze_language = freeze_language
        self.default_video_duration_sec = float(default_video_duration_sec)

        if timeple_codec_config is None:
            timeple_codec_config = {
                "token_dim": hidden_size,
                "transform": {
                    "tau": 0.0,
                    "eps": 1e-6,
                },
                "grid": {
                    "num_u_bins": 128,
                    "num_v_bins": 128,
                    "span_sigma_u": 0.015,
                    "span_sigma_v": 0.015,
                },
                "encoder": {
                    "hidden_dims": [512, 2048],
                    "activation": "gelu",
                    "dropout": 0.0,
                    "use_layer_norm": True,
                },
                "decoder": {
                    "trunk_hidden_dims": [2048, 1024],
                    "activation": "gelu",
                    "dropout": 0.0,
                    "use_layer_norm": True,
                    "duration_adaptive_residual": {
                        "enabled": True,
                        "hidden_dims": [256],
                        "scale": 0.02,
                        "activation": "gelu",
                        "dropout": 0.0,
                        "use_layer_norm": True,
                        "weight_mode": "relative_sqrt_inv",
                        "weight_eps": 1e-4,
                        "weight_max": 10.0,
                    },
                },
                "loss": {
                    "lambda_dfl": 1.0,
                    "lambda_iou": 1.0,
                    "lambda_boundary": 0.5,
                    "boundary_weight_mode": "absolute_bucket",
                    "boundary_weight_eps": 1e-4,
                    "boundary_weight_max": 10.0,
                    "boundary_weight_normalize": True,
                    "absolute_short_threshold_sec": 10.0,
                    "absolute_medium_threshold_sec": 30.0,
                    "absolute_short_weight": 3.0,
                    "absolute_medium_weight": 1.5,
                    "absolute_long_weight": 1.0,
                },
            }

        self.timeple_codec_config = timeple_codec_config

    def get_timeple_codec_output_dim(self) -> int:
        return int(self.timeple_codec_config.get("token_dim", self.text_config.hidden_size))

    def validate_config(self):
        if not self.use_timeple_codec:
            return

        codec_dim = self.get_timeple_codec_output_dim()
        model_hidden_size = self.text_config.hidden_size
        if codec_dim != model_hidden_size:
            raise ValueError(
                f"TimePLE output dimension ({codec_dim}) must match "
                f"model hidden size ({model_hidden_size})."
            )
        if self.use_timeple_interface_adapter and not self.use_timeple_codec:
            raise ValueError("`use_timeple_interface_adapter` requires `use_timeple_codec=True`.")


__all__ = ["Qwen3VLTimePLEConfig"]
