from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

from transformers import AutoConfig

import qwen_cis_codec_vllm as _cis_base


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_PATCHED = False


def _resolve_cis_span_duration_adaptive_codec_dir() -> Path:
    return PROJECT_ROOT / "src" / "timeple"


def _get_cis_span_duration_adaptive_codec_classes():
    from timeple import TimePLECodec, TimePLEInterfaceAdapter

    return TimePLEInterfaceAdapter, TimePLECodec


class Qwen3VLCISSpanDurationAdaptiveCodecConfig(_cis_base.Qwen3VLCISCodecConfig):
    model_type = "qwen3_vl_timeple"

    def __init__(
        self,
        use_cis_span_duration_adaptive_codec: bool = True,
        timestamp_token_id: int = 151669,
        timespan_token_id: int = 151670,
        cis_span_duration_adaptive_codec_config: dict[str, Any] | None = None,
        use_cis_span_duration_adaptive_interface_adapter: bool = False,
        cis_span_duration_adaptive_interface_adapter: dict[str, Any] | None = None,
        freeze_vision: bool = False,
        freeze_language: bool = False,
        default_video_duration_sec: float = 1.0,
        **kwargs: Any,
    ) -> None:
        generic_use_cis_codec = kwargs.pop(
            "use_timeple_codec",
            kwargs.pop("use_cis_codec", use_cis_span_duration_adaptive_codec),
        )
        generic_cis_codec_config = kwargs.pop(
            "timeple_codec_config",
            kwargs.pop("cis_codec_config", cis_span_duration_adaptive_codec_config),
        )
        generic_use_adapter = kwargs.pop(
            "use_timeple_interface_adapter",
            kwargs.pop("use_cis_interface_adapter", use_cis_span_duration_adaptive_interface_adapter),
        )
        generic_adapter = kwargs.pop(
            "timeple_interface_adapter",
            kwargs.pop("cis_interface_adapter", cis_span_duration_adaptive_interface_adapter),
        )

        super().__init__(
            use_cis_codec=generic_use_cis_codec,
            timestamp_token_id=timestamp_token_id,
            timespan_token_id=timespan_token_id,
            cis_codec_config=generic_cis_codec_config,
            use_cis_interface_adapter=generic_use_adapter,
            cis_interface_adapter=generic_adapter,
            freeze_vision=freeze_vision,
            freeze_language=freeze_language,
            default_video_duration_sec=default_video_duration_sec,
            **kwargs,
        )

        self.use_cis_span_duration_adaptive_codec = bool(generic_use_cis_codec)
        self.cis_span_duration_adaptive_codec_config = deepcopy(self.cis_codec_config)
        self.use_cis_span_duration_adaptive_interface_adapter = bool(generic_use_adapter)
        self.cis_span_duration_adaptive_interface_adapter = (
            deepcopy(self.cis_interface_adapter)
            if self.cis_interface_adapter is not None
            else None
        )

    def get_cis_span_duration_adaptive_codec_output_dim(self) -> int:
        return int(
            self.cis_span_duration_adaptive_codec_config.get(
                "token_dim",
                self.text_config.hidden_size,
            )
        )


class Qwen3VLCISSpanDurationAdaptiveCodecProcessingInfo(
    _cis_base.Qwen3VLCISCodecProcessingInfo
):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3VLCISSpanDurationAdaptiveCodecConfig)


def _rename_cis_span_duration_adaptive_weight(name: str) -> str:
    if name.startswith("timeple_codec."):
        return "cis_codec." + name.removeprefix("timeple_codec.")
    if name.startswith("timeple_interface_adapter."):
        return "cis_interface_adapter." + name.removeprefix("timeple_interface_adapter.")
    if name.startswith("cis_span_duration_adaptive_codec."):
        return "cis_codec." + name.removeprefix(
            "cis_span_duration_adaptive_codec."
        )
    if name.startswith("cis_span_duration_adaptive_interface_adapter."):
        return "cis_interface_adapter." + name.removeprefix(
            "cis_span_duration_adaptive_interface_adapter."
        )
    return name


@_cis_base.MULTIMODAL_REGISTRY.register_processor(
    _cis_base.Qwen3VLCISCodecMultiModalProcessor,
    info=Qwen3VLCISSpanDurationAdaptiveCodecProcessingInfo,
    dummy_inputs=_cis_base.Qwen3VLDummyInputsBuilder,
)
class Qwen3VLForConditionalGenerationWithCISSpanDurationAdaptiveCodec(
    _cis_base.Qwen3VLForConditionalGenerationWithCISCodec
):
    config_class = Qwen3VLCISSpanDurationAdaptiveCodecConfig

    def __init__(self, *, vllm_config, prefix: str = "model"):
        original_get_classes = _cis_base._get_cis_codec_classes
        _cis_base._get_cis_codec_classes = (
            _get_cis_span_duration_adaptive_codec_classes
        )
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            _cis_base._get_cis_codec_classes = original_get_classes

    def decode_cis_span_duration_adaptive_hidden_states(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        return self.decode_cis_hidden_states(*args, **kwargs)

    def load_weights(self, weights) -> set[str]:
        renamed_weights = (
            (_rename_cis_span_duration_adaptive_weight(name), tensor)
            for name, tensor in weights
        )
        return super().load_weights(renamed_weights)


def apply_timeple_vllm_runtime_patch() -> None:
    global _RUNTIME_PATCHED
    if _RUNTIME_PATCHED:
        return

    _cis_base._CONFIG_REGISTRY[
        "qwen3_vl_timeple"
    ] = Qwen3VLCISSpanDurationAdaptiveCodecConfig
    AutoConfig.register(
        "qwen3_vl_timeple",
        Qwen3VLCISSpanDurationAdaptiveCodecConfig,
        exist_ok=True,
    )
    _cis_base.ModelRegistry.register_model(
        "Qwen3VLForConditionalGenerationWithTimePLECodec",
        Qwen3VLForConditionalGenerationWithCISSpanDurationAdaptiveCodec,
    )
    _cis_base._patch_gpu_model_runner_sample_tokens()
    _cis_base._patch_gpu_model_runner_update_states()
    _cis_base._patch_scheduler_update_from_output()
    _RUNTIME_PATCHED = True
    LOGGER.info(
        "Registered TimePLE vLLM runtime patch using local package at %s",
        _resolve_cis_span_duration_adaptive_codec_dir(),
    )
