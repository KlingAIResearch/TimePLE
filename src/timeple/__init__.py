"""Duration-adaptive TimePLE-Codec package."""

from .models import (
    TimePLEDecoderOutput,
    TimePLEInputAdapter,
    TimePLEInputAdapterOutput,
    TimePLEInterfaceAdapter,
    TimePLEOutputAdapter,
    TimePLEOutputAdapterOutput,
    TimePLEProjectorInputAdapter,
    TimePLEProjectorOutputAdapter,
    CanonicalSpanIntervalCodec,
    CanonicalSpanIntervalTransform,
    TimePLECodec,
    Qwen3VLTimePLEConfig,
    Qwen3VLForConditionalGenerationWithTimePLECodec,
    Qwen3VLOutputWithTimePLECodec,
    Qwen3VLProcessorWithTimePLECodec,
)


def register_transformers_auto_classes() -> None:
    """Register TimePLE with Transformers without executing Hub-hosted code."""

    from transformers import AutoConfig, AutoModel, AutoModelForImageTextToText, AutoProcessor

    AutoConfig.register(Qwen3VLTimePLEConfig.model_type, Qwen3VLTimePLEConfig, exist_ok=True)
    AutoModel.register(
        Qwen3VLTimePLEConfig,
        Qwen3VLForConditionalGenerationWithTimePLECodec,
        exist_ok=True,
    )
    AutoModelForImageTextToText.register(
        Qwen3VLTimePLEConfig,
        Qwen3VLForConditionalGenerationWithTimePLECodec,
        exist_ok=True,
    )
    AutoProcessor.register(
        Qwen3VLTimePLEConfig,
        Qwen3VLProcessorWithTimePLECodec,
        exist_ok=True,
    )


register_transformers_auto_classes()

__all__ = [
    "TimePLEDecoderOutput",
    "TimePLEInputAdapter",
    "TimePLEInputAdapterOutput",
    "TimePLEInterfaceAdapter",
    "TimePLEOutputAdapter",
    "TimePLEOutputAdapterOutput",
    "TimePLEProjectorInputAdapter",
    "TimePLEProjectorOutputAdapter",
    "CanonicalSpanIntervalCodec",
    "CanonicalSpanIntervalTransform",
    "TimePLECodec",
    "Qwen3VLTimePLEConfig",
    "Qwen3VLForConditionalGenerationWithTimePLECodec",
    "Qwen3VLOutputWithTimePLECodec",
    "Qwen3VLProcessorWithTimePLECodec",
    "register_transformers_auto_classes",
]

__version__ = "0.1.0"
