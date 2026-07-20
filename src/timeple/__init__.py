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
)

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
]

__version__ = "0.1.0"
