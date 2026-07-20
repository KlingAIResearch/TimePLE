from __future__ import annotations

import inspect
import logging
from typing import Any


LOGGER = logging.getLogger(__name__)


def _filter_kwargs_for_callable(
    callable_obj: Any,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return dict(kwargs)

    allowed_keys = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {key: value for key, value in kwargs.items() if key in allowed_keys}


def apply_internvl_video_processor_patch() -> None:
    """Patch vLLM InternVL video processor construction for mixed mm kwargs.

    vLLM 0.19 forwards the merged `mm_processor_kwargs` dictionary to both the
    image and video processors when building InternVL's multimodal processor.
    InternVL's image processor accepts keys such as `dynamic_image_size` and
    `min_dynamic_patch`, while the paired video processor only accepts
    `image_size`. When legacy job manifests still include image-only kwargs,
    vLLM fails during model initialization before any samples are processed.

    The patch keeps the image-side behavior intact and filters video processor
    constructor kwargs down to the parameters that it actually supports.
    """

    try:
        from vllm.model_executor.models.internvl import InternVLProcessingInfo
        from vllm.transformers_utils.processors.internvl import InternVLVideoProcessor
    except ModuleNotFoundError:
        LOGGER.warning(
            "Skipped InternVL vLLM compatibility patch because the current "
            "interpreter does not expose the expected vLLM InternVL modules."
        )
        return

    original_get_video_processor = InternVLProcessingInfo.get_video_processor
    if getattr(original_get_video_processor, "_eval_suite_patched", False):
        return

    def patched_get_video_processor(self, **kwargs):
        config = self.get_hf_config()
        vision_config = config.vision_config

        merged_kwargs = self.ctx.get_merged_mm_kwargs(kwargs)
        merged_kwargs.setdefault("image_size", vision_config.image_size)
        filtered_kwargs = _filter_kwargs_for_callable(
            InternVLVideoProcessor.__init__,
            merged_kwargs,
        )
        return InternVLVideoProcessor(**filtered_kwargs)

    patched_get_video_processor._eval_suite_patched = True  # type: ignore[attr-defined]
    InternVLProcessingInfo.get_video_processor = patched_get_video_processor
    LOGGER.info(
        "Applied eval_suite vLLM compatibility patch for InternVL video processor kwargs."
    )
