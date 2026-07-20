import os
from contextlib import contextmanager
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from ...utils.vllm_utils import VLLMHijack
from .base import BaseRollout
from .config import RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_timespan_token_id(processor: Optional[ProcessorMixin]) -> Optional[int]:
    if processor is None or not hasattr(processor, "tokenizer"):
        return None
    token_id = processor.tokenizer.convert_tokens_to_ids("<|TIMESPAN|>")
    if token_id is None:
        return None
    try:
        token_id = int(token_id)
    except (TypeError, ValueError):
        return None
    return token_id if token_id >= 0 else None


def _merge_logit_bias(
    logit_bias: Optional[dict[int, float]],
    token_id: Optional[int],
    bias: float,
) -> Optional[dict[int, float]]:
    if token_id is None or float(bias) == 0.0:
        return logit_bias
    merged = dict(logit_bias or {})
    merged[int(token_id)] = float(bias)
    return merged


def _get_logit_bias(
    processor: Optional[ProcessorMixin],
    *,
    timespan_logit_bias: float = 0.0,
) -> Optional[dict[int, float]]:
    logit_bias = None
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        logit_bias = {image_token_id: -100}
    return _merge_logit_bias(logit_bias, _get_timespan_token_id(processor), timespan_logit_bias)


def _build_mm_processor_kwargs(
    timestamp_labels: Any,
    *,
    include_video_durations: bool = False,
) -> Optional[dict[str, Any]]:
    if timestamp_labels is None or not isinstance(timestamp_labels, dict):
        return None

    starts = timestamp_labels.get("start")
    ends = timestamp_labels.get("end")
    if starts is None or ends is None:
        return None

    if isinstance(starts, np.ndarray):
        starts = starts.tolist()
    if isinstance(ends, np.ndarray):
        ends = ends.tolist()

    mm_processor_kwargs = {
        "timestamp_labels_start": starts,
        "timestamp_labels_end": ends,
    }
    if include_video_durations:
        durations = timestamp_labels.get("video_duration")
        if isinstance(durations, np.ndarray):
            durations = durations.tolist()
        if durations is not None:
            mm_processor_kwargs["timestamp_video_durations"] = durations

    return mm_processor_kwargs


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any],
    min_pixels: int,
    max_pixels: int,
    video_fps: float,
    video_max_token_num: Optional[int] = None,
    fps_max_frames: Optional[int] = None,
) -> dict[str, Any]:
    images, videos = [], []
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            processed_video, sampled_fps = process_video(
                video,
                min_pixels,
                max_pixels,
                video_fps,
                return_fps=True,
                video_max_token_num=video_max_token_num,
                fps_max_frames=fps_max_frames,
            )
            effective_fps = sampled_fps if sampled_fps is not None else video_fps
            video_metadata = {
                "fps": effective_fps,
                "duration": float(len(processed_video)) / float(effective_fps) if effective_fps else 0.0,
                "total_num_frames": len(processed_video),
                "frames_indices": list(range(len(processed_video))),
                "video_backend": "easyr1",
                "do_sample_frames": False,
            }
            videos.append((processed_video, video_metadata))

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        **kwargs,
    ):
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        self.use_cis_codec = bool(getattr(processor, "use_cis_codec", False)) if processor is not None else False
        self.use_timeple_codec = (
            bool(getattr(processor, "use_timeple_codec", False)) if processor is not None else False
        )
        self.use_timeple_codec = (
            bool(getattr(processor, "use_timeple_codec", False)) if processor is not None else False
        )
        self.use_timeed = bool(getattr(processor, "use_timeed", False)) if processor is not None else False
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs

        engine_kwargs = {}
        if processor is not None:
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if config.enable_sleep_mode and "expandable_segments:True" in alloc_conf:
            raise ValueError(
                "worker.rollout.enable_sleep_mode=True is incompatible with "
                "PYTORCH_CUDA_ALLOC_CONF containing 'expandable_segments:True' because "
                "vLLM sleep mode uses the CuMemAllocator memory pool. "
                "Unset PYTORCH_CUDA_ALLOC_CONF, choose a different allocator setting, "
                "or set worker.rollout.enable_sleep_mode=false."
            )

        VLLMHijack.hijack()

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy" if not self.lora_kwargs else "safetensors",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=config.enable_sleep_mode,
            **lora_kwargs,
            **engine_kwargs,
        )

        if config.enable_sleep_mode:
            self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor, timespan_logit_bias=config.timespan_logit_bias),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        input_ids: torch.Tensor = prompts.batch["input_ids"]
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        batch_timestamp_labels = non_tensor_batch.pop("timestamp_labels", None)
        batch_timestamp_positions = non_tensor_batch.pop("timestamp_positions", None)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            vllm_timestamp_labels = (
                [None] * len(batch_raw_prompt_ids) if batch_timestamp_labels is None else batch_timestamp_labels
            )

            for raw_prompt_ids, multi_modal_data, timestamp_labels in zip(
                batch_raw_prompt_ids,
                batch_multi_modal_data,
                vllm_timestamp_labels,
            ):
                vllm_input = {
                    "prompt_token_ids": list(raw_prompt_ids),
                    "multi_modal_data": _process_multi_modal_data(
                        multi_modal_data,
                        prompts.meta_info["min_pixels"],
                        prompts.meta_info["max_pixels"],
                        prompts.meta_info["video_fps"],
                        prompts.meta_info.get("video_max_token_num"),
                        prompts.meta_info.get("fps_max_frames"),
                    ),
                }
                mm_processor_kwargs = _build_mm_processor_kwargs(
                    timestamp_labels,
                    include_video_durations=(
                        self.use_cis_codec
                        or self.use_timeple_codec
                        or self.use_timeple_codec
                        or self.use_timeed
                    ),
                )
                if mm_processor_kwargs is not None:
                    vllm_input["mm_processor_kwargs"] = mm_processor_kwargs
                vllm_inputs.append(vllm_input)
        elif batch_timestamp_labels is not None:
            vllm_inputs = []
            for raw_prompt_ids, timestamp_labels in zip(batch_raw_prompt_ids, batch_timestamp_labels):
                vllm_input = {"prompt_token_ids": list(raw_prompt_ids)}
                mm_processor_kwargs = _build_mm_processor_kwargs(
                    timestamp_labels,
                    include_video_durations=(
                        self.use_cis_codec
                        or self.use_timeple_codec
                        or self.use_timeple_codec
                        or self.use_timeed
                    ),
                )
                if mm_processor_kwargs is not None:
                    vllm_input["mm_processor_kwargs"] = mm_processor_kwargs
                vllm_inputs.append(vllm_input)
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        with self.update_sampling_params(**prompts.meta_info):
            completions: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=self.use_tqdm,
            )
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)
                if batch_timestamp_labels is not None:
                    batch_timestamp_labels = _repeat_interleave(batch_timestamp_labels, self.sampling_params.n)
                if batch_timestamp_positions is not None:
                    batch_timestamp_positions = _repeat_interleave(batch_timestamp_positions, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        non_tensor_batch = {}
        if batch_multi_modal_data is not None:
            non_tensor_batch["multi_modal_data"] = batch_multi_modal_data
        if batch_timestamp_labels is not None:
            non_tensor_batch["timestamp_labels"] = batch_timestamp_labels
        if batch_timestamp_positions is not None:
            non_tensor_batch["timestamp_positions"] = batch_timestamp_positions

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)
