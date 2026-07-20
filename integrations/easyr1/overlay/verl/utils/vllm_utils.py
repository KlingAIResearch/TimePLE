from importlib.metadata import version
from typing import List

from msgspec import field
from packaging import version as vs
from vllm.lora.models import LoRAModel
from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager


class TensorLoRARequest(LoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


class VLLMHijack:
    _hijacked = False

    @staticmethod
    def hijack():
        if VLLMHijack._hijacked:
            return

        def hijack__load_adapter(self, lora_request: TensorLoRARequest) -> LoRAModel:
            supported_lora_modules = self._adapter_manager.supported_lora_modules
            packed_modules_mapping = self._adapter_manager.packed_modules_mapping
            expected_lora_modules: List[str] = []
            for module in supported_lora_modules:
                if module in packed_modules_mapping:
                    expected_lora_modules.extend(packed_modules_mapping[module])
                else:
                    expected_lora_modules.append(module)

            expected_lora_modules = list(set(expected_lora_modules))

            lora_tensors = None
            from vllm.lora.peft_helper import PEFTHelper

            if isinstance(lora_request, TensorLoRARequest):
                peft_config = lora_request.peft_config
                lora_tensors = lora_request.lora_tensors
                peft_helper = PEFTHelper.from_dict(peft_config)
            else:
                lora_path = get_adapter_absolute_path(lora_request.lora_path)
                peft_helper = PEFTHelper.from_local_dir(lora_path, self.max_position_embeddings)

            peft_helper.validate_legal(self.lora_config)

            model = self._adapter_manager.model
            hf_to_vllm_mapper = None
            if hasattr(model, "hf_to_vllm_mapper") and model.hf_to_vllm_mapper is not None:
                hf_to_vllm_mapper = model.hf_to_vllm_mapper

            if isinstance(lora_request, TensorLoRARequest):
                lora = self._lora_model_cls.from_lora_tensors(
                    lora_model_id=lora_request.lora_int_id,
                    tensors=lora_tensors,
                    peft_helper=peft_helper,
                    device="cpu",
                    dtype=self.lora_config.lora_dtype,
                    embeddings=None,
                    target_embedding_padding=self.vocab_size + self.lora_config.lora_extra_vocab_size,
                    embedding_modules=self.embedding_modules,
                    embedding_padding_modules=self.embedding_padding_modules,
                    weights_mapper=hf_to_vllm_mapper,
                )
            else:
                lora = self._lora_model_cls.from_local_checkpoint(
                    lora_path,
                    expected_lora_modules,
                    peft_helper=peft_helper,
                    lora_model_id=lora_request.lora_int_id,
                    device="cpu",
                    dtype=self.lora_config.lora_dtype,
                    target_embedding_padding=self.vocab_size + self.lora_config.lora_extra_vocab_size,
                    embedding_modules=self.embedding_modules,
                    embedding_padding_modules=self.embedding_padding_modules,
                    weights_mapper=hf_to_vllm_mapper,
                )

            if lora.extra_vocab_size > self.lora_config.lora_extra_vocab_size:
                raise ValueError(
                    f"LoRA added vocab size {lora.extra_vocab_size} "
                    f"is greater than lora_extra_vocab_size "
                    f"{self.lora_config.lora_extra_vocab_size}."
                )
            return lora

        setattr(LRUCacheWorkerLoRAManager, "_load_adapter", hijack__load_adapter)

        if vs.parse(version("vllm")).base_version == "0.11.0":
            from vllm.model_executor.models.module_mapping import MultiModelKeys
            from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
            from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

            def hijack__get_mm_mapping(self) -> MultiModelKeys:
                return MultiModelKeys.from_string_field(
                    language_model="language_model",
                    connector="visual.merger.",
                    tower_model="visual.",
                )

            setattr(Qwen3VLForConditionalGeneration, "get_mm_mapping", hijack__get_mm_mapping)

            original_get_input_positions_tensor = MRotaryEmbedding.get_input_positions_tensor.__func__

            def hijack__get_input_positions_tensor(
                cls,
                input_tokens,
                hf_config,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                context_len=0,
                seq_len=None,
                audio_feature_lengths=None,
                use_audio_in_video=False,
            ):
                original_model_type = getattr(hf_config, "model_type", None)
                if original_model_type in {
                    "qwen3_vl_time_codec",
                    "qwen3_vl_cis_codec",
                    "qwen3_vl_timeple_codec",
                    "qwen3_vl_timeple",
                    "qwen3_vl_timeed",
                }:
                    hf_config.model_type = "qwen3_vl"
                try:
                    return original_get_input_positions_tensor(
                        cls,
                        input_tokens=input_tokens,
                        hf_config=hf_config,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                        context_len=context_len,
                        seq_len=seq_len,
                        audio_feature_lengths=audio_feature_lengths,
                        use_audio_in_video=use_audio_in_video,
                    )
                finally:
                    if original_model_type in {
                        "qwen3_vl_time_codec",
                        "qwen3_vl_cis_codec",
                        "qwen3_vl_timeple_codec",
                        "qwen3_vl_timeple",
                        "qwen3_vl_timeed",
                    }:
                        hf_config.model_type = original_model_type

            setattr(MRotaryEmbedding, "get_input_positions_tensor", classmethod(hijack__get_input_positions_tensor))

        try:
            from transformers import AutoConfig, AutoProcessor
            from transformers.models.qwen3_vl.configuration_qwen3_vl_time_codec import Qwen3VLTimeCodecConfig
            from transformers.models.qwen3_vl.processing_qwen3_vl_time_codec import Qwen3VLProcessorWithTimeCodec

            AutoConfig.register("qwen3_vl_time_codec", Qwen3VLTimeCodecConfig, exist_ok=True)
            try:
                AutoProcessor.register(
                    Qwen3VLTimeCodecConfig,
                    Qwen3VLProcessorWithTimeCodec,
                    exist_ok=True,
                )
            except TypeError:
                AutoProcessor.register(Qwen3VLTimeCodecConfig, Qwen3VLProcessorWithTimeCodec)
        except Exception as exc:
            print(f"[verl][warn] Failed to register qwen3_vl_time_codec auto classes: {exc}")

        try:
            from transformers import AutoConfig, AutoProcessor
            from transformers.models.qwen3_vl.configuration_qwen3_vl_cis_codec import Qwen3VLCISCodecConfig
            from transformers.models.qwen3_vl.processing_qwen3_vl_cis_codec import Qwen3VLProcessorWithCISCodec

            AutoConfig.register("qwen3_vl_cis_codec", Qwen3VLCISCodecConfig, exist_ok=True)
            try:
                AutoProcessor.register(
                    Qwen3VLCISCodecConfig,
                    Qwen3VLProcessorWithCISCodec,
                    exist_ok=True,
                )
            except TypeError:
                AutoProcessor.register(Qwen3VLCISCodecConfig, Qwen3VLProcessorWithCISCodec)
        except Exception as exc:
            print(f"[verl][warn] Failed to register qwen3_vl_cis_codec auto classes: {exc}")

        try:
            from transformers import AutoConfig, AutoProcessor
            from transformers.models.qwen3_vl.configuration_qwen3_vl_timeple import (
                Qwen3VLTimePLECodecConfig,
            )
            from transformers.models.qwen3_vl.processing_qwen3_vl_timeple import (
                Qwen3VLProcessorWithTimePLECodec,
            )

            AutoConfig.register("qwen3_vl_timeple_codec", Qwen3VLTimePLECodecConfig, exist_ok=True)
            try:
                AutoProcessor.register(
                    Qwen3VLTimePLECodecConfig,
                    Qwen3VLProcessorWithTimePLECodec,
                    exist_ok=True,
                )
            except TypeError:
                AutoProcessor.register(Qwen3VLTimePLECodecConfig, Qwen3VLProcessorWithTimePLECodec)
        except Exception as exc:
            print(f"[verl][warn] Failed to register qwen3_vl_timeple_codec auto classes: {exc}")

        try:
            from transformers import AutoConfig, AutoProcessor
            from transformers.models.qwen3_vl.configuration_qwen3_vl_timeple import (
                Qwen3VLTimePLEConfig,
            )
            from transformers.models.qwen3_vl.processing_qwen3_vl_timeple import (
                Qwen3VLProcessorWithTimePLECodec,
            )

            AutoConfig.register(
                "qwen3_vl_timeple",
                Qwen3VLTimePLEConfig,
                exist_ok=True,
            )
            try:
                AutoProcessor.register(
                    Qwen3VLTimePLEConfig,
                    Qwen3VLProcessorWithTimePLECodec,
                    exist_ok=True,
                )
            except TypeError:
                AutoProcessor.register(
                    Qwen3VLTimePLEConfig,
                    Qwen3VLProcessorWithTimePLECodec,
                )
        except Exception as exc:
            print(
                "[verl][warn] Failed to register "
                f"qwen3_vl_timeple auto classes: {exc}"
            )

        try:
            from transformers import AutoConfig, AutoProcessor
            from transformers.models.qwen3_vl.configuration_qwen3_vl_timeed import Qwen3VLTimeEDConfig
            from transformers.models.qwen3_vl.processing_qwen3_vl_timeed import Qwen3VLProcessorWithTimeED

            AutoConfig.register("qwen3_vl_timeed", Qwen3VLTimeEDConfig, exist_ok=True)
            try:
                AutoProcessor.register(
                    Qwen3VLTimeEDConfig,
                    Qwen3VLProcessorWithTimeED,
                    exist_ok=True,
                )
            except TypeError:
                AutoProcessor.register(Qwen3VLTimeEDConfig, Qwen3VLProcessorWithTimeED)
        except Exception as exc:
            print(f"[verl][warn] Failed to register qwen3_vl_timeed auto classes: {exc}")

        try:
            from .vllm_qwen3_vl_time_codec import register_qwen3_vl_time_codec_model

            register_qwen3_vl_time_codec_model()
        except Exception as exc:
            print(f"[verl][warn] Failed to register qwen3_vl_time_codec vLLM model: {exc}")

        try:
            from .vllm_qwen3_vl_cis_codec import register_qwen3_vl_cis_codec_model

            register_qwen3_vl_cis_codec_model()
        except Exception as exc:
            print(f"[verl][warn] Failed to register qwen3_vl_cis_codec vLLM model: {exc}")

        try:
            from .vllm_qwen3_vl_timeple import register_qwen3_vl_timeple_codec_model

            register_qwen3_vl_timeple_codec_model()
        except Exception as exc:
            print(f"[verl][warn] Failed to register qwen3_vl_timeple_codec vLLM model: {exc}")

        try:
            from .vllm_qwen3_vl_timeple import (
                register_qwen3_vl_timeple_model,
            )

            register_qwen3_vl_timeple_model()
        except Exception as exc:
            print(
                "[verl][warn] Failed to register "
                f"qwen3_vl_timeple vLLM model: {exc}"
            )

        try:
            from .vllm_qwen3_vl_timeed import register_qwen3_vl_timeed_model

            register_qwen3_vl_timeed_model()
        except Exception as exc:
            print(f"[verl][warn] Failed to register qwen3_vl_timeed vLLM model: {exc}")

        VLLMHijack._hijacked = True
