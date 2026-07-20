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
The main entry point to run the PPO algorithm
"""

from contextlib import nullcontext
import json
import os
from typing import Literal, Optional, Union, cast

import numpy as np
import peft
import psutil
import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from codetiming import Timer
from peft import TaskType, get_peft_model
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTokenClassification,
    GenerationConfig,
    PreTrainedModel,
)
from transformers.modeling_utils import no_init_weights

from ..models.monkey_patch import apply_ulysses_patch
from ..protocol import DataProto
from ..single_controller.base import Worker
from ..single_controller.base.decorator import Dispatch, register
from ..utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from ..utils.dataset import process_image, process_video
from ..utils.flops_counter import FlopsCounter
from ..utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_fn,
    load_fsdp_model,
    load_fsdp_optimizer,
    offload_fsdp_model,
    offload_fsdp_optimizer,
)
from ..utils.model_utils import print_gpu_memory_usage, print_model_size
from ..utils.tokenizer import get_processor, get_tokenizer
from ..utils.torch_dtypes import PrecisionType
from ..utils.torch_functional import (
    AnyPrecisionAdamW,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from .config import ActorConfig, CriticConfig, FSDPConfig, ModelConfig, OptimConfig, WorkerConfig
from .rollout import vLLMRollout
from .sharding_manager import FSDPVLLMShardingManager
from .sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager


class FSDPWorker(Worker):
    def __init__(
        self,
        config: WorkerConfig,
        role: Literal["actor", "critic", "rollout", "ref", "actor_rollout", "actor_rollout_ref"],
    ):
        super().__init__()
        self.config = config
        self.role = role
        self._cache = {}

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        # improve numerical stability
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        self._has_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._has_critic = self.role == "critic"
        self._has_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._has_ref = self.role in ["ref", "actor_rollout_ref"]
        if self._has_actor and self._has_critic:
            raise ValueError("Actor and critic cannot be both initialized.")

        if self.config.actor.disable_kl:
            self._has_ref = False

        self._lora_rank = self.config.actor.model.lora.rank
        self._is_lora = self._lora_rank > 0

        self._use_param_offload = False
        self._use_optimizer_offload = False
        self._use_ref_param_offload = False
        if self._has_actor:
            self._use_param_offload = self.config.actor.offload.offload_params
            self._use_optimizer_offload = self.config.actor.offload.offload_optimizer
            self._init_dist_mesh(self.config.actor, "actor")

        if self._has_critic:
            self._use_param_offload = self.config.critic.offload.offload_params
            self._use_optimizer_offload = self.config.critic.offload.offload_optimizer
            self._init_dist_mesh(self.config.critic, "critic")

        if self._has_ref:  # NOTE: it seems that manual offload is slower than FSDP offload
            self._use_ref_param_offload = self.config.ref.offload.offload_params

    def _init_dist_mesh(self, config: Union[ActorConfig, CriticConfig], role: Literal["actor", "critic"]):
        world_size = dist.get_world_size()
        # create main device mesh
        fsdp_size = config.fsdp.fsdp_size
        if fsdp_size <= 0 or fsdp_size >= world_size:
            self.device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
        else:  # hsdp
            self.device_mesh = init_device_mesh(
                "cuda", mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=("ddp", "fsdp")
            )

        # create ulysses device mesh
        if config.ulysses_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(world_size // config.ulysses_size, config.ulysses_size),
                mesh_dim_names=("dp", "sp"),
            )
        else:
            self.ulysses_device_mesh = None

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # validate and normalize config
        span_samples_per_response = 1
        rollout_multiplier = self.config.rollout.n * span_samples_per_response
        if rollout_multiplier > 1:
            config.global_batch_size *= rollout_multiplier
            self.print_rank0(
                f"{role} will use global batch size {config.global_batch_size} "
                f"(rollout.n={self.config.rollout.n}, timeed_span_samples_per_response={span_samples_per_response})."
            )

        config.global_batch_size_per_device = config.global_batch_size // (world_size // config.ulysses_size)
        if config.global_batch_size_per_device == 0:
            raise ValueError(f"{role} global batch size * ulysses size must be larger than num gpus.")

        if config.global_batch_size_per_device % config.micro_batch_size_per_device_for_update != 0:
            raise ValueError(f"{role} global batch size per device must be divisible by the micro batch size.")

        if (
            config.fsdp.enable_cpu_offload
            and config.global_batch_size_per_device != config.micro_batch_size_per_device_for_update
        ):
            raise ValueError(f"{role} cannot use FSDP's CPU offload when gradient accumulation is enabled.")

    @staticmethod
    def _read_model_type(model_path: Optional[str]) -> Optional[str]:
        if model_path is None:
            return None
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            return None
        try:
            with open(config_path, encoding="utf-8") as f:
                return json.load(f).get("model_type")
        except Exception:
            return None

    def _load_model_config(self, model_config: ModelConfig):
        common_kwargs = {
            "trust_remote_code": model_config.trust_remote_code,
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            **model_config.override_config,
        }
        try:
            return AutoConfig.from_pretrained(model_config.model_path, **common_kwargs)
        except ValueError:
            model_type = self._read_model_type(model_config.model_path)
            if model_type == "qwen3_vl_time_codec":
                from transformers.models.qwen3_vl.configuration_qwen3_vl_time_codec import Qwen3VLTimeCodecConfig

                self.print_rank0("Fallback to Qwen3VLTimeCodecConfig for custom model_type=qwen3_vl_time_codec.")
                return Qwen3VLTimeCodecConfig.from_pretrained(model_config.model_path, **common_kwargs)
            if model_type == "qwen3_vl_cis_codec":
                from transformers.models.qwen3_vl.configuration_qwen3_vl_cis_codec import Qwen3VLCISCodecConfig

                self.print_rank0("Fallback to Qwen3VLCISCodecConfig for custom model_type=qwen3_vl_cis_codec.")
                return Qwen3VLCISCodecConfig.from_pretrained(model_config.model_path, **common_kwargs)
            if model_type == "qwen3_vl_timeple_codec":
                from transformers.models.qwen3_vl.configuration_qwen3_vl_timeple import (
                    Qwen3VLTimePLECodecConfig,
                )

                self.print_rank0(
                    "Fallback to Qwen3VLTimePLECodecConfig for custom model_type=qwen3_vl_timeple_codec."
                )
                return Qwen3VLTimePLECodecConfig.from_pretrained(model_config.model_path, **common_kwargs)
            if model_type == "qwen3_vl_timeple":
                from transformers.models.qwen3_vl.configuration_qwen3_vl_timeple import (
                    Qwen3VLTimePLEConfig,
                )

                self.print_rank0(
                    "Fallback to Qwen3VLTimePLEConfig for "
                    "custom model_type=qwen3_vl_timeple."
                )
                return Qwen3VLTimePLEConfig.from_pretrained(
                    model_config.model_path,
                    **common_kwargs,
                )
            if model_type == "qwen3_vl_timeed":
                from transformers.models.qwen3_vl.configuration_qwen3_vl_timeed import Qwen3VLTimeEDConfig

                self.print_rank0("Fallback to Qwen3VLTimeEDConfig for custom model_type=qwen3_vl_timeed.")
                return Qwen3VLTimeEDConfig.from_pretrained(model_config.model_path, **common_kwargs)
            else:
                raise

    def _build_model_optimizer(
        self,
        model_config: ModelConfig,
        fsdp_config: FSDPConfig,
        optim_config: Optional[OptimConfig],
        padding_free: bool,
        role: Literal["actor", "critic", "ref"],
    ) -> None:
        if role != "ref":  # ref model's tokenizer is same as actor
            self.tokenizer = get_tokenizer(
                model_config.tokenizer_path,
                trust_remote_code=model_config.trust_remote_code,
                use_fast=True,
            )
            self.processor = get_processor(
                model_config.tokenizer_path,
                trust_remote_code=model_config.trust_remote_code,
                use_fast=True,
            )
            self.model_config = self._load_model_config(model_config)

            try:
                self.generation_config = GenerationConfig.from_pretrained(model_config.model_path)
            except Exception:
                self.generation_config = GenerationConfig.from_model_config(self.model_config)

            self.print_rank0(f"Model config: {self.model_config}")

        custom_qwen3_vl_codec = getattr(self.model_config, "model_type", None) in {
            "qwen3_vl_time_codec",
            "qwen3_vl_cis_codec",
            "qwen3_vl_timeple_codec",
            "qwen3_vl_timeple",
            "qwen3_vl_timeed",
        }
        if padding_free or custom_qwen3_vl_codec:
            ulysses_patch_model_type = self.model_config.model_type
            if custom_qwen3_vl_codec:
                # Codec models reuse the Qwen3-VL backbone and require the mixed-data forward patch
                # even when padding-free is disabled, because timestamp runtime is injected there.
                # Reuse the Qwen3-VL ulysses patch on the shared backbone modules.
                ulysses_patch_model_type = "qwen3_vl"
            apply_ulysses_patch(ulysses_patch_model_type)
            self.print_rank0(f"Ulysses patch applied for model_type={ulysses_patch_model_type}.")

        if fsdp_config.torch_dtype is None:
            torch_dtype = torch.float32 if role != "ref" else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(fsdp_config.torch_dtype)

        if role == "critic":
            AutoClass = AutoModelForTokenClassification
        elif getattr(self.model_config, "model_type", None) in {
            "qwen3_vl_time_codec",
            "qwen3_vl_cis_codec",
            "qwen3_vl_timeple_codec",
            "qwen3_vl_timeple",
            "qwen3_vl_timeed",
        }:
            AutoClass = None
        elif type(self.model_config) in AutoModelForImageTextToText._model_mapping.keys():
            AutoClass = AutoModelForImageTextToText
        else:
            AutoClass = AutoModelForCausalLM

        if AutoClass is None:
            model_type = getattr(self.model_config, "model_type", None)
            if model_type == "qwen3_vl_time_codec":
                from transformers.models.qwen3_vl.modeling_qwen3_vl_time_codec import (
                    Qwen3VLForConditionalGenerationWithTimeCodec as CustomQwen3VLForConditionalGeneration,
                )
            elif model_type == "qwen3_vl_cis_codec":
                from transformers.models.qwen3_vl.modeling_qwen3_vl_cis_codec import (
                    Qwen3VLForConditionalGenerationWithCISCodec as CustomQwen3VLForConditionalGeneration,
                )
            elif model_type == "qwen3_vl_timeple_codec":
                from transformers.models.qwen3_vl.modeling_qwen3_vl_timeple import (
                    Qwen3VLForConditionalGenerationWithTimePLECodec as CustomQwen3VLForConditionalGeneration,
                )
            elif model_type == "qwen3_vl_timeple":
                from transformers.models.qwen3_vl.modeling_qwen3_vl_timeple import (
                    Qwen3VLForConditionalGenerationWithTimePLECodec as CustomQwen3VLForConditionalGeneration,
                )
            elif model_type == "qwen3_vl_timeed":
                from transformers.models.qwen3_vl.modeling_qwen3_vl_timeed import (
                    Qwen3VLForConditionalGenerationWithTimeED as CustomQwen3VLForConditionalGeneration,
                )
            else:
                raise ValueError(f"Unsupported custom model_type for FSDP worker: {model_type}")

            if (not fsdp_config.enable_rank0_init) or self.device_mesh.get_local_rank("fsdp") == 0:
                model = CustomQwen3VLForConditionalGeneration.from_pretrained(
                    model_config.model_path,
                    config=self.model_config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    device_map="cpu" if fsdp_config.enable_rank0_init else "cuda",
                    low_cpu_mem_usage=True,
                    trust_remote_code=model_config.trust_remote_code,
                )
            else:
                with no_init_weights(), init_empty_weights():
                    model = CustomQwen3VLForConditionalGeneration.from_config(
                        self.model_config,
                        torch_dtype=torch_dtype,
                        attn_implementation="flash_attention_2",
                        trust_remote_code=model_config.trust_remote_code,
                    )
        else:
            if (not fsdp_config.enable_rank0_init) or self.device_mesh.get_local_rank("fsdp") == 0:
                model = AutoClass.from_pretrained(
                    model_config.model_path,
                    config=self.model_config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    device_map="cpu" if fsdp_config.enable_rank0_init else "cuda",
                    low_cpu_mem_usage=True,
                    trust_remote_code=model_config.trust_remote_code,
                )
            else:
                with no_init_weights(), init_empty_weights():
                    model = AutoClass.from_config(
                        self.model_config,
                        torch_dtype=torch_dtype,
                        attn_implementation="flash_attention_2",
                        trust_remote_code=model_config.trust_remote_code,
                    )

        model = cast(PreTrainedModel, model)  # lint
        model.tie_weights()  # avoid hanging

        if role == "ref":
            model.requires_grad_(False)

        is_lora_model = self._is_lora and role == "actor"
        if is_lora_model:
            self.print_rank0("Applying LoRA to actor module")
            model.enable_input_require_grads()
            if model_config.lora.target_modules == "all-linear":
                target_modules = model_config.lora.target_modules
            else:
                target_modules = [item.strip() for item in model_config.lora.target_modules.split(",") if item.strip()]

            lora_config = peft.LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=model_config.lora.rank,
                lora_alpha=model_config.lora.alpha,
                target_modules=target_modules,
                exclude_modules=model_config.lora.exclude_modules,
            )
            model = get_peft_model(model, lora_config)
            for p in model.parameters():
                if not p.requires_grad:
                    p.data = p.to(torch.bfloat16)
                else:
                    p.data = p.to(torch_dtype)
        else:
            model = model.to(torch_dtype)

        if model_config.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if model_config.freeze_vision_tower:
            if hasattr(model, "model") and hasattr(model.model, "visual"):  # transformers >= 4.52.0
                model.model.visual.requires_grad_(False)
                fsdp_config.use_orig_params = True
                self.print_rank0("Vision tower is set to not trainable.")
            elif hasattr(model, "visual"):  # transformers < 4.52.0
                model.visual.requires_grad_(False)
                fsdp_config.use_orig_params = True
                self.print_rank0("Vision tower is set to not trainable.")
            else:
                self.print_rank0("No vision tower found.")

        if model_config.freeze_cis_codec_encoder or model_config.freeze_cis_codec_decoder:
            cis_codec = getattr(model, "cis_codec", None)
            if cis_codec is None:
                self.print_rank0("No CIS codec found; skip CIS codec freeze config.")
            else:
                if model_config.freeze_cis_codec_encoder:
                    if hasattr(cis_codec, "encoder"):
                        cis_codec.encoder.requires_grad_(False)
                        fsdp_config.use_orig_params = True
                        self.print_rank0("CIS codec encoder is set to not trainable.")
                    else:
                        self.print_rank0("No CIS codec encoder found.")

                if model_config.freeze_cis_codec_decoder:
                    if hasattr(cis_codec, "decoder"):
                        cis_codec.decoder.requires_grad_(False)
                        fsdp_config.use_orig_params = True
                        self.print_rank0("CIS codec decoder is set to not trainable.")
                    else:
                        self.print_rank0("No CIS codec decoder found.")

        if model_config.freeze_timeple_codec_encoder or model_config.freeze_timeple_codec_decoder:
            timeple_codec = getattr(model, "timeple_codec", None)
            if timeple_codec is None:
                self.print_rank0("No TimePLE codec found; skip TimePLE codec freeze config.")
            else:
                if model_config.freeze_timeple_codec_encoder:
                    if hasattr(timeple_codec, "encoder"):
                        timeple_codec.encoder.requires_grad_(False)
                        fsdp_config.use_orig_params = True
                        self.print_rank0("TimePLE codec encoder is set to not trainable.")
                    else:
                        self.print_rank0("No TimePLE codec encoder found.")

                if model_config.freeze_timeple_codec_decoder:
                    if hasattr(timeple_codec, "decoder"):
                        timeple_codec.decoder.requires_grad_(False)
                        fsdp_config.use_orig_params = True
                        self.print_rank0("TimePLE codec decoder is set to not trainable.")
                    else:
                        self.print_rank0("No TimePLE codec decoder found.")

        if (
            model_config.freeze_timeple_codec_encoder
            or model_config.freeze_timeple_codec_decoder
        ):
            timeple_codec = getattr(model, "timeple_codec", None)
            if timeple_codec is None:
                self.print_rank0(
                    "No TimePLE codec found; skip TimePLE codec freeze config."
                )
            else:
                if model_config.freeze_timeple_codec_encoder:
                    if hasattr(timeple_codec, "encoder"):
                        timeple_codec.encoder.requires_grad_(False)
                        fsdp_config.use_orig_params = True
                        self.print_rank0(
                            "TimePLE codec encoder is set to not trainable."
                        )
                    else:
                        self.print_rank0("No TimePLE codec encoder found.")

                if model_config.freeze_timeple_codec_decoder:
                    if hasattr(timeple_codec, "decoder"):
                        timeple_codec.decoder.requires_grad_(False)
                        fsdp_config.use_orig_params = True
                        self.print_rank0(
                            "TimePLE codec decoder is set to not trainable."
                        )
                    else:
                        self.print_rank0("No TimePLE codec decoder found.")

        if model_config.freeze_timeed_encoder:
            timeed_encoder = getattr(model, "timeed_encoder", None)
            if timeed_encoder is None:
                self.print_rank0("No TimeED encoder found; skip TimeED encoder freeze config.")
            else:
                timeed_encoder.requires_grad_(False)
                fsdp_config.use_orig_params = True
                self.print_rank0("TimeED encoder is set to not trainable.")

        if model_config.freeze_timeed_decoder:
            timeed_decoder = getattr(model, "timeed_decoder", None)
            timeed_codebook = getattr(model, "timeed_codebook", None)
            if timeed_decoder is None:
                self.print_rank0("No TimeED decoder found; skip TimeED decoder freeze config.")
            else:
                timeed_decoder.requires_grad_(False)
                if timeed_codebook is not None:
                    timeed_codebook.requires_grad_(False)
                fsdp_config.use_orig_params = True
                self.print_rank0("TimeED decoder/codebook is set to not trainable.")

        trainable_param_tensors = 0
        frozen_param_tensors = 0
        for param in model.parameters():
            if param.requires_grad:
                trainable_param_tensors += 1
            else:
                frozen_param_tensors += 1

        if trainable_param_tensors > 0 and frozen_param_tensors > 0 and not fsdp_config.use_orig_params:
            fsdp_config.use_orig_params = True
            self.print_rank0(
                "Detected mixed requires_grad parameters "
                f"(trainable={trainable_param_tensors}, frozen={frozen_param_tensors}); "
                "set fsdp.use_orig_params=True for FSDP compatibility."
            )

        dist.barrier()
        print_model_size(model)
        print_gpu_memory_usage("After huggingface model init")
        mixed_precision = MixedPrecision(
            param_dtype=PrecisionType.to_dtype(fsdp_config.mp_param_dtype),
            reduce_dtype=PrecisionType.to_dtype(fsdp_config.mp_reduce_dtype),
            buffer_dtype=PrecisionType.to_dtype(fsdp_config.mp_buffer_dtype),
            cast_forward_inputs=True,
        )
        auto_wrap_policy = get_fsdp_wrap_policy(model, is_lora_model=is_lora_model)
        self.print_rank0(f"FSDP wrap policy: {auto_wrap_policy}.")

        if self.device_mesh.ndim == 2:
            if fsdp_config.enable_full_shard:
                sharding_strategy = ShardingStrategy.HYBRID_SHARD
            else:
                sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
        else:
            if fsdp_config.enable_full_shard:
                sharding_strategy = ShardingStrategy.FULL_SHARD
            else:
                sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

        if fsdp_config.enable_cpu_offload:
            cpu_offload = CPUOffload(offload_params=True)
        else:
            cpu_offload = None

        if fsdp_config.enable_rank0_init:
            sync_module_states = True
            param_init_fn = get_init_fn(model, device="cuda") if self.rank != 0 else None
        else:
            sync_module_states = False
            param_init_fn = None

        fsdp_module = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            param_init_fn=param_init_fn,
            device_id=torch.cuda.current_device(),
            sync_module_states=sync_module_states,
            forward_prefetch=False,
            use_orig_params=fsdp_config.use_orig_params,
            device_mesh=self.device_mesh,
        )
        print_gpu_memory_usage("After FSDP module init")

        if role in ["actor", "critic"]:
            self.fsdp_module = fsdp_module
            if optim_config.strategy == "adamw":
                self.optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                    fused=True,
                )
            elif optim_config.strategy == "adamw_bf16":
                self.optimizer = AnyPrecisionAdamW(
                    filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                )
            else:
                raise NotImplementedError(f"Optimizer {optim_config.strategy} not supported.")

            if optim_config.lr_warmup_steps is not None:
                num_warmup_steps = optim_config.lr_warmup_steps
            else:
                num_warmup_steps = int(optim_config.lr_warmup_ratio * optim_config.training_steps)

            if optim_config.lr_scheduler_type == "constant":
                self.lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer=self.optimizer, num_warmup_steps=num_warmup_steps
                )
            elif optim_config.lr_scheduler_type == "cosine":
                total_steps = optim_config.training_steps
                min_lr_ratio = optim_config.min_lr_ratio
                num_cycles = 0.5
                self.lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=self.optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps,
                    min_lr_ratio=min_lr_ratio,
                    num_cycles=num_cycles,
                )
            else:
                raise NotImplementedError(f"LR scheduler type {optim_config.lr_scheduler_type} is not supported")
            print_gpu_memory_usage("After optimizer init")
            if self._use_param_offload:
                offload_fsdp_model(self.fsdp_module)
                print_gpu_memory_usage(f"After offload {role} model during init")

            if self._use_optimizer_offload:
                offload_fsdp_optimizer(optimizer=self.optimizer)
                print_gpu_memory_usage(f"After offload {role} optimizer during init")
        else:
            self.ref_fsdp_module = fsdp_module
            if self._use_ref_param_offload:
                offload_fsdp_model(self.ref_fsdp_module)
                print_gpu_memory_usage(f"After offload {role} model during init")

    def _build_rollout(self) -> None:
        tp_size = self.config.rollout.tensor_parallel_size
        dp_size = self.world_size // tp_size
        if self.world_size % tp_size != 0:
            raise ValueError(f"rollout world size {self.world_size} is not divisible by tp size {tp_size}.")

        rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        lora_kwargs = (
            {"lora_kwargs": {"enable_lora": True, "max_loras": 1, "max_lora_rank": self._lora_rank}}
            if self._is_lora
            else {}
        )
        rollout_model_path = self.config.rollout.model_path or self.config.actor.model.model_path
        self.print_rank0(f"Rollout model path: {rollout_model_path}")
        self.rollout = vLLMRollout(
            model_path=rollout_model_path,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
            processor=self.processor,
            **lora_kwargs,
        )
        self.rollout_sharding_manager = FSDPVLLMShardingManager(
            module=self.fsdp_module,
            inference_engine=self.rollout.inference_engine,
            device_mesh=rollout_device_mesh,
            use_param_offload=self._use_param_offload,
        )
        print_gpu_memory_usage("After vllm init")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self._has_critic:
            self._build_model_optimizer(
                model_config=self.config.critic.model,
                fsdp_config=self.config.critic.fsdp,
                optim_config=self.config.critic.optim,
                padding_free=self.config.critic.padding_free,
                role="critic",
            )

        if self._has_actor:
            self._build_model_optimizer(
                model_config=self.config.actor.model,
                fsdp_config=self.config.actor.fsdp,
                optim_config=self.config.actor.optim,
                padding_free=self.config.actor.padding_free,
                role="actor",
            )

        if self._has_ref:
            if self._is_lora:
                self.ref_fsdp_module = self.fsdp_module
            else:
                self._build_model_optimizer(
                    model_config=self.config.actor.model,
                    fsdp_config=self.config.ref.fsdp,
                    optim_config=None,
                    padding_free=self.config.ref.padding_free,
                    role="ref",
                )

        if self._has_actor:
            from .actor.dp_actor import DataParallelPPOActor  # lazy import

            self.actor = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.fsdp_module,
                actor_optimizer=self.optimizer,
            )

        if self._has_critic:
            from .critic.dp_critic import DataParallelPPOCritic  # lazy import

            self.critic = DataParallelPPOCritic(
                config=self.config,
                critic_module=self.fsdp_module,
                critic_optimizer=self.optimizer,
            )

        if self._has_rollout:  # must after actor
            self._build_rollout()

        if self._has_ref:
            from .actor.dp_actor import DataParallelPPOActor  # lazy import

            self.ref_policy = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.ref_fsdp_module,
            )

        if self._has_actor or self._has_critic:
            self.flops_counter = FlopsCounter(self.model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.fsdp_module,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                processing_class=self.processor or self.tokenizer,
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path: str, save_model_only: bool = False):
        assert self._has_actor or self._has_critic
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.save_checkpoint(path, save_model_only)
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path: str):
        assert self._has_actor or self._has_critic
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.load_checkpoint(path)
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:  # avoid OOM in resuming
            offload_fsdp_optimizer(self.optimizer)

    def _process_multi_modal_inputs(self, data: DataProto):
        if "multi_modal_data" not in data.non_tensor_batch:
            return

        current_uid = data.non_tensor_batch.get("uid")
        cached_uid = self._cache.get("uid")
        if cached_uid is not None:
            if current_uid is None or not np.array_equal(np.asarray(current_uid, dtype=object), np.asarray(cached_uid, dtype=object)):
                self._cache.clear()

        if "multi_modal_inputs" not in self._cache:
            min_pixels = data.meta_info["min_pixels"]
            max_pixels = data.meta_info["max_pixels"]
            video_fps = data.meta_info["video_fps"]
            video_max_token_num = data.meta_info.get("video_max_token_num")
            fps_max_frames = data.meta_info.get("fps_max_frames")
            batch_multi_modal_inputs = []
            multi_modal_inputs_cache = {}  # avoid repeated processing for n > 1 samples
            for index, multi_modal_data in zip(
                data.non_tensor_batch["uid"], data.non_tensor_batch["multi_modal_data"]
            ):  # process multi modal data per sample
                if index not in multi_modal_inputs_cache:
                    images, videos = [], []
                    if "images" in multi_modal_data:
                        for image in multi_modal_data["images"]:
                            images.append(process_image(image, min_pixels, max_pixels))

                    if "videos" in multi_modal_data:
                        for video in multi_modal_data["videos"]:
                            videos.append(
                                process_video(
                                    video,
                                    min_pixels,
                                    max_pixels,
                                    video_fps,
                                    video_max_token_num=video_max_token_num,
                                    fps_max_frames=fps_max_frames,
                                )
                            )

                    if len(images) != 0:
                        # it's necessary to add `dict` to properly convert batch features to dict
                        # otherwise the batch features will be converted to dict keys
                        # see https://github.com/hiyouga/EasyR1/pull/339
                        multi_modal_inputs = dict(self.processor.image_processor(images=images, return_tensors="pt"))
                    elif len(videos) != 0:
                        if hasattr(self.processor, "video_processor"):
                            video_processor_kwargs = {"return_tensors": "pt", "do_sample_frames": False}
                            if video_max_token_num is not None:
                                video_processor_kwargs["frame_max_token"] = video_max_token_num
                                video_processor_kwargs["frame_token_only"] = True
                            if fps_max_frames is not None:
                                video_processor_kwargs["max_frames"] = fps_max_frames
                            multi_modal_inputs = dict(self.processor.video_processor(videos=videos, **video_processor_kwargs))
                        else:
                            multi_modal_inputs = dict(
                                self.processor.image_processor(images=None, videos=videos, return_tensors="pt")
                            )
                    else:
                        multi_modal_inputs = {}

                    multi_modal_inputs_cache[index] = multi_modal_inputs

                batch_multi_modal_inputs.append(multi_modal_inputs_cache[index])

            self._cache["uid"] = data.non_tensor_batch["uid"]
            self._cache["multi_modal_inputs"] = np.array(batch_multi_modal_inputs, dtype=object)

        data.non_tensor_batch["multi_modal_inputs"] = self._cache["multi_modal_inputs"]

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        assert self._has_actor

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu_actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / (promised_flops * self.world_size)
            )
            metrics["perf/max_memory_allocated_gb"] = (
                torch.cuda.max_memory_allocated() - self.rollout_sharding_manager.freed_bytes
            ) / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = (
                torch.cuda.max_memory_reserved() - self.rollout_sharding_manager.freed_bytes
            ) / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr
            self.lr_scheduler.step()

            # Metrics should be in non_tensor_batch instead of meta_info, as DataProto not concat meta_info
            output = DataProto(
                non_tensor_batch={
                    key: np.array([value] if np.isscalar(value) else value) for key, value in metrics.items()
                }
            )
            # Metrics do not need post processing since their batch size is 1

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prepare_rollout_engine(self):
        self.rollout_sharding_manager.load_vllm_and_sync_weights()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_rollout_engine(self):
        self.rollout_sharding_manager.offload_vllm()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        assert self._has_rollout

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        prompts = self.rollout_sharding_manager.preprocess_data(prompts)
        output = self.rollout.generate_sequences(prompts=prompts)
        output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_probs(self, data: DataProto):
        assert self._has_actor

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        # Recompute log_probs with the same sampling temperature that produced the responses.
        # Validation may override rollout.temperature via val_override_config.
        temperature = data.meta_info.get("temperature", self.config.rollout.temperature)
        data.meta_info["temperature"] = temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            old_log_probs, aux_tensors, aux_non_tensors = self.actor.compute_log_prob_and_aux(data=data)
            output_tensors = {"old_log_probs": old_log_probs, **aux_tensors}
            output = DataProto.from_dict(
                tensors=output_tensors,
                non_tensors=aux_non_tensors,
                meta_info={"temperature": temperature},
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.fsdp_module._handle.reshard(True)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_probs(self, data: DataProto):
        assert self._has_ref

        # when is_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        adapter_ctx = self.ref_fsdp_module.disable_adapter() if self._is_lora else nullcontext()

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        # the fsdp module is the same as the ref fsdp module when lora is enabled
        if self._use_ref_param_offload or (self._is_lora and self._use_param_offload):
            load_fsdp_model(self.ref_fsdp_module)

        temperature = data.meta_info.get("temperature", self.config.rollout.temperature)
        data.meta_info["temperature"] = temperature
        with self.ulysses_sharding_manager, adapter_ctx:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            ref_log_probs, ref_aux_tensors = self.ref_policy.compute_log_prob_and_timeed_span_ref(data=data)
            output = DataProto.from_dict(
                tensors={"ref_log_probs": ref_log_probs, **ref_aux_tensors},
                meta_info={"temperature": temperature},
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.ref_fsdp_module._handle.reshard(True)

        if self._use_ref_param_offload or (self._is_lora and self._use_param_offload):
            offload_fsdp_model(self.ref_fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        assert self._has_critic

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        assert self._has_critic

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu_critic"] = (
                estimated_flops * self.config.actor.ppo_epochs / (promised_flops * self.world_size)
            )

            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr

            # Metrics should be in non_tensor_batch instead of meta_info, as DataProto not concat meta_info
            output = DataProto(
                non_tensor_batch={
                    key: np.array([value] if np.isscalar(value) else value) for key, value in metrics.items()
                }
            )
            # Metrics do not need post processing since their batch size is 1

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        output = output.to("cpu")
        return output
