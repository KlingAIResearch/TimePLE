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
Actor config
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple


@dataclass
class LoraConfig:
    rank: int = 0
    alpha: int = 64
    target_modules: str = "all-linear"
    exclude_modules: Optional[str] = None

    def post_init(self):
        if not isinstance(self.target_modules, str):
            raise TypeError("lora.target_modules must be a string like 'all-linear' or 'q_proj,k_proj,v_proj,o_proj'.")

        self.target_modules = self.target_modules.strip()
        if self.exclude_modules is not None:
            if not isinstance(self.exclude_modules, str):
                raise TypeError("lora.exclude_modules must be a string like '.*visual.*'.")

            self.exclude_modules = self.exclude_modules.strip()


@dataclass
class ModelConfig:
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    override_config: dict[str, Any] = field(default_factory=dict)
    enable_gradient_checkpointing: bool = True
    trust_remote_code: bool = True
    freeze_vision_tower: bool = False
    freeze_cis_codec_encoder: bool = False
    freeze_cis_codec_decoder: bool = False
    freeze_timeple_codec_encoder: bool = False
    freeze_timeple_codec_decoder: bool = False
    freeze_timeple_codec_encoder: bool = False
    freeze_timeple_codec_decoder: bool = False
    freeze_timeed_encoder: bool = False
    freeze_timeed_decoder: bool = False
    lora: LoraConfig = field(default_factory=LoraConfig)

    def post_init(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path

        if self.model_path is not None and os.path.exists(self.model_path):  # ray job uses absolute path
            self.model_path = os.path.abspath(self.model_path)

        if self.tokenizer_path is not None and os.path.exists(self.tokenizer_path):
            self.tokenizer_path = os.path.abspath(self.tokenizer_path)


@dataclass
class OptimConfig:
    lr: float = 1e-6
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    strategy: str = "adamw"
    lr_warmup_ratio: float = 0.0
    lr_warmup_steps: Optional[int] = None
    min_lr_ratio: Optional[float] = None
    lr_scheduler_type: str = "constant"
    # below are auto keys
    training_steps: int = field(default=-1, init=False)


@dataclass
class FSDPConfig:
    enable_full_shard: bool = True
    enable_cpu_offload: bool = False
    enable_rank0_init: bool = True
    use_orig_params: bool = False
    torch_dtype: Optional[str] = None
    fsdp_size: int = -1
    mp_param_dtype: str = "bf16"
    mp_reduce_dtype: str = "fp32"
    mp_buffer_dtype: str = "fp32"


@dataclass
class OffloadConfig:
    offload_params: bool = False
    offload_optimizer: bool = False


@dataclass
class ActorConfig:
    strategy: str = "fsdp"
    global_batch_size: int = 256
    """number of samples per minibatch for updating actor"""
    micro_batch_size_per_device_for_update: int = 4
    """number of samples per forward pass for updating actor"""
    micro_batch_size_per_device_for_experience: int = 16
    """number of samples per forward pass for computing log probs"""
    max_grad_norm: float = 1.0
    """number to clip grad norm"""
    clip_ratio_low: float = 0.2
    """clip ratio in PPO & DAPO"""
    clip_ratio_high: float = 0.3
    """clip ratio in PPO & DAPO"""
    clip_ratio_dual: float = 3.0
    """constant C in dual-clip PPO, clips when advantage < -C"""
    loss_avg_mode: str = "token"
    """loss average mode: `token`, `seq`"""
    loss_type: str = "default"
    """loss type: `default`, `gspo`, `cispo`"""
    ppo_epochs: int = 1
    """number of ppo epochs for each rollout batch"""
    padding_free: bool = True
    """use padding-free training"""
    dynamic_batching: bool = True
    """enable dynamic batching"""
    ulysses_size: int = 1
    """ulysses sequence parallel size"""
    use_torch_compile: bool = True
    """enable torch compile"""
    tau_positive: float = 1.0
    """temperature for positive tokens"""
    tau_negative: float = 1.05
    """temperature for negative tokens"""
    cis_aux_loss_weight: float = 0.0
    """outer weight for differentiable CIS regression loss on generated TIMESPAN tokens"""
    cis_aux_decode_loss_weight: float = 0.2
    """weight for CIS decoded interval loss inside the auxiliary objective"""
    cis_aux_codec_recon_loss_weight: float = 0.0
    """weight for CIS codec reconstruction loss inside the auxiliary objective"""
    cis_aux_embedding_loss_weight: float = 0.5
    """weight for CIS latent MSE loss inside the auxiliary objective"""
    cis_aux_embedding_cosine_loss_weight: float = 0.5
    """weight for CIS latent cosine loss inside the auxiliary objective"""
    cis_aux_reencoding_loss_weight: float = 0.1
    """weight for CIS re-encoding loss inside the auxiliary objective"""
    timeed_span_grpo_enabled: bool = False
    """enable TimeED canonical-cell span policy optimization"""
    timeed_span_samples_per_response: int = 1
    """legacy expansion knob; enumerated TimeED span policy uses one text row per response"""
    timeed_span_loss_weight: float = 1.0
    """outer weight for exact canonical-cell span GRPO loss"""
    timeed_span_logprob_weight: float = 1.0
    """legacy sampled-cell log-prob scale kept for config compatibility"""
    timeed_text_reward_aggregation: str = "mean"
    """legacy sampled-cell text reward aggregation kept for config compatibility"""
    timeed_span_kl_coef: float = 1e-2
    """weight for exact categorical KL between current and reference TimeED span distributions"""
    timeed_span_aux_loss_weight: float = 1e-2
    """outer weight for DFL/GIoU expectation auxiliary stabilizer"""
    timeed_span_aux_dfl_weight: float = 1.0
    """DFL component weight inside TimeED auxiliary loss"""
    timeed_span_aux_giou_weight: float = 1.0
    """GIoU component weight inside TimeED auxiliary loss"""
    timeed_span_pref_loss_weight: float = 0.05
    """outer weight for canonical-cell span preference loss"""
    timeed_span_pref_beta: float = 1.0
    """DPO-style span preference inverse temperature"""
    timeed_span_pref_delta: float = 0.05
    """minimum reward gap required to form a span preference pair"""
    timeed_span_reward_iou_weight: float = 0.8
    """IoU weight for cell-level TimeED reward map"""
    timeed_span_reward_boundary_weight: float = 0.2
    """boundary reward weight for cell-level TimeED reward map"""
    timeed_span_reward_boundary_tau: float = 10.0
    """boundary reward temperature in seconds"""
    timeed_span_advantage_eps: float = 1e-6
    """epsilon for old-policy reward standard deviation in enumerated span policy loss"""
    timeed_span_pref_negative_reward_threshold: float = 0.2
    """hard negative reward threshold for TimeED span preference loss"""
    csdo_enabled: bool = False
    """enable independent TimePLE Counterfactual Span Distribution Optimization (CSDO) loss"""
    csdo_loss_weight: float = 1.0
    """outer weight for TimePLE Counterfactual Span Distribution Optimization (CSDO) cross-entropy loss"""
    csdo_eta: float = 0.05
    """counterfactual mass-shift coefficient for TimePLE Counterfactual Span Distribution Optimization (CSDO) target construction"""
    csdo_tau: float = 0.5
    """temperature for reward-improved TimePLE Counterfactual Span Distribution Optimization (CSDO) target distribution"""
    csdo_adv_norm: bool = True
    """normalize Counterfactual Span Distribution Optimization (CSDO) reward improvements over canonical cells per sample"""
    csdo_adv_clip: float = 5.0
    """clip normalized Counterfactual Span Distribution Optimization (CSDO) reward improvements before target softmax"""
    csdo_min_adv_std: float = 1.0e-4
    """minimum per-sample Counterfactual Span Distribution Optimization (CSDO) advantage std required to use normalized improvements"""
    csdo_reward_type: str = "iou_boundary"
    """reward used to construct Counterfactual Span Distribution Optimization (CSDO) target distribution: `iou` or `iou_boundary`"""
    csdo_boundary_weight: float = 0.1
    """boundary penalty coefficient inside TimePLE Counterfactual Span Distribution Optimization (CSDO) target reward"""
    csdo_use_ref_kl: bool = True
    """enable TimePLE Counterfactual Span Distribution Optimization (CSDO) reference span-distribution KL"""
    csdo_span_kl_coef: float = 1.0e-2
    """independent KL coefficient for TimePLE Counterfactual Span Distribution Optimization (CSDO) span distribution"""
    tr_spd_enabled: bool = False
    """enable independent TimePLE Trust-Region Span Posterior Distillation (TR-SPD) loss"""
    tr_spd_loss_weight: float = 1.0
    """outer weight for verified posterior span distillation"""
    tr_spd_tau_candidates: Tuple[float, ...] = (1.0, 1.5, 2.0)
    """offline-selected posterior temperatures"""
    tr_spd_support_mode: str = "none"
    """posterior support mode: `none`, `gaussian_95`, or `gaussian_99`"""
    tr_spd_accept_delta: float = 0.0
    """minimum decoded reward improvement required to accept posterior target"""
    tr_spd_trust_region_kl_budget: Optional[float] = None
    """optional rho budget for KL(q_tau || p_old) posterior trust-region selection"""
    tr_spd_rejected_retention_weight: float = 0.1
    """weak KL(p_old || p_current) weight for rejected samples"""
    tr_spd_use_improvement_weight: bool = True
    """scale accepted posterior distillation by decoded reward improvement"""
    tr_spd_improvement_gamma: float = 0.5
    """accepted-sample improvement weighting strength"""
    tr_spd_improvement_scale: float = 0.05
    """normalizer for decoded reward improvement weighting"""
    tr_spd_improvement_max_extra_weight: float = 2.0
    """maximum clipped extra improvement weight before gamma scaling"""
    tr_spd_reward_type: str = "iou"
    """decoded reward used for Trust-Region Span Posterior Distillation (TR-SPD) quality filtering: `iou` or `iou_boundary`"""
    tr_spd_boundary_weight: float = 0.0
    """boundary penalty coefficient for `iou_boundary` reward"""
    tr_spd_text_pg_loss_weight: float = 0.0
    """weight for EasyR1 response-level PG loss when Trust-Region Span Posterior Distillation (TR-SPD) is enabled"""
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    # below are auto keys
    global_batch_size_per_device: int = field(default=-1, init=False)
    disable_kl: bool = field(default=False, init=False)
    use_kl_loss: bool = field(default=False, init=False)
    kl_penalty: str = field(default="kl", init=False)
    kl_coef: float = field(default=0.0, init=False)

    def post_init(self):
        if self.csdo_enabled:
            if not self.model.freeze_timeple_codec_decoder:
                raise ValueError(
                    "csdo_enabled=True requires "
                    "worker.actor.model.freeze_timeple_codec_decoder=true."
                )
            if self.timeed_span_grpo_enabled:
                raise ValueError(
                    "csdo_enabled=True must not be combined with timeed_span_grpo_enabled=True."
                )
            if float(getattr(self, "timeed_span_kl_coef", 0.0)) > 0.0:
                warnings.warn(
                    "timeed_span_kl_coef is ignored by TimePLE Counterfactual Span Distribution Optimization (CSDO); "
                    "use csdo_span_kl_coef for Counterfactual Span Distribution Optimization (CSDO) span KL.",
                    stacklevel=2,
                )
        if self.tr_spd_enabled:
            if not self.model.freeze_timeple_codec_decoder:
                raise ValueError(
                    "tr_spd_enabled=True requires "
                    "worker.actor.model.freeze_timeple_codec_decoder=true."
                )
            if self.csdo_enabled:
                raise ValueError("tr_spd_enabled=True must not be combined with csdo_enabled=True.")
            if self.timeed_span_grpo_enabled:
                raise ValueError("tr_spd_enabled=True must not be combined with timeed_span_grpo_enabled=True.")
            if len(tuple(self.tr_spd_tau_candidates)) == 0:
                raise ValueError("tr_spd_tau_candidates must not be empty.")
            if self.tr_spd_trust_region_kl_budget is not None:
                if float(self.tr_spd_trust_region_kl_budget) <= 0.0:
                    raise ValueError("tr_spd_trust_region_kl_budget must be positive when set.")
            if str(self.tr_spd_support_mode).replace("-", "_") not in {
                "none",
                "gaussian_95",
                "gaussian_99",
            }:
                raise ValueError(
                    "tr_spd_support_mode must be one of: none, gaussian_95, gaussian_99."
                )


@dataclass
class RefConfig:
    strategy: str = "fsdp"
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    # below are auto keys
    micro_batch_size_per_device_for_experience: int = field(default=-1, init=False)
    padding_free: bool = field(default=False, init=False)
    dynamic_batching: bool = field(default=False, init=False)
    ulysses_size: int = field(default=1, init=False)
    use_torch_compile: bool = field(default=True, init=False)
