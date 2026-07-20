# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import yaml

from transformers import Seq2SeqTrainingArguments
from transformers.utils.versions import require_version

from swift.trainers import TrainerFactory
from swift.trainers.arguments import TrainArgumentsMixin
from swift.utils import (add_version_to_work_dir, get_device_count, get_logger, get_pai_tensorboard_dir, is_master,
                         is_mp, is_pai_training_job, is_swanlab_available, json_parse_to_dict)
from .base_args import BaseArguments, to_abspath
from .tuner_args import TunerArguments

logger = get_logger()


def _load_mapping_config(path: str) -> Dict[str, object]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        elif ext == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path} (expected .yaml/.yml/.json)")
    if data is None:
        raise ValueError(f"Empty config file: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a dict/mapping, got {type(data)}")
    return data


def _resolve_relative_path(base_dir: str, value: Optional[str]) -> Optional[str]:
    if value is None or not isinstance(value, str) or not value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(base_dir, value))


def _resolve_existing_relative_path(base_dir: str, value: Optional[str]) -> Optional[str]:
    if value is None or not isinstance(value, str) or not value:
        return value
    if os.path.isabs(value):
        return value
    if value.startswith('.') or value.startswith('..'):
        return os.path.normpath(os.path.join(base_dir, value))
    candidate = os.path.normpath(os.path.join(base_dir, value))
    if os.path.exists(candidate):
        return candidate
    return value


def _resolve_deepspeed_path(base_dir: str, value):
    if not isinstance(value, str) or not value:
        return value
    if value in {'zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload'}:
        return value
    stripped = value.lstrip()
    if stripped.startswith('{') or stripped.startswith('['):
        return value
    return _resolve_relative_path(base_dir, value)


def _resolve_training_args_paths(training_args: Dict[str, object], base_dir: str) -> Dict[str, object]:
    training_args = dict(training_args)
    for key in ('output_dir', 'logging_dir', 'resume_from_checkpoint'):
        if key in training_args and isinstance(training_args[key], str):
            training_args[key] = _resolve_relative_path(base_dir, training_args[key])
    if 'deepspeed' in training_args:
        training_args['deepspeed'] = _resolve_deepspeed_path(base_dir, training_args['deepspeed'])
    return training_args


def _normalize_logging_dir_for_output(logging_dir: Optional[str], output_dir: str) -> str:
    output_dir = to_abspath(output_dir)
    if not logging_dir:
        return os.path.join(output_dir, 'runs')

    logging_dir = to_abspath(logging_dir)
    # When users configure a shared ".../logs" root, isolate TensorBoard events per run.
    if os.path.basename(os.path.normpath(logging_dir)) == 'logs':
        logging_parent = os.path.dirname(os.path.normpath(logging_dir))
        try:
            rel_output = os.path.relpath(output_dir, logging_parent)
        except ValueError:
            rel_output = None
        if rel_output and rel_output != '.' and not rel_output.startswith('..'):
            return os.path.join(logging_dir, rel_output)
    return logging_dir


def _resolve_config_paths(config: Dict[str, object], base_dir: str) -> Dict[str, object]:
    config = dict(config)

    model_cfg = dict(config.get('model', {}) or {})
    for key in ('time_codec_config', 'time_codec_state_path', 'timeple_codec_config', 'timeple_codec_state_path'):
        if key in model_cfg and isinstance(model_cfg[key], str):
            model_cfg[key] = _resolve_relative_path(base_dir, model_cfg[key])
    cis_adapter_cfg = dict(model_cfg.get('timeple_interface_adapter', {}) or {})
    if cis_adapter_cfg:
        model_cfg['timeple_interface_adapter'] = cis_adapter_cfg
    config['model'] = model_cfg

    data_cfg = dict(config.get('data', {}) or {})
    for key in ('dataset_path', 'video_base_dir'):
        if key in data_cfg and isinstance(data_cfg[key], str):
            data_cfg[key] = _resolve_existing_relative_path(base_dir, data_cfg[key])
    if 'annotation_file' in data_cfg and isinstance(data_cfg['annotation_file'], str):
        data_cfg['annotation_file'] = _resolve_existing_relative_path(base_dir, data_cfg['annotation_file'])
    config['data'] = data_cfg

    training_cfg = dict(config.get('training', {}) or {})
    config['training'] = _resolve_training_args_paths(training_cfg, base_dir)

    if 'stages_config' in config and isinstance(config['stages_config'], str):
        config['stages_config'] = _resolve_relative_path(base_dir, config['stages_config'])

    stages = config.get('stages')
    if isinstance(stages, list):
        resolved_stages = []
        for stage in stages:
            if not isinstance(stage, dict):
                resolved_stages.append(stage)
                continue
            stage = dict(stage)
            if 'training_args' in stage and isinstance(stage['training_args'], dict):
                stage['training_args'] = _resolve_training_args_paths(stage['training_args'], base_dir)
            resolved_stages.append(stage)
        config['stages'] = resolved_stages

    return config


def _load_time_codec_config(path: str) -> Dict[str, object]:
    return _load_mapping_config(path)


def _load_timeple_codec_config(path: str) -> Dict[str, object]:
    return _load_mapping_config(path)


def _apply_config_sections(args: "TrainArguments", config: Dict[str, object]) -> None:
    model_cfg = config.get("model", {}) or {}
    data_cfg = config.get("data", {}) or {}
    training_cfg = config.get("training", {}) or {}

    for k, v in model_cfg.items():
        if not hasattr(args, k):
            raise ValueError(f"Unknown model arg in config: {k}")
        setattr(args, k, v)

    for k, v in data_cfg.items():
        if not hasattr(args, k):
            raise ValueError(f"Unknown data arg in config: {k}")
        setattr(args, k, v)

    # Training args use ms-swift TrainArguments fields directly.
    for k, v in training_cfg.items():
        if not hasattr(args, k):
            raise ValueError(f"Unknown training arg in config: {k}")
        setattr(args, k, v)


def _apply_time_codec_config(args: "TrainArguments", config: Dict[str, object], *, base_dir: Optional[str] = None) -> None:
    if base_dir is not None:
        config = _resolve_config_paths(config, base_dir)
    model_cfg = config.get("model", {}) or {}
    if "time_codec_config" in model_cfg and isinstance(model_cfg["time_codec_config"], str):
        model_cfg["time_codec_config"] = _load_time_codec_config(model_cfg["time_codec_config"])
    config = dict(config)
    config["model"] = model_cfg
    _apply_config_sections(args, config)
    if "stages" in config:
        args.time_codec_stages = config.get("stages")
    if "stages_config" in config:
        args.time_codec_stages_config = config.get("stages_config")


def _apply_timeple_codec_config(args: "TrainArguments", config: Dict[str, object], *, base_dir: Optional[str] = None) -> None:
    if base_dir is not None:
        # Public TimePLE configs intentionally use paths relative to the project
        # root (for example ``./data`` and ``./checkpoints``).  The launchers set
        # TIMEPLE_ROOT so an installed ms-swift overlay does not accidentally
        # resolve those paths relative to ``configs/sft``.
        project_root = os.environ.get("TIMEPLE_ROOT", base_dir)
        config = _resolve_config_paths(config, os.path.abspath(project_root))
    model_cfg = config.get("model", {}) or {}
    if "timeple_codec_config" in model_cfg and isinstance(model_cfg["timeple_codec_config"], str):
        model_cfg["timeple_codec_config"] = _load_timeple_codec_config(model_cfg["timeple_codec_config"])
    config = dict(config)
    config["model"] = model_cfg
    _apply_config_sections(args, config)


def _apply_qwen3_vl_config(args: "TrainArguments", config: Dict[str, object], *, base_dir: Optional[str] = None) -> None:
    if base_dir is not None:
        config = _resolve_config_paths(config, base_dir)
    _apply_config_sections(args, config)


@dataclass
class Seq2SeqTrainingOverrideArguments(TrainArgumentsMixin, Seq2SeqTrainingArguments):
    """Override the default value in `Seq2SeqTrainingArguments`"""
    output_dir: Optional[str] = None
    learning_rate: Optional[float] = None
    eval_strategy: Optional[str] = None  # steps, epoch
    fp16: Optional[bool] = None
    bf16: Optional[bool] = None
    preserve_checkpoint_snapshots: bool = False
    train_metric_for_best_model: Optional[str] = 'train_loss'
    train_greater_is_better: Optional[bool] = None
    last_checkpoint_snapshot_name: str = 'checkpoint-last'
    best_checkpoint_snapshot_name: str = 'checkpoint-val-best'
    train_best_checkpoint_snapshot_name: str = 'checkpoint-train-best'

    def _init_output_dir(self):
        if self.output_dir is None:
            self.output_dir = f'output/{self.model_suffix}'
        self.output_dir = to_abspath(self.output_dir)

    def _init_eval_strategy(self):
        if self.eval_strategy is None:
            self.eval_strategy = self.save_strategy
        if self.eval_strategy == 'no':
            self.eval_steps = None
            if self.split_dataset_ratio > 0:
                self.split_dataset_ratio = 0.
                logger.info(f'Setting args.split_dataset_ratio: {self.split_dataset_ratio}')
        elif self.eval_strategy == 'steps' and self.eval_steps is None:
            self.eval_steps = self.save_steps
        self.evaluation_strategy = self.eval_strategy

    def _init_metric(self):
        if self.metric is None and self.predict_with_generate:
            self.metric = 'nlg'
        if self.metric_for_best_model is None:
            self.metric_for_best_model = 'rouge-l' if self.predict_with_generate else 'loss'
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = 'loss' not in self.metric_for_best_model
        if self.train_greater_is_better is None and self.train_metric_for_best_model is not None:
            self.train_greater_is_better = 'loss' not in self.train_metric_for_best_model

    def __post_init__(self):
        self._init_output_dir()
        self._init_metric()

        if self.learning_rate is None:
            if self.train_type == 'full':
                self.learning_rate = 1e-5
            else:
                self.learning_rate = 1e-4
        self._init_eval_strategy()


@dataclass
class SwanlabArguments:

    swanlab_token: Optional[str] = None
    swanlab_project: Optional[str] = None
    swanlab_workspace: Optional[str] = None
    swanlab_exp_name: Optional[str] = None
    swanlab_lark_webhook_url: Optional[str] = None
    swanlab_lark_secret: Optional[str] = None
    swanlab_mode: Literal['cloud', 'local'] = 'cloud'

    def _init_swanlab(self):
        if not is_swanlab_available():
            raise ValueError('You are using swanlab as `report_to`, please install swanlab by ' '`pip install swanlab`')
        if not self.swanlab_exp_name:
            self.swanlab_exp_name = self.output_dir
        from transformers.integrations import INTEGRATION_TO_CALLBACK
        import swanlab
        from swanlab.integration.transformers import SwanLabCallback
        if self.swanlab_token:
            swanlab.login(self.swanlab_token)

        if self.swanlab_lark_webhook_url is not None:
            from swanlab.plugin.notification import LarkCallback
            lark_callback = LarkCallback(
                webhook_url=self.swanlab_lark_webhook_url,
                secret=self.swanlab_lark_secret,
            )
            swanlab.register_callbacks([lark_callback])

        INTEGRATION_TO_CALLBACK['swanlab'] = SwanLabCallback(
            project=self.swanlab_project,
            workspace=self.swanlab_workspace,
            experiment_name=self.swanlab_exp_name,
            config={'UPPERFRAME': 'ms-swift'},
            mode=self.swanlab_mode,
        )


@dataclass
class TrainArguments(SwanlabArguments, TunerArguments, BaseArguments, Seq2SeqTrainingOverrideArguments):
    """
    TrainArguments class is a dataclass that inherits from multiple argument classes:
    TunerArguments, Seq2SeqTrainingOverrideArguments, and BaseArguments.

    Args:
        add_version (bool): Flag to add version information to output_dir. Default is True.
        max_new_tokens (int): Maximum number of new tokens to generate. Default is 64.
        temperature (float): Temperature for sampling. Default is 0.
    """
    add_version: bool = True
    create_checkpoint_symlink: bool = False

    # extra
    max_new_tokens: int = 64
    temperature: float = 0.
    load_args: bool = False

    # time codec (Qwen3-VL)
    use_time_codec: bool = False
    use_timeple_codec: bool = False
    dataset_path: Optional[str] = None
    video_base_dir: Optional[str] = None
    annotation_file: Optional[str] = None
    start_idx: int = 0
    end_idx: Optional[int] = None
    sample_fps: float = 2.0
    video_load_backend: Optional[str] = None
    max_frames: int = 64
    min_frames: int = 4
    total_pixels: Optional[int] = None
    min_pixels: Optional[int] = None
    frame_min_token: Optional[int] = None
    frame_max_token: Optional[int] = None
    frame_token_only: bool = False
    video_timestamp_text_interleave: bool = False
    cis_timestamp_interleave: bool = False
    max_timespans_per_sample: int = 0
    use_cot_thinking: bool = False
    time_codec_stages_config: Optional[str] = None
    time_codec_start_stage: int = 0
    time_codec_config_file: Optional[str] = None
    time_codec_stages: Optional[List[Dict[str, object]]] = None
    qwen3_vl_config_file: Optional[str] = None
    timeple_codec_config_file: Optional[str] = None

    # qwen_vl_utils overrides (set via YAML, applied to env before model load)
    max_ratio: Optional[int] = None
    frame_factor: Optional[int] = None
    fps: Optional[float] = None
    fps_min_frames: Optional[int] = None
    fps_max_frames: Optional[int] = None
    image_max_token_num: Optional[int] = None
    image_min_token_num: Optional[int] = None
    spatial_merge_size: Optional[int] = None
    video_max_token_num: Optional[int] = None
    video_min_token_num: Optional[int] = None

    use_time_codec_projection: bool = False
    time_codec_projection_type: str = "linear"
    time_codec_projection_hidden_dim: Optional[int] = None
    time_codec_projection_activation: str = "gelu"
    time_codec_projection_identity_init: bool = False
    time_codec_projection_bias: bool = True
    time_codec_config: Optional[object] = None
    time_codec_state_path: Optional[str] = None
    freeze_time_codec: bool = False
    freeze_time_codec_encoders: bool = False
    freeze_time_codec_decoders: bool = False
    use_timeple_interface_adapter: bool = False
    timeple_interface_adapter: Optional[Dict[str, object]] = None
    timeple_codec_config: Optional[object] = None
    timeple_codec_state_path: Optional[str] = None
    freeze_timeple_codec: bool = False
    freeze_timeple_codec_encoder: bool = False
    freeze_timeple_codec_decoder: bool = False
    freeze_vision: Optional[bool] = None
    freeze_language: Optional[bool] = None
    freeze_lm_head: Optional[bool] = None

    lm_loss_weight: float = 1.0
    time_loss_weight: float = 1.0
    compute_time_loss: bool = True
    time_decode_loss_weight: float = 1.0
    time_codec_recon_loss_weight: float = 0.0
    time_embedding_loss_weight: float = 0.0
    time_embedding_cosine_loss_weight: float = 0.0
    time_iou_loss_weight: float = 0.0
    timeple_loss_weight: float = 1.0
    compute_timeple_loss: bool = True
    timeple_decode_loss_weight: float = 1.0
    timeple_dfl_loss_weight: Optional[float] = None
    timeple_iou_loss_weight: Optional[float] = None
    timeple_boundary_loss_weight: Optional[float] = None
    timeple_codec_recon_loss_weight: float = 0.0
    timeple_embedding_loss_weight: float = 0.0
    timeple_embedding_cosine_loss_weight: float = 0.0
    timeple_reencoding_loss_weight: float = 0.0

    # zero++
    zero_hpz_partition_size: Optional[int] = None

    # auto_tp
    deepspeed_autotp_size: Optional[int] = None

    # early_step
    early_stop_interval: Optional[int] = None

    def _check_padding_free(self):
        if self.padding_free or self.packing:
            if self.packing:
                feature = 'packing'
                self.padding_free = True
            else:
                feature = 'padding_free'
            if self.attn_impl not in {'flash_attn', 'flash_attention_2', 'flash_attention_3'}:
                raise ValueError(f'The "{feature}" feature requires a flash attention implementation. '
                                 'Please use one of: "flash_attn", "flash_attention_2", "flash_attention_3".')

    def __post_init__(self) -> None:
        if self.time_codec_config_file:
            self.use_time_codec = True
        if self.timeple_codec_config_file:
            self.use_timeple_codec = True

        if self.use_time_codec and self.use_timeple_codec:
            raise ValueError("`use_time_codec` and `use_timeple_codec` cannot both be enabled.")
        if self.qwen3_vl_config_file and (self.use_time_codec or self.use_timeple_codec):
            raise ValueError(
                "`qwen3_vl_config_file` cannot be combined with time-codec or "
                "TimePLE config files."
            )

        if self.qwen3_vl_config_file:
            self.qwen3_vl_config_file = to_abspath(self.qwen3_vl_config_file, True)
            config = _load_mapping_config(self.qwen3_vl_config_file)
            _apply_qwen3_vl_config(self, config, base_dir=os.path.dirname(self.qwen3_vl_config_file))
            if self.dataset_path and not self.dataset:
                self.dataset = [self.dataset_path]
            if self.model_type is None:
                from swift.llm.model.constant import MLLMModelType
                self.model_type = MLLMModelType.qwen3_vl
            if self.freeze_vision is not None:
                self.freeze_vit = self.freeze_vision
            if self.freeze_language is not None:
                self.freeze_llm = self.freeze_language
            if self.freeze_lm_head:
                self.freeze_parameters.append("lm_head")

        if self.use_time_codec and self.time_codec_config_file:
            self.time_codec_config_file = to_abspath(self.time_codec_config_file, True)
            config = _load_time_codec_config(self.time_codec_config_file)
            _apply_time_codec_config(self, config, base_dir=os.path.dirname(self.time_codec_config_file))

        if self.use_timeple_codec and self.timeple_codec_config_file:
            self.timeple_codec_config_file = to_abspath(self.timeple_codec_config_file, True)
            config = _load_timeple_codec_config(self.timeple_codec_config_file)
            _apply_timeple_codec_config(self, config, base_dir=os.path.dirname(self.timeple_codec_config_file))

        if self.use_time_codec:
            if self.dataset_path and not self.dataset:
                self.dataset = [self.dataset_path]
            if self.model_type is None:
                from swift.llm.model.constant import MLLMModelType
                self.model_type = MLLMModelType.qwen3_vl_time_codec
            if self.freeze_vision is None:
                self.freeze_vision = False
            if self.freeze_language is None:
                self.freeze_language = False
            if self.time_codec_stages_config:
                self.time_codec_stages_config = to_abspath(self.time_codec_stages_config, True)

        if self.use_timeple_codec:
            if self.dataset_path and not self.dataset:
                self.dataset = [self.dataset_path]
            from swift.llm.model.constant import MLLMModelType
            if self.model_type is None:
                self.model_type = MLLMModelType.qwen3_vl_timeple
            elif self.model_type == MLLMModelType.qwen2_5_vl_timeple:
                self.model_type = MLLMModelType.qwen2_5_vl_timeple
            if self.freeze_vision is None:
                self.freeze_vision = False
            if self.freeze_language is None:
                self.freeze_language = False

        # Apply qwen_vl_utils-related overrides from YAML into env vars (used by patch_qwen_vl_utils).
        fps_value = self.fps if self.fps is not None else self.sample_fps
        fps_min_frames_value = self.fps_min_frames if self.fps_min_frames is not None else self.min_frames
        fps_max_frames_value = self.fps_max_frames if self.fps_max_frames is not None else self.max_frames
        env_overrides = {
            "MAX_RATIO": self.max_ratio,
            "FRAME_FACTOR": self.frame_factor,
            "FPS": fps_value,
            "FPS_MIN_FRAMES": fps_min_frames_value,
            "FPS_MAX_FRAMES": fps_max_frames_value,
            "IMAGE_MAX_TOKEN_NUM": self.image_max_token_num,
            "IMAGE_MIN_TOKEN_NUM": self.image_min_token_num,
            "SPATIAL_MERGE_SIZE": self.spatial_merge_size,
            "VIDEO_MAX_TOKEN_NUM": self.video_max_token_num,
            "VIDEO_MIN_TOKEN_NUM": self.video_min_token_num,
        }
        for key, value in env_overrides.items():
            if value is None:
                continue
            os.environ[key] = str(value)

        if self.resume_from_checkpoint:
            self.resume_from_checkpoint = to_abspath(self.resume_from_checkpoint, True)
            # The non-resume_only_model will have its weights loaded in the trainer.
            if self.resume_only_model:
                if self.train_type == 'full':
                    self.model = self.resume_from_checkpoint
                else:
                    self.adapters = [self.resume_from_checkpoint]
        BaseArguments.__post_init__(self)
        Seq2SeqTrainingOverrideArguments.__post_init__(self)
        TunerArguments.__post_init__(self)
        self._check_padding_free()
        if self.optimizer is None:
            if self.lorap_lr_ratio:
                self.optimizer = 'lorap'
            elif self.use_galore:
                self.optimizer = 'galore'

        if len(self.dataset) == 0 and len(self.cached_dataset) == 0:
            raise ValueError(f'self.dataset: {self.dataset}, self.cached_dataset: {self.cached_dataset}. '
                             'Please input the training dataset.')

        self._handle_pai_compat()

        self._init_deepspeed()
        self._init_device()

        if getattr(self, 'accelerator_config', None) is None:
            self.accelerator_config = {'dispatch_batches': False}
        if self.split_dataset_ratio == 0 and not self.val_dataset and not self.eval_dataset:
            self.eval_strategy = 'no'
        self.training_args = TrainerFactory.get_training_args(self)
        self.training_args.remove_unused_columns = False
        self._add_version()

        if 'swanlab' in self.report_to:
            self._init_swanlab()

    def _init_deepspeed(self):
        if self.deepspeed:
            require_version('deepspeed')
            if is_mp():
                raise ValueError('DeepSpeed is not compatible with `device_map`. '
                                 f'n_gpu: {get_device_count()}, '
                                 f'local_world_size: {self.local_world_size}.')

            ds_config_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ds_config'))
            deepspeed_mapping = {
                name: f'{name}.json'
                for name in ['zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload']
            }
            for ds_name, ds_config in deepspeed_mapping.items():
                if self.deepspeed == ds_name:
                    self.deepspeed = os.path.join(ds_config_folder, ds_config)
                    break

            self.deepspeed = json_parse_to_dict(self.deepspeed)
            if self.zero_hpz_partition_size is not None:
                assert 'zero_optimization' in self.deepspeed
                self.deepspeed['zero_optimization']['zero_hpz_partition_size'] = self.zero_hpz_partition_size
                logger.warn('If `zero_hpz_partition_size`(ZeRO++) causes grad_norm NaN, please'
                            ' try `--torch_dtype float16`')
            if self.deepspeed_autotp_size is not None:
                assert self.deepspeed is not None, (
                    'To use `deepspeed_autotp_size`, you need to additionally set the `--deepspeed` argument.')
                self.deepspeed['tensor_parallel'] = {'autotp_size': self.deepspeed_autotp_size}
                self.deepspeed['zero_optimization']['gather_16bit_weights_on_model_save'] = True
            logger.info(f'Using deepspeed: {self.deepspeed}')

    def _handle_pai_compat(self) -> None:
        if not is_pai_training_job():
            return

        logger.info('Handle pai compat...')
        pai_tensorboard_dir = get_pai_tensorboard_dir()
        if self.logging_dir is None and pai_tensorboard_dir is not None:
            self.logging_dir = pai_tensorboard_dir
            logger.info(f'Setting args.logging_dir: {self.logging_dir}')
        self.add_version = False
        logger.info(f'Setting args.add_version: {self.add_version}')

    def _add_version(self):
        """Prepare the output_dir"""
        if self.add_version:
            self.output_dir = add_version_to_work_dir(self.output_dir)
            logger.info(f'output_dir: {self.output_dir}')

        self.logging_dir = _normalize_logging_dir_for_output(self.logging_dir, self.output_dir)
        self.logging_dir = to_abspath(self.logging_dir)
        if is_master():
            os.makedirs(self.output_dir, exist_ok=True)

        if self.run_name is None:
            self.run_name = self.output_dir

        self.training_args.output_dir = self.output_dir
        self.training_args.run_name = self.run_name
        self.training_args.logging_dir = self.logging_dir
