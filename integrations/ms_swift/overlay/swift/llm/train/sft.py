# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from functools import partial
from typing import List, Optional, Union

from datasets import Dataset as HfDataset
from datasets import load_from_disk

from swift.llm.dataset.loader import DatasetLoader
from swift.plugin import extra_callbacks
from swift.trainers import TimePLECodecTrainer, TrainerFactory
from swift.utils import (append_to_jsonl, get_logger, get_model_parameter_info, is_master, parse_args, plot_images,
                         stat_array)
from ..argument import TrainArguments
from ..base import SwiftPipeline
from ..dataset import EncodePreprocessor, IterablePackingDataset, LazyLLMDataset, PackingDataset, load_dataset
from ..infer import prepare_generation_config
from .tuner import TunerMixin

logger = get_logger()


def _set_requires_grad(module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def _apply_timeple_codec_freeze(model, args: TrainArguments) -> None:
    freeze_vision = args.freeze_vision if args.freeze_vision is not None else args.freeze_vit
    freeze_language = args.freeze_language if args.freeze_language is not None else args.freeze_llm
    freeze_lm_head = args.freeze_lm_head if args.freeze_lm_head is not None else False

    if hasattr(model, 'visual') and getattr(model, 'visual') is not None:
        _set_requires_grad(model.visual, not freeze_vision)

    language_model = None
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        language_model = model.model.language_model
    elif hasattr(model, 'language_model'):
        language_model = model.language_model
    if language_model is not None:
        _set_requires_grad(language_model, not freeze_language)

    if hasattr(model, 'lm_head') and getattr(model, 'lm_head') is not None:
        _set_requires_grad(model.lm_head, not freeze_lm_head)

    if hasattr(model, 'timeple_codec') and getattr(model, 'timeple_codec') is not None:
        if args.freeze_timeple_codec:
            _set_requires_grad(model.timeple_codec, False)
        else:
            _set_requires_grad(model.timeple_codec, True)
            if args.freeze_timeple_codec_encoder and hasattr(model.timeple_codec, 'encoder'):
                _set_requires_grad(model.timeple_codec.encoder, False)
        if args.freeze_timeple_codec_decoder and hasattr(model.timeple_codec, 'decoder'):
            _set_requires_grad(model.timeple_codec.decoder, False)


def _get_timeple_codec_trainer_kwargs(args: TrainArguments):
    return {
        'lm_loss_weight': args.lm_loss_weight,
        'timeple_loss_weight': args.timeple_loss_weight,
        'compute_timeple_loss': args.compute_timeple_loss,
        'timeple_decode_loss_weight': args.timeple_decode_loss_weight,
        'timeple_dfl_loss_weight': args.timeple_dfl_loss_weight,
        'timeple_iou_loss_weight': args.timeple_iou_loss_weight,
        'timeple_boundary_loss_weight': args.timeple_boundary_loss_weight,
        'timeple_codec_recon_loss_weight': args.timeple_codec_recon_loss_weight,
        'timeple_embedding_loss_weight': args.timeple_embedding_loss_weight,
        'timeple_embedding_cosine_loss_weight': args.timeple_embedding_cosine_loss_weight,
        'timeple_reencoding_loss_weight': args.timeple_reencoding_loss_weight,
    }


class SwiftSft(SwiftPipeline, TunerMixin):
    args_class = TrainArguments
    args: args_class

    def __init__(self, args: Optional[Union[List[str], TrainArguments]] = None) -> None:
        super().__init__(args)
        self.train_msg = {}
        self._prepare_model_tokenizer()
        self._prepare_template()
        self._prepare_callbacks()
        if self.args.use_flash_ckpt:
            try:
                import dlrover.trainer.torch.flash_checkpoint.hf_trainer
            except ImportError:
                raise ValueError('Please install dlrover to use flash ckpt `pip install dlrover[k8s,torch]')

    def _prepare_generation_config(self):
        args = self.args
        self.model.origin_generation_config = self.model.generation_config
        self.model.generation_config = prepare_generation_config(self.model.generation_config,
                                                                 args.get_request_config(), self.tokenizer)
        logger.info(f'model.generation_config: {self.model.generation_config}')

    def _prepare_model_tokenizer(self, **kwargs):
        args = self.args
        self.model, self.processor = args.get_model_processor(**kwargs)
        if args.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            sequence_parallel.prepare(
                args.sequence_parallel_size, model=self.model, tokenizer=self.processor, padding_free=args.padding_free)
        if self.model is None:
            return
        if hasattr(self.model, 'hf_device_map'):
            logger.info(f'model.hf_device_map: {self.model.hf_device_map}')

        logger.info(f'model_info: {self.model.model_info}')

        self._prepare_generation_config()

    def _prepare_template(self) -> None:
        args = self.args
        for name in ('video_timestamp_text_interleave', 'cis_timestamp_interleave'):
            if self.processor is not None and hasattr(args, name):
                setattr(self.processor, name, getattr(args, name))
        template = args.get_template(self.processor)
        template.set_mode('train')
        if template.use_model:
            template.model = self.model
        if args.model_meta.is_multimodal and (args.padding_free or args.packing) and not template.support_padding_free:
            raise ValueError(f'Template `{args.template}` does not support padding free or packing.')
        self.template = template

    def _get_dataset(self):
        # The random shuffling of the training set occurs in the dataloader of the trainer.
        args = self.args
        dataset_kwargs = args.get_dataset_kwargs()
        train_dataset, val_dataset = load_dataset(
            args.dataset, split_dataset_ratio=args.split_dataset_ratio, shuffle=args.dataset_shuffle, **dataset_kwargs)
        if len(args.val_dataset) > 0:
            # Loading val dataset
            _, val_dataset = load_dataset(
                args.val_dataset, split_dataset_ratio=1.0, shuffle=args.val_dataset_shuffle, **dataset_kwargs)
            assert args.split_dataset_ratio == 0.
        logger.info(f'train_dataset: {train_dataset}')
        logger.info(f'val_dataset: {val_dataset}')

        return train_dataset, val_dataset

    def _get_data_collator(self):
        args = self.args
        template = self.template
        padding_to = args.max_length if args.train_type == 'longlora' else None
        return partial(template.data_collator, padding_to=padding_to)

    def _save_val_dataset(self, val_dataset):
        args = self.args
        output_dir = getattr(args, 'output_dir', None) or getattr(args, 'save')
        if is_master() and isinstance(val_dataset, HfDataset) and not args.val_dataset:
            os.makedirs(output_dir, exist_ok=True)
            val_dataset_path = os.path.join(output_dir, 'val_dataset.jsonl')
            append_to_jsonl(val_dataset_path, val_dataset.to_list())
            logger.info(f'The split dataset from the training set will be saved at: {val_dataset_path}.')

    def _get_cached_dataset(self):
        args = self.args
        assert not args.streaming and not args.lazy_tokenize
        train_datasets, val_datasets = [], []
        for cached_dataset in args.cached_dataset:
            train_path = os.path.join(cached_dataset, 'train')
            val_path = os.path.join(cached_dataset, 'val')
            train_datasets.append(load_from_disk(train_path))
            if os.path.exists(val_path):
                val_datasets.append(load_from_disk(val_path))
        return train_datasets, val_datasets

    def _prepare_dataset(self):
        args = self.args
        is_grpo = hasattr(args, 'rlhf_type') and args.rlhf_type == 'grpo'
        if args.cached_dataset:
            train_datasets, val_datasets = self._get_cached_dataset()
        else:
            train_datasets, val_datasets = [], []
        if args.dataset:
            train_dataset, val_dataset = self._get_dataset()
            train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset, pre_process=not is_grpo)
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
        train_dataset = DatasetLoader._concat_datasets(train_datasets)
        val_dataset = DatasetLoader._concat_datasets(val_datasets)
        datasets = [train_dataset, val_dataset]
        if is_grpo:
            return datasets
        datasets = self._post_process_datasets(datasets)

        return datasets

    def _post_process_datasets(self, datasets: List) -> List:
        args = self.args
        predict_with_generate = getattr(args, 'predict_with_generate', False)

        template = self.template
        for i, dataset in enumerate(datasets):
            if dataset is None:
                continue
            if i == 1 and predict_with_generate:
                # val_dataset
                continue
            if (args.model_meta.is_multimodal or args.lazy_tokenize) and not args.streaming:
                dataset = LazyLLMDataset(dataset, template.encode, strict=args.strict, random_state=args.data_seed)
            if args.packing:
                packing_dataset_cls = IterablePackingDataset if args.streaming else PackingDataset
                dataset = packing_dataset_cls(
                    template,
                    dataset,
                    num_proc=args.dataset_num_proc,
                    strict=args.strict,
                    load_from_cache_file=args.load_from_cache_file)
            elif args.streaming:
                preprocessor = EncodePreprocessor(template=template)
                dataset = preprocessor(
                    dataset,
                    num_proc=args.dataset_num_proc,
                    load_from_cache_file=args.load_from_cache_file,
                    strict=args.strict)
            datasets[i] = dataset
        self._show_dataset(*datasets)
        return datasets

    def run(self):
        args = self.args
        train_dataset, val_dataset = self._prepare_dataset()

        if args.task_type == 'seq_cls':
            args.problem_type = args.problem_type or getattr(self.model.config, 'problem_type', None)
            logger.info(f'args.problem_type: {args.problem_type}')
        args.save_args()

        data_collator = self._get_data_collator()
        # Some tuners require train_dataset and data_collator for preparation: LoRA-GA
        self.model = self.prepare_model(self.args, self.model, template=self.template, train_dataset=train_dataset)
        if args.use_timeple_codec:
            _apply_timeple_codec_freeze(self.model, args)
        logger.info(f'model: {self.model}')
        model_parameter_info = get_model_parameter_info(self.model)
        self.train_msg['model_parameter_info'] = model_parameter_info
        logger.info(f'model_parameter_info: {model_parameter_info}')

        trainer_cls = TrainerFactory.get_trainer_cls(args)
        trainer_kwargs = self._get_trainer_kwargs()
        if args.use_timeple_codec:
            trainer_cls = TimePLECodecTrainer
            trainer_kwargs.update(_get_timeple_codec_trainer_kwargs(args))
        trainer = trainer_cls(
            model=self.model,
            args=self.args.training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=self.callbacks,
            template=self.template,
            **trainer_kwargs,
        )
        return self.train(trainer)

    def _get_trainer_kwargs(self):
        return {}

    def _save_trainer_state(self, trainer):
        training_args = trainer.args
        state = trainer.state
        if hasattr(state, 'last_model_checkpoint'):
            if self.args.create_checkpoint_symlink:
                last_checkpoint = os.path.join(self.args.output_dir, 'last')
                best_checkpoint = os.path.join(self.args.output_dir, 'best')
                if is_master():
                    os.symlink(state.last_model_checkpoint, last_checkpoint)
                    os.symlink(state.best_model_checkpoint, best_checkpoint)
                state.last_model_checkpoint = last_checkpoint
                state.best_model_checkpoint = best_checkpoint
        else:
            state.last_model_checkpoint = None
        logger.info(f'last_model_checkpoint: {state.last_model_checkpoint}')
        logger.info(f'best_model_checkpoint: {state.best_model_checkpoint}')

        # Visualization
        if is_master():
            if 'tensorboard' in training_args.report_to:
                images_dir = os.path.join(training_args.output_dir, 'images')
                logger.info(f'images_dir: {images_dir}')
                plot_images(images_dir, training_args.logging_dir, ['train/loss'], 0.9)
            if training_args.push_to_hub:
                trainer.push_to_hub()

        self.train_msg.update({
            'last_model_checkpoint': state.last_model_checkpoint,
            'best_model_checkpoint': state.best_model_checkpoint,
            'best_metric': state.best_metric,
            'train_metric_for_best_model': getattr(training_args, 'train_metric_for_best_model', None),
            'best_train_model_checkpoint': getattr(trainer, 'best_train_model_checkpoint', None),
            'best_train_snapshot_checkpoint': getattr(trainer, 'best_train_snapshot_checkpoint', None),
            'best_train_metric': getattr(trainer, 'best_train_metric', None),
            'best_train_metric_name': getattr(trainer, 'best_train_metric_name', None),
            'best_train_logged_step': getattr(trainer, 'best_train_metric_logged_step', None),
            'best_train_checkpoint_step': getattr(trainer, 'best_train_checkpoint_step', None),
            'global_step': state.global_step,
            'log_history': state.log_history,
            'memory': getattr(state, 'max_memory', None),
        })
        if is_master():
            jsonl_path = os.path.join(training_args.output_dir, 'logging.jsonl')
            append_to_jsonl(jsonl_path, self.train_msg, strict=False)
        return self.train_msg

    def train(self, trainer):
        logging_path = os.path.join(trainer.args.output_dir, 'logging.jsonl')
        logger.info(f'The logging file will be saved in: {logging_path}')
        try:
            trainer.train(trainer.args.resume_from_checkpoint)
        finally:
            res = self._save_trainer_state(trainer)
            if self.args.use_flash_ckpt:
                trainer.wait_latest_checkpoint(trainer.FLASH_CKPT_WAIT_TIMEOUT)

        return res

    def _prepare_callbacks(self):
        from .callback import DynamicLayerActivationCallback, TrainerAdapterCallback
        args = self.args
        callbacks = []
        if args.lisa_activated_layers > 0:
            assert args.train_type == 'full', 'LISA only supports full parameter training.'
            lisa_callback = DynamicLayerActivationCallback(
                n_layers=args.lisa_activated_layers,  # Number of layers to activate
                step_interval=args.lisa_step_interval,  # Step interval to update active layers
                model=self.model)
            lisa_callback.switch_active_layers()  # Make trainable parameters printing a correct value
            callbacks.append(lisa_callback)

        if args.is_adapter and args.train_type == 'adalora':
            callbacks.append(TrainerAdapterCallback(args))
        callbacks += extra_callbacks
        self.callbacks = callbacks

        if args.early_stop_interval is not None and args.early_stop_interval > 0:
            from swift.plugin.callback import EarlyStopCallback
            self.callbacks.append(EarlyStopCallback(args.early_stop_interval))
            logger.info('You are using the default early stop callback, this is a implementation of '
                        'stopping training when the best metric showing no improvement within {} steps, '
                        'you can write a new implementation in the plugin/callback.py.')

    @staticmethod
    def _stat_dataset(dataset: Union[HfDataset, PackingDataset, LazyLLMDataset]):
        if isinstance(dataset, LazyLLMDataset):
            dataset = dataset.dataset
        if isinstance(dataset, HfDataset):
            length = dataset['length']
        else:
            length = dataset.packed_length
        _, stat_str = stat_array(length)
        logger.info(f'Dataset Token Length: {stat_str}')
        return stat_str

    def _show_dataset(self, train_dataset, val_dataset):
        args = self.args
        predict_with_generate = getattr(args, 'predict_with_generate', False)
        if is_master():
            inputs = train_dataset[0] if hasattr(train_dataset, '__len__') else next(iter(train_dataset))
            if isinstance(inputs, list):
                inputs = inputs[0]
            self.template.print_inputs(inputs)
        elif hasattr(train_dataset, '__len__'):
            # Avoid the random mismatch issue in LazyLLMDataset.
            inputs = train_dataset[0]
        if val_dataset is not None and hasattr(val_dataset, '__len__') and len(val_dataset) == 0:
            val_dataset = None
        if not args.lazy_tokenize and not args.streaming:
            self.train_msg['train_dataset'] = self._stat_dataset(train_dataset)
            if val_dataset is not None and not predict_with_generate:
                self.train_msg['val_dataset'] = self._stat_dataset(val_dataset)

    def _encode_dataset(self, train_dataset, val_dataset, pre_process=True):
        template = self.template
        args = self.args
        self._save_val_dataset(val_dataset)

        predict_with_generate = getattr(args, 'predict_with_generate', False)
        datasets = [train_dataset, val_dataset]
        if not pre_process:
            return datasets

        origin_template_model = template.model
        template.model = None  # Avoid serializing the model.
        for i, dataset in enumerate(datasets):
            if dataset is None:
                continue
            if i == 1 and predict_with_generate:
                # val_dataset
                continue
            if not args.lazy_tokenize and not args.streaming:
                preprocessor = EncodePreprocessor(template=template, pre_tokenize=args.model_meta.is_multimodal)
                batch_size = 100 if args.model_meta.is_multimodal else 1000
                dataset = preprocessor(
                    dataset,
                    num_proc=args.dataset_num_proc,
                    load_from_cache_file=args.load_from_cache_file,
                    strict=args.strict,
                    batch_size=batch_size)
            datasets[i] = dataset
        template.model = origin_template_model

        return datasets


def sft_main(args: Optional[Union[List[str], TrainArguments]] = None):
    if isinstance(args, TrainArguments):
        parsed_args = args
        remaining_argv = []
    else:
        parsed_args, remaining_argv = parse_args(TrainArguments, args)
    if len(remaining_argv) > 0:
        if getattr(parsed_args, 'ignore_args_error', False):
            logger.warning(f'remaining_argv: {remaining_argv}')
        else:
            raise ValueError(f'remaining_argv: {remaining_argv}')

    return SwiftSft(parsed_args).main()
