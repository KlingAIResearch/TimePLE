# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import inspect
import math
import os
import shutil
from contextlib import contextmanager, nullcontext
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from peft import PeftModel
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import EvalPrediction
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers import Trainer as HfTrainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import is_peft_available

from swift.utils import JsonlWriter, Serializer, gc_collect, get_logger, unwrap_model_for_generation
from .arguments import Seq2SeqTrainingArguments, TrainingArguments
from .mixin import DataLoaderMixin, SwiftMixin
from .utils import per_token_loss_func, per_token_loss_func_sp

logger = get_logger()

_TIMEPLE_DETAIL_EXPLICIT_TAGS = {
    'timeple_mae_total': 'timeple/summary/mae_total',
    'span_iou': 'timeple/summary/span_iou',
    'full_path_decode_loss': 'timeple/path/full_decode_loss',
    'dfl_loss': 'timeple/decode/full/dfl_loss',
    'iou_loss': 'timeple/decode/full/iou_loss',
    'base_path_decode_loss': 'timeple/path/base_decode_loss',
    'base_vs_full_gap': 'timeple/path/base_vs_full_gap',
    'embedding_mse_loss': 'timeple/aux/embedding_mse_loss',
    'embedding_cosine_loss': 'timeple/aux/embedding_cosine_loss',
    'reencoding_loss': 'timeple/aux/reencoding_loss',
    'input_residual_ratio': 'timeple/adapter/input_residual_ratio',
    'alpha_ts': 'timeple/adapter/anchor_alpha',
}

_TIMEPLE_DETAIL_PREFIX_TAGS = (
    ('codec_recon_', 'timeple/recon'),
    ('base_codec_', 'timeple/decode/base'),
    ('codec_', 'timeple/decode/full'),
)

_TIMEPLE_DETAIL_SKIP_KEYS = {
    'timeple_loss',
    'codec_total_loss',
    'codec_mae_total',
    'codec_span_iou',
    'base_codec_total_loss',
}

_TIMEPLE_DETAIL_SKIP_PREFIXES = (
    'base_codec_',
)


def _map_timeple_detail_to_log_tag(key: str) -> Optional[str]:
    if key in _TIMEPLE_DETAIL_SKIP_KEYS:
        return None
    for prefix in _TIMEPLE_DETAIL_SKIP_PREFIXES:
        if key.startswith(prefix):
            return None

    tag = _TIMEPLE_DETAIL_EXPLICIT_TAGS.get(key)
    if tag is not None:
        return tag

    for prefix, group in _TIMEPLE_DETAIL_PREFIX_TAGS:
        if key.startswith(prefix):
            suffix = key[len(prefix):]
            return f'{group}/{suffix}'

    return f'timeple/misc/{key}'


def _hardlink_or_copy(src: str, dst: str) -> str:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def _replace_dir_with_snapshot(src_dir: str, dst_dir: str) -> str:
    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)
    if src_dir == dst_dir:
        return dst_dir

    if os.path.lexists(dst_dir):
        if os.path.islink(dst_dir) or os.path.isfile(dst_dir):
            os.remove(dst_dir)
        else:
            shutil.rmtree(dst_dir)

    shutil.copytree(src_dir, dst_dir, copy_function=_hardlink_or_copy)
    return dst_dir


class Trainer(SwiftMixin, DataLoaderMixin, HfTrainer):
    args: TrainingArguments

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_metric_for_best_model = getattr(self.args, 'train_metric_for_best_model', None)
        self._train_greater_is_better = getattr(self.args, 'train_greater_is_better', None)
        if self._train_metric_for_best_model and self._train_greater_is_better is None:
            self._train_greater_is_better = 'loss' not in self._train_metric_for_best_model

        self._latest_train_metric: Optional[float] = None
        self._latest_train_metric_name: Optional[str] = None
        self._latest_train_metric_step: Optional[int] = None

        self.best_train_metric: Optional[float] = None
        self.best_train_metric_name: Optional[str] = None
        self.best_train_metric_logged_step: Optional[int] = None
        self.best_train_checkpoint_step: Optional[int] = None
        self.best_train_model_checkpoint: Optional[str] = None
        self.best_train_snapshot_checkpoint: Optional[str] = None

    def _candidate_train_metric_keys(self) -> List[str]:
        metric = self._train_metric_for_best_model
        if not metric:
            return []
        keys = [metric]
        if metric.startswith('train_'):
            keys.append(metric[len('train_'):])
        else:
            keys.append(f'train_{metric}')

        deduped: List[str] = []
        seen = set()
        for key in keys:
            if key and key not in seen:
                deduped.append(key)
                seen.add(key)
        return deduped

    def _extract_train_metric_from_logs(self, logs: Dict[str, float]) -> Optional[Tuple[str, float]]:
        for key in self._candidate_train_metric_keys():
            if key not in logs:
                continue
            metric_value = logs[key]
            if isinstance(metric_value, torch.Tensor):
                if metric_value.numel() == 0:
                    continue
                metric_value = metric_value.detach().float().mean().item()
            else:
                metric_value = float(metric_value)
            if not math.isfinite(metric_value):
                continue
            return key, metric_value
        return None

    def _record_train_metric_from_logs(self, logs: Dict[str, float]) -> None:
        if not self.model.training or not self._train_metric_for_best_model:
            return
        metric = self._extract_train_metric_from_logs(logs)
        if metric is None:
            return
        metric_name, metric_value = metric
        self._latest_train_metric_name = metric_name
        self._latest_train_metric = metric_value
        self._latest_train_metric_step = self.state.global_step

    def _resolve_train_metric_for_checkpoint(self, checkpoint_step: int) -> Optional[Tuple[int, str, float]]:
        if self._latest_train_metric_step == checkpoint_step and self._latest_train_metric is not None:
            return checkpoint_step, self._latest_train_metric_name, self._latest_train_metric

        for row in reversed(self.state.log_history):
            row_step = row.get('step')
            if row_step is None or row_step > checkpoint_step:
                continue
            metric = self._extract_train_metric_from_logs(row)
            if metric is None:
                continue
            metric_name, metric_value = metric
            return int(row_step), metric_name, metric_value
        return None

    def snapshot_checkpoint(self, src_dir: Optional[str], dst_dir: Optional[str]) -> Optional[str]:
        if not src_dir or not dst_dir or not os.path.isdir(src_dir):
            return None
        snapshot_dir = os.path.abspath(dst_dir)
        if self.is_world_process_zero():
            snapshot_dir = _replace_dir_with_snapshot(src_dir, snapshot_dir)
        distributed_state = getattr(self.args, 'distributed_state', None)
        if distributed_state is not None:
            distributed_state.wait_for_everyone()
        return snapshot_dir

    def _maybe_update_best_train_checkpoint(self, checkpoint_dir: str) -> None:
        if not self._train_metric_for_best_model:
            return

        resolved = self._resolve_train_metric_for_checkpoint(self.state.global_step)
        if resolved is None:
            return

        logged_step, metric_name, metric_value = resolved
        better = (
            self.best_train_metric is None
            or (metric_value > self.best_train_metric if self._train_greater_is_better else metric_value < self.best_train_metric)
        )
        if not better:
            return

        self.best_train_metric = metric_value
        self.best_train_metric_name = metric_name
        self.best_train_metric_logged_step = logged_step
        self.best_train_checkpoint_step = self.state.global_step
        self.best_train_model_checkpoint = os.path.abspath(checkpoint_dir)

        if getattr(self.args, 'preserve_checkpoint_snapshots', False):
            snapshot_name = getattr(self.args, 'train_best_checkpoint_snapshot_name', 'checkpoint-train-best')
            snapshot_dir = os.path.join(self.args.output_dir, snapshot_name)
            self.best_train_snapshot_checkpoint = self.snapshot_checkpoint(checkpoint_dir, snapshot_dir)

    @contextmanager
    def _patch_loss_function(self):
        model = self.model
        if isinstance(model, PeftModel):
            model = model.model
        model_cls = model.__class__
        if not hasattr(model_cls, 'loss_function'):
            yield
            return

        loss_function = model.loss_function
        _old_loss_function = model_cls.loss_function

        @staticmethod
        @wraps(loss_function)
        def new_loss_function(logits, labels, **kwargs):
            labels = labels.to(logits.device)  # fix device_map
            return loss_function(logits=logits, labels=labels, **kwargs)

        model_cls.loss_function = new_loss_function
        try:
            yield
        finally:
            model_cls.loss_function = _old_loss_function

    def train(self, *args, **kwargs):
        with self._patch_loss_function():
            return super().train(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        if inputs.get('labels') is not None:
            self._compute_acc(outputs, inputs['labels'])
        if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
            loss = loss / self.args.gradient_accumulation_steps
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        self._record_train_metric_from_logs(logs)
        return super().log(logs, *args, **kwargs)

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics=metrics)
        if not self._train_metric_for_best_model:
            return

        run_dir = self._get_output_dir(trial=trial)
        checkpoint_dir = os.path.join(run_dir, f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}')
        if os.path.isdir(checkpoint_dir):
            self._maybe_update_best_train_checkpoint(checkpoint_dir)


def gather_for_unpadded_tensors(input_data, use_gather_object=False):
    from accelerate.utils import gather_object
    input_data = gather_object(input_data)
    output = []
    for _data in input_data:
        if len(_data.shape) == 0:
            _data = _data.unsqueeze(0)
        _data = _data.cpu()
        output.append(_data)
    if len(output[0].shape) == 1 and output[0].shape[0] > 1:
        data = torch.stack(output, dim=0)
    else:
        data = torch.concat(output, dim=0)
    return data


class EmbeddingTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_metrics = self.calculate_metric
        self.preprocess_logits_for_metrics = None
        self.label_names = ['labels']
        self.gather_function = gather_for_unpadded_tensors

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        self.gather_function = gather_for_unpadded_tensors
        return output

    def calculate_metric(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        from swift.plugin.loss import calculate_paired_metrics, calculate_infonce_metrics
        args = self.args
        if args.loss_type == 'infonce':
            return calculate_infonce_metrics(eval_prediction.predictions, eval_prediction.label_ids)
        else:
            return calculate_paired_metrics(eval_prediction.predictions, eval_prediction.label_ids)


class RerankerTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_metrics = self.calculate_metric
        self.label_names = ['labels']

        # Set up preprocess_logits_for_metrics to reduce memory usage for generative reranker
        if self.args.loss_type in {'generative_reranker', 'listwise_generative_reranker'}:
            self.preprocess_logits_for_metrics = self._preprocess_generative_reranker_logits
        else:
            self.preprocess_logits_for_metrics = None
        self.gather_function = gather_for_unpadded_tensors

    def _preprocess_generative_reranker_logits(self, logits, labels):
        """
        Preprocess logits for generative reranker to reduce memory usage.
        Extract only the yes/no token logits at the last valid (non -100) timestep
        for each sample, avoiding padded timesteps created by multi-GPU gather.
        """
        import torch
        import os

        # Get token IDs for positive and negative tokens
        positive_token = os.environ.get('GENERATIVE_RERANKER_POSITIVE_TOKEN', 'yes')
        negative_token = os.environ.get('GENERATIVE_RERANKER_NEGATIVE_TOKEN', 'no')

        tokenizer = getattr(self, 'processing_class', None)
        if tokenizer is None:
            # Fallback: return full logits if tokenizer not available
            return logits

        try:
            positive_token_id = tokenizer.convert_tokens_to_ids(positive_token)
            negative_token_id = tokenizer.convert_tokens_to_ids(negative_token)
        except Exception:
            # Fallback: return full logits if token conversion fails
            return logits

        # Extract only the yes/no token logits from the last non -100 position per sample
        # Shapes: logits [batch, seq_len, vocab]
        if len(logits.shape) == 3:
            batch_size, _, vocab_size = logits.shape

            # Identify padded rows whose entire vocab logits are -100
            row_is_pad = (logits == -100).all(dim=-1)  # [batch, seq_len]
            valid_mask = ~row_is_pad
            lengths = valid_mask.long().sum(dim=1) - 1
            lengths = torch.clamp(lengths, min=0)
            last_indices = lengths.to(device=logits.device)

            # Gather the logits at the last valid index for each sample: [batch, vocab]
            gather_index = last_indices.view(batch_size, 1, 1).expand(batch_size, 1, vocab_size)
            last_step_logits = torch.gather(logits, dim=1, index=gather_index).squeeze(1)

            positive_logits = last_step_logits[:, positive_token_id]
            negative_logits = last_step_logits[:, negative_token_id]
            logits = positive_logits - negative_logits
            return logits
        else:
            # Unexpected shape, return as-is
            return logits

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        self.gather_function = gather_for_unpadded_tensors
        return output

    def calculate_metric(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        from swift.plugin.loss import calculate_reranker_metrics
        return calculate_reranker_metrics(eval_prediction.predictions, eval_prediction.label_ids)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Check if we have a custom loss function
        if self.compute_loss_func is not None:
            # Get labels and compute outputs
            labels = inputs.get('labels')
            if labels is not None:
                labels = inputs.pop('labels')

            outputs = model(**inputs)

            if labels is not None:
                # Call custom loss function
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch, trainer=self)
            else:
                # Fallback to model's loss
                loss = outputs.loss

            if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
                loss = loss / self.args.gradient_accumulation_steps

            if labels is not None:
                self._compute_acc(outputs, labels)

            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)


class Seq2SeqTrainer(SwiftMixin, DataLoaderMixin, HfSeq2SeqTrainer):
    args: Seq2SeqTrainingArguments

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True  # fix transformers>=4.46.2
        if self.args.predict_with_generate:
            from swift.llm import PtEngine
            self.infer_engine = PtEngine.from_model_template(
                self.model, self.template, max_batch_size=self.args.per_device_eval_batch_size)
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'predict.jsonl'))

    @staticmethod
    def _predict_data_collator(batch):
        return {'_data': batch}

    @contextmanager
    def _patch_predict_with_generate(self):
        origin_data_collator = self.data_collator
        self.data_collator = self._predict_data_collator
        packing = self.template.packing
        padding_free = self.template.padding_free
        self.template.packing = False
        self.template.padding_free = False
        try:
            yield
        finally:
            self.template.packing = packing
            self.template.padding_free = padding_free
            self.data_collator = origin_data_collator

    def evaluate(self, *args, **kwargs):
        context = self._patch_predict_with_generate() if self.args.predict_with_generate else nullcontext()
        with context:
            res = super().evaluate(*args, **kwargs)
            gc_collect()
            return res

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            with self.template.forward_context(self.model, inputs):
                return super().prediction_step(
                    model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)
        from swift.llm import RequestConfig, InferRequest
        data_list = inputs['_data']
        labels_list = [InferRequest.remove_response(data['messages']) for data in data_list]
        with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation), self.template.generate_context():
            resp_list = self.infer_engine.infer(
                data_list,
                RequestConfig(max_tokens=self.model.generation_config.max_new_tokens),
                use_tqdm=False,
                template=self.template)

        response_list = []
        jsonl_cache = []
        device = self.args.device
        for data, resp, labels in zip(data_list, resp_list, labels_list):
            response = resp.choices[0].message.content
            jsonl_cache.append({'response': response, 'labels': labels, **data})
            response_list.append(Serializer.to_tensor(resp.choices[0].message.content).to(device=device))
        self.jsonl_writer.append(jsonl_cache, gather_obj=True)
        labels_list = [Serializer.to_tensor(labels).to(device=device) for labels in labels_list]
        response_list = pad_sequence(response_list, batch_first=True, padding_value=0)
        labels_list = pad_sequence(labels_list, batch_first=True, padding_value=0)
        return None, response_list, labels_list

    def _prepare_inputs(self, inputs):
        from swift.llm import HfConfigFactory
        args = self.args
        inputs = super()._prepare_inputs(inputs)
        if self.template.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            sequence_parallel.prepare_inputs(inputs)

        use_logits_to_keep = self.get_use_logits_to_keep(self.template.sequence_parallel_size == 1)
        if use_logits_to_keep:
            self.prepare_logits_to_keep(inputs)
            if args.tuner_backend == 'unsloth' and isinstance(inputs['logits_to_keep'], torch.Tensor):
                inputs['logits_to_keep'] = int(inputs['logits_to_keep'].sum())

        base_model = self.template.get_base_model(self.model)
        if self.model.model_info.is_moe_model and 'output_router_logits' in inspect.signature(
                base_model.forward).parameters:
            HfConfigFactory.set_config_attr(base_model.config, 'router_aux_loss_coef', args.router_aux_loss_coef)
            base_model.router_aux_loss_coef = args.router_aux_loss_coef
            logger.info_once(f'router_aux_loss_coef: {args.router_aux_loss_coef}')
            if args.router_aux_loss_coef > 0:
                inputs['output_router_logits'] = True
        inputs['compute_loss_func'] = self.compute_loss_func
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = None
        compute_loss_func: Callable = inputs.pop('compute_loss_func', None)
        loss_scale = inputs.pop('loss_scale', None)
        text_position_ids = inputs.pop('text_position_ids', None)
        if text_position_ids is None:
            text_position_ids = inputs.get('position_ids')
        channels = inputs.pop('channel', None)

        if (self.label_smoother is not None or compute_loss_func is not None or loss_scale is not None
                or self.args.enable_dft_loss or self.args.enable_channel_loss
                or self.template.sequence_parallel_size > 1) and 'labels' in inputs:
            if self.args.use_liger_kernel:
                logger.warning_once('The cross_entropy loss function defined in Liger Kernel will not '
                                    'take effect, potentially leading to increased GPU memory consumption.')
            labels = inputs.pop('labels')
        outputs = model(**inputs)
        if getattr(outputs, 'aux_loss', None) is not None:
            mode = 'train' if self.model.training else 'eval'
            self.custom_metrics[mode]['aux_loss'].update(outputs.aux_loss)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is None:
            labels = inputs['labels']
            outputs.loss = outputs.loss.to(labels.device)
            # fix https://github.com/huggingface/transformers/issues/34263
            if num_items_in_batch is not None:
                outputs.loss = outputs.loss * ((labels[:, 1:] != -100).sum() / num_items_in_batch)

            if isinstance(outputs, dict) and 'loss' not in outputs:
                raise ValueError(
                    'The model did not return a loss from the inputs, only the following keys: '
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.")
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        else:
            outputs.loss = None
            if (self.args.enable_dft_loss or loss_scale is not None or self.args.enable_channel_loss
                    or self.template.sequence_parallel_size > 1):
                if self.template.sequence_parallel_size > 1:
                    outputs.loss = per_token_loss_func_sp(outputs, labels, enable_dft_loss=self.args.enable_dft_loss)
                else:
                    outputs.loss = per_token_loss_func(outputs, labels, enable_dft_loss=self.args.enable_dft_loss)

                if loss_scale is not None:
                    loss_scale = torch.roll(loss_scale, shifts=-1, dims=-1).view(-1)
                    outputs.loss = outputs.loss * loss_scale

                if self.args.enable_channel_loss and channels is not None:
                    mode = 'train' if self.model.training else 'eval'
                    metrics = self.custom_metrics[mode]
                    masks = torch.roll(labels, shifts=-1, dims=-1).view(-1) != -100
                    if self.template.padding_free:
                        cu_seqlens = self.get_cu_seqlens(text_position_ids, inputs.get('logits_to_keep'))
                    else:
                        cu_seqlens = torch.arange(0, labels.shape[0] + 1) * labels.shape[1]
                    for i in range(cu_seqlens.shape[0] - 1):
                        channel = channels[i]
                        slice_ = slice(cu_seqlens[i], cu_seqlens[i + 1])
                        metrics[f'loss_{channel}'].update(outputs.loss[slice_][masks[slice_]])

            unwrapped_model = self.accelerator.unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if compute_loss_func is not None:
                loss = compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch, trainer=self)
            elif self.label_smoother is None:
                # Handle the outputs.loss generated by loss_scale.
                if num_items_in_batch is None:
                    num_items_in_batch = (labels[:, 1:] != -100).sum()
                loss = outputs.loss.sum() / num_items_in_batch
            else:
                if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)

            if self.model.model_info.is_moe_model and self.args.router_aux_loss_coef is not None:
                aux_loss = outputs.get('aux_loss')
                if aux_loss is not None:
                    if num_items_in_batch is not None:
                        aux_loss = aux_loss * ((labels[:, 1:] != -100).sum() / num_items_in_batch)
                    loss = loss + self.args.router_aux_loss_coef * aux_loss.to(loss.device)

        if getattr(self.args, 'average_tokens_across_devices',
                   False) and self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            loss *= self.accelerator.num_processes

        if (outputs.logits is not None and labels is not None and self.args.tuner_backend != 'unsloth'):
            # Liger does not have logits
            # Unsloth has a bug with output logits
            self._compute_acc(outputs, labels)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, *args, **kwargs):
        with self.template.forward_context(self.model, inputs):
            return super().training_step(model, inputs, *args, **kwargs)


class TimeCodecTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with custom loss mixing for Qwen3-VL time codec."""

    def __init__(
        self,
        *args,
        lm_loss_weight: float = 1.0,
        time_loss_weight: float = 1.0,
        compute_time_loss: bool = True,
        time_decode_loss_weight: float = 1.0,
        time_codec_recon_loss_weight: float = 0.0,
        time_embedding_loss_weight: float = 0.0,
        time_embedding_cosine_loss_weight: float = 0.0,
        time_iou_loss_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._lm_loss_weight = lm_loss_weight
        self._time_loss_weight = time_loss_weight
        self._compute_time_loss = compute_time_loss
        self._time_decode_loss_weight = time_decode_loss_weight
        self._time_codec_recon_loss_weight = time_codec_recon_loss_weight
        self._time_embedding_loss_weight = time_embedding_loss_weight
        self._time_embedding_cosine_loss_weight = time_embedding_cosine_loss_weight
        self._time_iou_loss_weight = time_iou_loss_weight
        self._time_loss_log_acc: Dict[str, torch.Tensor] = {}
        self._time_loss_log_count = 0

    def _update_eval_tpfd_metrics(self, outputs) -> None:
        if self.model.training or outputs is None:
            return

        time_loss_details = getattr(outputs, "time_loss_details", None)
        if not isinstance(time_loss_details, dict):
            return

        metric_value = None
        for metric_key in ("tpfd_mae_total", "mae_total"):
            if metric_key in time_loss_details:
                metric_value = time_loss_details[metric_key]
                break
        if metric_value is None:
            return

        if isinstance(metric_value, torch.Tensor):
            if metric_value.numel() == 0:
                return
            metric_value = metric_value.detach().float().mean()
            if not torch.isfinite(metric_value):
                return
        else:
            metric_value = float(metric_value)
            if not math.isfinite(metric_value):
                return

        self.custom_metrics["eval"]["tpfd_mae_total"].update(metric_value)

    def _cache_time_loss_logs(self, outputs, num_items_in_batch=None) -> None:
        if not self.model.training:
            return
        if outputs is None:
            return

        log_items: Dict[str, torch.Tensor] = {}
        time_loss = getattr(outputs, "time_loss", None)
        total_loss = getattr(outputs, "loss", None)
        if time_loss is not None:
            log_items["time_loss"] = time_loss
            log_items["time_loss_weighted"] = time_loss * float(self._time_loss_weight)

        if total_loss is not None:
            if time_loss is not None:
                # Model returns total_loss = lm_loss + time_loss (time_loss_weight=1.0 in compute_loss)
                lm_loss = total_loss - time_loss
            else:
                lm_loss = total_loss
            log_items["lm_loss"] = lm_loss
            log_items["lm_loss_weighted"] = lm_loss * float(self._lm_loss_weight)
            # Training objective uses weighted lm/time losses; log for comparability across stages.
            if time_loss is not None:
                log_items["train_loss"] = (
                    lm_loss * float(self._lm_loss_weight) + time_loss * float(self._time_loss_weight)
                )
            else:
                log_items["train_loss"] = lm_loss * float(self._lm_loss_weight)

        time_loss_details = getattr(outputs, "time_loss_details", None)
        if isinstance(time_loss_details, dict):
            for key, value in time_loss_details.items():
                log_items[f"time_loss_details/{key}"] = value

        acc_device = self.args.device
        for key, value in log_items.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().float().mean()
            else:
                value = torch.tensor(float(value), device=acc_device)
            if key in self._time_loss_log_acc:
                self._time_loss_log_acc[key] = self._time_loss_log_acc[key] + value.to(self._time_loss_log_acc[key])
            else:
                self._time_loss_log_acc[key] = value.to(device=acc_device)
        if log_items:
            self._time_loss_log_count += 1

    def _flush_time_loss_logs(self) -> Dict[str, float]:
        if self._time_loss_log_count <= 0:
            return {}

        count_tensor = torch.tensor(float(self._time_loss_log_count), device=self.args.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            count_tensor = self._nested_gather(count_tensor).sum()

        logs: Dict[str, float] = {}
        for key, value in self._time_loss_log_acc.items():
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                value = self._nested_gather(value).sum()
            else:
                value = value.to(self.args.device)
            logs[key] = (value / count_tensor.clamp_min(1.0)).item()

        self._time_loss_log_acc = {}
        self._time_loss_log_count = 0
        return logs

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        if self.model.training:
            logs.update(self._flush_time_loss_logs())
        return super().log(logs, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(
            **inputs,
            compute_time_loss=self._compute_time_loss,
            time_loss_weight=1.0,
            time_decode_loss_weight=self._time_decode_loss_weight,
            time_codec_recon_loss_weight=self._time_codec_recon_loss_weight,
            time_embedding_loss_weight=self._time_embedding_loss_weight,
            time_embedding_cosine_loss_weight=self._time_embedding_cosine_loss_weight,
            time_iou_loss_weight=self._time_iou_loss_weight,
        )

        loss = torch.tensor(0.0, device=outputs.logits.device)
        if outputs.loss is not None and self._lm_loss_weight != 0:
            if getattr(outputs, "time_loss", None) is not None:
                lm_loss = outputs.loss - outputs.time_loss
            else:
                lm_loss = outputs.loss
            loss = loss + self._lm_loss_weight * lm_loss
        if getattr(outputs, "time_loss", None) is not None and self._time_loss_weight != 0:
            loss = loss + self._time_loss_weight * outputs.time_loss

        if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
            loss = loss / self.args.gradient_accumulation_steps
        self._update_eval_tpfd_metrics(outputs)
        self._cache_time_loss_logs(outputs, num_items_in_batch=num_items_in_batch)
        return (loss, outputs) if return_outputs else loss


class TimePLECodecTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with custom loss mixing for Qwen3-VL TimePLE."""

    def __init__(
        self,
        *args,
        lm_loss_weight: float = 1.0,
        timeple_loss_weight: float = 1.0,
        compute_timeple_loss: bool = True,
        timeple_decode_loss_weight: float = 1.0,
        timeple_dfl_loss_weight: Optional[float] = None,
        timeple_iou_loss_weight: Optional[float] = None,
        timeple_boundary_loss_weight: Optional[float] = None,
        timeple_codec_recon_loss_weight: float = 0.0,
        timeple_embedding_loss_weight: float = 0.0,
        timeple_embedding_cosine_loss_weight: float = 0.0,
        timeple_reencoding_loss_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # The CIS forward signature also contains timestamp_labels/timespan_labels.
        # HF's automatic label-name discovery would require all of them at eval
        # time, so Qwen2.5 span-only eval would skip compute_loss entirely.
        self.label_names = ['labels']
        self._lm_loss_weight = lm_loss_weight
        self._timeple_loss_weight = timeple_loss_weight
        self._compute_timeple_loss = compute_timeple_loss
        self._timeple_decode_loss_weight = timeple_decode_loss_weight
        self._timeple_dfl_loss_weight = timeple_dfl_loss_weight
        self._timeple_iou_loss_weight = timeple_iou_loss_weight
        self._timeple_boundary_loss_weight = timeple_boundary_loss_weight
        self._timeple_codec_recon_loss_weight = timeple_codec_recon_loss_weight
        self._timeple_embedding_loss_weight = timeple_embedding_loss_weight
        self._timeple_embedding_cosine_loss_weight = timeple_embedding_cosine_loss_weight
        self._timeple_reencoding_loss_weight = timeple_reencoding_loss_weight
        self._timeple_loss_log_acc: Dict[str, torch.Tensor] = {}
        self._timeple_loss_log_count = 0

    def evaluation_loop(self, *args, **kwargs):
        metric_key_prefix = kwargs.get('metric_key_prefix', 'eval')
        output = super().evaluation_loop(*args, **kwargs)
        output.metrics.update(self.compute_custom_metrics(self.custom_metrics['eval'], f'{metric_key_prefix}_'))
        return output

    def _update_eval_timeple_metrics(self, outputs) -> None:
        if self.model.training or outputs is None:
            return

        timeple_loss_details = getattr(outputs, "timeple_loss_details", None)
        if not isinstance(timeple_loss_details, dict):
            return

        metric_specs = (
            ("timeple_mae_total", ("timeple_mae_total", "mae_total")),
            ("timeple_span_iou", ("span_iou", "codec_span_iou")),
        )
        for metric_name, metric_keys in metric_specs:
            metric_value = None
            for metric_key in metric_keys:
                if metric_key in timeple_loss_details:
                    metric_value = timeple_loss_details[metric_key]
                    break
            if metric_value is None:
                continue

            if isinstance(metric_value, torch.Tensor):
                if metric_value.numel() == 0:
                    continue
                metric_value = metric_value.detach().float().mean()
                if not torch.isfinite(metric_value):
                    continue
            else:
                metric_value = float(metric_value)
                if not math.isfinite(metric_value):
                    continue

            self.custom_metrics["eval"][metric_name].update(metric_value)

    def _cache_timeple_loss_logs(self, outputs, num_items_in_batch=None) -> None:
        if not self.model.training:
            return
        if outputs is None:
            return

        log_items: Dict[str, torch.Tensor] = {}
        timeple_loss = getattr(outputs, "timeple_loss", None)
        total_loss = getattr(outputs, "loss", None)
        if timeple_loss is not None:
            log_items["timeple_loss"] = timeple_loss
            log_items["timeple_loss_weighted"] = timeple_loss * float(self._timeple_loss_weight)

        if total_loss is not None:
            if timeple_loss is not None:
                # Model returns total_loss = lm_loss + timeple_loss (timeple_loss_weight=1.0 in compute_loss)
                lm_loss = total_loss - timeple_loss
            else:
                lm_loss = total_loss
            log_items["lm_loss"] = lm_loss
            log_items["lm_loss_weighted"] = lm_loss * float(self._lm_loss_weight)
            if timeple_loss is not None:
                log_items["train_loss"] = (
                    lm_loss * float(self._lm_loss_weight) + timeple_loss * float(self._timeple_loss_weight)
                )
            else:
                log_items["train_loss"] = lm_loss * float(self._lm_loss_weight)

        timeple_loss_details = getattr(outputs, "timeple_loss_details", None)
        if isinstance(timeple_loss_details, dict):
            for key, value in timeple_loss_details.items():
                log_tag = _map_timeple_detail_to_log_tag(key)
                if log_tag is None:
                    continue
                log_items[log_tag] = value

        acc_device = self.args.device
        for key, value in log_items.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().float().mean()
            else:
                value = torch.tensor(float(value), device=acc_device)
            if key in self._timeple_loss_log_acc:
                self._timeple_loss_log_acc[key] = self._timeple_loss_log_acc[key] + value.to(self._timeple_loss_log_acc[key])
            else:
                self._timeple_loss_log_acc[key] = value.to(device=acc_device)
        if log_items:
            self._timeple_loss_log_count += 1

    def _flush_timeple_loss_logs(self) -> Dict[str, float]:
        if self._timeple_loss_log_count <= 0:
            return {}

        count_tensor = torch.tensor(float(self._timeple_loss_log_count), device=self.args.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            count_tensor = self._nested_gather(count_tensor).sum()

        logs: Dict[str, float] = {}
        for key, value in self._timeple_loss_log_acc.items():
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                value = self._nested_gather(value).sum()
            else:
                value = value.to(self.args.device)
            logs[key] = (value / count_tensor.clamp_min(1.0)).item()

        self._timeple_loss_log_acc = {}
        self._timeple_loss_log_count = 0
        return logs

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        if self.model.training:
            logs.update(self._flush_timeple_loss_logs())
        return super().log(logs, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(
            **inputs,
            compute_timeple_loss=self._compute_timeple_loss,
            timeple_loss_weight=1.0,
            timeple_decode_loss_weight=self._timeple_decode_loss_weight,
            timeple_dfl_loss_weight=self._timeple_dfl_loss_weight,
            timeple_iou_loss_weight=self._timeple_iou_loss_weight,
            timeple_boundary_loss_weight=self._timeple_boundary_loss_weight,
            timeple_codec_recon_loss_weight=self._timeple_codec_recon_loss_weight,
            timeple_embedding_loss_weight=self._timeple_embedding_loss_weight,
            timeple_embedding_cosine_loss_weight=self._timeple_embedding_cosine_loss_weight,
            timeple_reencoding_loss_weight=self._timeple_reencoding_loss_weight,
        )

        loss = torch.tensor(0.0, device=outputs.logits.device)
        if outputs.loss is not None and self._lm_loss_weight != 0:
            if getattr(outputs, "timeple_loss", None) is not None:
                lm_loss = outputs.loss - outputs.timeple_loss
            else:
                lm_loss = outputs.loss
            loss = loss + self._lm_loss_weight * lm_loss
        if getattr(outputs, "timeple_loss", None) is not None and self._timeple_loss_weight != 0:
            loss = loss + self._timeple_loss_weight * outputs.timeple_loss

        if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
            loss = loss / self.args.gradient_accumulation_steps
        self._update_eval_timeple_metrics(outputs)
        self._cache_timeple_loss_logs(outputs, num_items_in_batch=num_items_in_batch)
        return (loss, outputs) if return_outputs else loss
