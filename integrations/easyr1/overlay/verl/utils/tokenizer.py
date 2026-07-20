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
"""Utils for tokenization."""

import json
import os
from typing import Optional

from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizer, ProcessorMixin


def get_tokenizer(model_path: str, override_chat_template: Optional[str] = None, **kwargs) -> PreTrainedTokenizer:
    """Create a huggingface pretrained tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    if override_chat_template is not None:
        with open(override_chat_template) as f:
            tokenizer.chat_template = f.read()

        print(f"New chat template: {tokenizer.chat_template}")

    if tokenizer.bos_token == "<bos>" and tokenizer.eos_token == "<eos>":
        # the EOS token in gemma2 & gemma3 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        print("Found gemma model. Set eos_token and eos_token_id to <end_of_turn> and 107.")
        tokenizer.eos_token = "<end_of_turn>"

    if tokenizer.pad_token_id is None:
        print("Pad token is None. Set it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_processor(model_path: str, override_chat_template: Optional[str] = None, **kwargs) -> Optional[ProcessorMixin]:
    """Create a huggingface pretrained processor."""
    try:
        processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    except ValueError:
        processor_class = None
        processor_cfg_path = os.path.join(model_path, "processor_config.json")
        if os.path.exists(processor_cfg_path):
            with open(processor_cfg_path, encoding="utf-8") as f:
                processor_class = json.load(f).get("processor_class")
        if processor_class == "Qwen3VLProcessorWithTimeCodec":
            from transformers.models.qwen3_vl.processing_qwen3_vl_time_codec import Qwen3VLProcessorWithTimeCodec

            processor = Qwen3VLProcessorWithTimeCodec.from_pretrained(model_path, **kwargs)
        elif processor_class == "Qwen3VLProcessorWithCISCodec":
            from transformers.models.qwen3_vl.processing_qwen3_vl_cis_codec import Qwen3VLProcessorWithCISCodec

            processor = Qwen3VLProcessorWithCISCodec.from_pretrained(model_path, **kwargs)
        elif processor_class == "Qwen3VLProcessorWithTimePLECodec":
            from transformers.models.qwen3_vl.processing_qwen3_vl_timeple import (
                Qwen3VLProcessorWithTimePLECodec,
            )

            processor = Qwen3VLProcessorWithTimePLECodec.from_pretrained(model_path, **kwargs)
        elif processor_class == "Qwen3VLProcessorWithTimePLECodec":
            from transformers.models.qwen3_vl.processing_qwen3_vl_timeple import (
                Qwen3VLProcessorWithTimePLECodec,
            )

            processor = Qwen3VLProcessorWithTimePLECodec.from_pretrained(model_path, **kwargs)
        elif processor_class == "Qwen3VLProcessorWithTimeED":
            from transformers.models.qwen3_vl.processing_qwen3_vl_timeed import Qwen3VLProcessorWithTimeED

            processor = Qwen3VLProcessorWithTimeED.from_pretrained(model_path, **kwargs)
        else:
            raise

    if override_chat_template is not None:
        with open(override_chat_template) as f:
            processor.chat_template = f.read()

        print(f"New chat template: {processor.chat_template}")

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/auto/processing_auto.py#L386
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    return processor
