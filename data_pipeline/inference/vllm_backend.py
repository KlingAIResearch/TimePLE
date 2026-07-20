"""
vLLM推理引擎 - 支持大规模模型的Tensor Parallelism推理

主要用于Qwen3-VL系列超大模型(如235B)的单机多卡推理
使用vLLM的tensor parallelism和expert parallelism特性实现高效推理
"""

import os
import logging
import socket
from typing import Dict, List, Optional, Tuple, Any
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)

# 条件导入vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available. Please install with: uv pip install -U vllm>=0.11.0")

# 条件导入transformers (仅用于processor)
try:
    from transformers import AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Please install: uv pip install transformers>=4.57.0")


class VLLMInferenceEngine:
    """
    基于vLLM的推理引擎

    特性:
        - Tensor Parallelism: 将模型参数分片到多个GPU
        - Expert Parallelism: MoE模型的专家并行
        - 高效推理: PagedAttention等优化技术
        - 支持video_only和interleaved两种输入格式

    适用场景:
        - 超大规模模型(如Qwen3-VL-235B)
        - 单机多卡环境(8x A100/H100等)
        - 生产级推理服务
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 8,
        dtype: str = "bfloat16",
        mm_encoder_tp_mode: str = "data",
        enable_expert_parallel: bool = True,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        enforce_eager: bool = False,
        seed: int = 0,
        system_prompt: Optional[str] = None,
        include_frame_timestamps: bool = True
    ):
        """
        初始化vLLM推理引擎

        参数:
            model_path: 模型路径
            tensor_parallel_size: Tensor并行度(GPU数量), 默认8
            dtype: 数据类型, 支持 'auto', 'float16', 'bfloat16', 'float32'
            mm_encoder_tp_mode: 多模态编码器的并行模式, 'data'(数据并行)或'tensor'(张量并行)
            enable_expert_parallel: 是否启用MoE专家并行
            gpu_memory_utilization: GPU显存利用率(0.0-1.0), 默认0.9
            max_model_len: 最大序列长度, None表示使用模型默认值
            max_num_seqs: 最大并发序列数, 用于控制warm up时的内存使用, None表示使用默认值
            enforce_eager: 是否强制使用eager模式(调试用), 默认False
            seed: 随机种子
            system_prompt: 系统提示词（可选）
            include_frame_timestamps: 是否在query中包含帧时间戳信息
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is required for VLLMInferenceEngine. "
                "Please install: uv pip install -U vllm>=0.11.0"
            )

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers is required for processor. "
                "Please install: uv pip install transformers>=4.57.0"
            )

        logger.info(f"初始化vLLM推理引擎: {model_path}")
        logger.info(f"Tensor并行度: {tensor_parallel_size}")
        logger.info(f"数据类型: {dtype}")
        logger.info(f"多模态编码器TP模式: {mm_encoder_tp_mode}")
        logger.info(f"启用专家并行: {enable_expert_parallel}")

        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.system_prompt = system_prompt
        self.include_frame_timestamps = include_frame_timestamps
        self.last_frame_timestamps = None  # Store timestamps from last inference

        # 设置vLLM环境变量
        # 关键: 设置多进程启动方法为spawn以避免CUDA初始化问题
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

        self._log_runtime_diagnostics()

        # 加载processor (用于构造输入)
        logger.info("加载processor...")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        # 初始化vLLM引擎
        logger.info("初始化vLLM LLM引擎...")

        # 构建vLLM初始化参数
        vllm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
            "seed": seed,
        }

        # 添加多模态相关参数
        vllm_kwargs["mm_encoder_tp_mode"] = mm_encoder_tp_mode

        # 如果启用专家并行
        if enable_expert_parallel:
            vllm_kwargs["enable_expert_parallel"] = True

        # 如果指定了最大序列长度
        if max_model_len is not None:
            vllm_kwargs["max_model_len"] = max_model_len

        # 如果指定了最大并发序列数
        if max_num_seqs is not None:
            vllm_kwargs["max_num_seqs"] = max_num_seqs

        # 调试模式
        if enforce_eager:
            vllm_kwargs["enforce_eager"] = True
            logger.warning("使用eager模式(性能较低, 仅用于调试)")

        try:
            self.llm = LLM(**vllm_kwargs)
            logger.info("vLLM引擎初始化成功")
        except Exception as e:
            logger.error(f"vLLM引擎初始化失败: {e}")
            raise

    @staticmethod
    def _log_runtime_diagnostics() -> None:
        try:
            import torch

            logger.info(
                "vLLM runtime env: host=%s pid=%s CUDA_VISIBLE_DEVICES=%s VLLM_CACHE_ROOT=%s "
                "VLLM_WORKER_MULTIPROC_METHOD=%s torch.cuda.is_available=%s torch.cuda.device_count=%s",
                socket.gethostname(),
                os.getpid(),
                os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
                os.environ.get("VLLM_CACHE_ROOT", "<unset>"),
                os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "<unset>"),
                torch.cuda.is_available(),
                torch.cuda.device_count(),
            )
            if torch.cuda.is_available():
                visible_devices = []
                for idx in range(torch.cuda.device_count()):
                    try:
                        visible_devices.append(f"{idx}:{torch.cuda.get_device_name(idx)}")
                    except Exception:
                        visible_devices.append(f"{idx}:<unavailable>")
                logger.info("vLLM visible CUDA devices: %s", ", ".join(visible_devices))
        except Exception as exc:
            logger.warning("Failed to collect vLLM runtime diagnostics: %s", exc)

    def _prepare_vllm_inputs(
        self,
        messages: List[Dict],
        return_timestamps: bool = True
    ) -> Dict[str, Any]:
        """
        准备vLLM输入

        参数:
            messages: 聊天消息列表
            return_timestamps: 是否返回帧时间戳

        返回:
            包含prompt和multi_modal_data的字典
        """
        # 应用聊天模板生成文本prompt
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 处理视觉信息
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True
        )

        # 处理视频元数据并提取时间戳
        if return_timestamps and video_inputs is not None:
            # video_inputs 是 [(video_data, metadata), ...] 格式
            # 提取第一个视频的时间戳，但保持原始格式传给vLLM
            if len(video_inputs) > 0:
                _, metadata = video_inputs[0]
                if isinstance(metadata, dict) and 'fps' in metadata and 'frames_indices' in metadata:
                    fps = metadata['fps']
                    frame_indices = metadata['frames_indices']
                    self.last_frame_timestamps = [frame_idx / fps for frame_idx in frame_indices]
                    logger.info(f"提取到 {len(self.last_frame_timestamps)} 个帧时间戳")

            # vLLM需要完整的 (video_data, metadata) 格式，不要解包
            # video_inputs 保持原样

        # 构建vLLM输入格式
        vllm_input = {
            "prompt": text,
            "multi_modal_data": {}
        }

        # 添加图像输入
        if image_inputs is not None:
            vllm_input["multi_modal_data"]["image"] = image_inputs

        # 添加视频输入
        if video_inputs is not None:
            vllm_input["multi_modal_data"]["video"] = video_inputs

        return vllm_input

    def inference_video_only(
        self,
        video_path: str,
        query: str,
        video_params: Dict,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        use_tqdm: bool = True,
    ) -> str:
        """
        video_only模式推理

        输入格式: {"type": "video", "video": video_path, ...} + text

        参数:
            video_path: 视频路径
            query: 查询文本
            video_params: 视频参数(total_pixels, min_pixels, max_frames, sample_fps等)
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus sampling参数
            top_k: top-k sampling参数
            repetition_penalty: 重复惩罚

        返回:
            生成的文本
        """
        # 先提取帧时间戳（如果需要）
        frame_timestamps = []
        if self.include_frame_timestamps:
            # 临时构造messages用于提取timestamp
            temp_messages = [{
                "role": "user",
                "content": [video_params, {"type": "text", "text": query}]
            }]

            # 处理视觉信息以获取元数据
            _, video_inputs, _ = process_vision_info(
                temp_messages,
                image_patch_size=self.processor.image_processor.patch_size,
                return_video_kwargs=True,
                return_video_metadata=True
            )

            # 提取时间戳
            if video_inputs is not None and len(video_inputs) > 0:
                _, metadata = video_inputs[0]
                if isinstance(metadata, dict) and 'fps' in metadata and 'frames_indices' in metadata:
                    fps = metadata['fps']
                    frame_indices = metadata['frames_indices']
                    frame_timestamps = [frame_idx / fps for frame_idx in frame_indices]
                    logger.info(f"提取到 {len(frame_timestamps)} 个帧时间戳")

        # 格式化 query（根据配置决定是否添加timestamp）
        if self.include_frame_timestamps and frame_timestamps:
            timestamp_str = ", ".join([f"{t:.2f}s" for t in frame_timestamps])
            formatted_query = f"[Video: {len(frame_timestamps)} frames at timestamps: {timestamp_str}]\n{query}"
            logger.info(f"在query中添加了 {len(frame_timestamps)} 个帧时间戳")
        else:
            formatted_query = query
            if frame_timestamps:
                logger.info(f"帧时间戳可用 ({len(frame_timestamps)} 帧) 但未添加到query (include_frame_timestamps=False)")

        # 构造消息
        messages = []

        # 添加system prompt（如果配置了）
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        messages.append({
            "role": "user",
            "content": [
                video_params,
                {"type": "text", "text": formatted_query}
            ]
        })

        # 准备vLLM输入
        vllm_input = self._prepare_vllm_inputs(messages, return_timestamps=True)

        # 构建采样参数
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            skip_special_tokens=True
        )

        # 执行推理
        logger.info(f"开始vLLM推理 (video_only模式)")
        outputs = self.llm.generate(
            [vllm_input],
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
        )

        # 提取生成文本
        if outputs and len(outputs) > 0:
            generated_text = outputs[0].outputs[0].text
            logger.info(f"vLLM推理完成, 生成长度: {len(generated_text)} 字符")
            return generated_text
        else:
            logger.error("vLLM推理失败: 无输出")
            return ""

    def inference_interleaved(
        self,
        messages: List[Dict],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        use_tqdm: bool = True,
    ) -> str:
        """
        interleaved模式推理 (时间戳-图像交错)

        输入格式: [<t1 seconds>, <image1>, <t2 seconds>, <image2>, ..., query]

        参数:
            messages: 消息列表(已构造好的交错格式)
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus sampling参数
            top_k: top-k sampling参数
            repetition_penalty: 重复惩罚

        返回:
            生成的文本
        """
        # 准备vLLM输入
        vllm_input = self._prepare_vllm_inputs(messages, return_timestamps=False)

        # 构建采样参数
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            skip_special_tokens=True
        )

        # 执行推理
        logger.info(f"开始vLLM推理 (interleaved模式)")
        outputs = self.llm.generate(
            [vllm_input],
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
        )

        # 提取生成文本
        if outputs and len(outputs) > 0:
            generated_text = outputs[0].outputs[0].text
            logger.info(f"vLLM推理完成, 生成长度: {len(generated_text)} 字符")
            return generated_text
        else:
            logger.error("vLLM推理失败: 无输出")
            return ""

    def batch_inference(
        self,
        batch_messages: List[List[Dict]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        use_tqdm: bool = True,
    ) -> List[str]:
        """
        批量推理(vLLM原生支持)

        参数:
            batch_messages: 批量消息列表
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus sampling参数
            top_k: top-k sampling参数
            repetition_penalty: 重复惩罚

        返回:
            生成文本列表
        """
        # 准备批量输入
        vllm_inputs = [
            self._prepare_vllm_inputs(messages, return_timestamps=False)
            for messages in batch_messages
        ]

        # 构建采样参数
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            skip_special_tokens=True
        )

        # 执行批量推理
        logger.info(f"开始vLLM批量推理: batch_size={len(batch_messages)}")
        outputs = self.llm.generate(
            vllm_inputs,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
        )

        # 提取生成文本
        results = []
        for output in outputs:
            if output.outputs and len(output.outputs) > 0:
                results.append(output.outputs[0].text)
            else:
                results.append("")

        logger.info(f"vLLM批量推理完成")
        return results
