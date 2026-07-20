"""
Gemini KS wrapper for Qwen3-VL inference via Google Vertex AI API

This module provides a wrapper around Google's Gemini API (Vertex AI)
to enable image sequence + text inference for temporal grounding tasks.

依赖说明：
- google-genai 包：需要通过 uv 安装 (uv add google-genai)
- Google Cloud 凭证：通过配置文件指定 credentials_path

使用环境变量（可选）：
- export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
"""

import logging
import time
import os
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING
from PIL import Image
import io

logger = logging.getLogger(__name__)

# For type hints
if TYPE_CHECKING:
    from google.genai import types

# ==============================================================================
# 导入 Google Gemini SDK
# ==============================================================================
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
    logger.debug("Google Gemini SDK imported successfully")
except ImportError as e:
    GEMINI_AVAILABLE = False
    # Create dummy types to avoid NameError
    genai = None
    types = None
    logger.warning(f"Failed to import google.genai: {e}. Install with: uv add google-genai")

# ==============================================================================
# 综合检查依赖状态
# ==============================================================================
if not GEMINI_AVAILABLE:
    logger.warning(
        "Gemini KS mode will not be available. "
        "Missing dependency: google-genai (run: uv add google-genai)"
    )


class GeminiKSWrapper:
    """
    Wrapper for Google Gemini API (Vertex AI) to support Qwen3-VL inference

    Features:
    - REST API-based communication via Google Gemini SDK
    - Support for image sequence + text input
    - Automatic retry with exponential backoff
    - Configurable safety settings and generation parameters
    - Single-turn inference (non-conversational mode)

    Example:
        >>> api_config = {
        ...     'project_id': 'my-gcp-project',
        ...     'location': 'global',
        ...     'model_name': 'gemini-2.0-flash',
        ...     'credentials_path': '/path/to/credentials.json',
        ...     'max_retries': 3
        ... }
        >>> wrapper = GeminiKSWrapper(api_config=api_config)
        >>> response = wrapper.generate(
        ...     text="Describe this image sequence",
        ...     images=[img1, img2, img3]
        ... )
    """

    def __init__(
        self,
        api_config: Dict[str, Any],
        timeout: int = 1000,
        retry: int = 3,
        wait: float = 5.0
    ):
        """
        初始化 Gemini KS wrapper

        参数:
            api_config: API 配置字典，包含:
                - project_id: Google Cloud project ID (必需)
                - location: Vertex AI location (默认: "global")
                - model_name: Gemini model name (默认: "gemini-2.0-flash")
                - credentials_path: Service account JSON 文件路径 (可选)
                - temperature: 生成温度 (默认: 0.0)
                - max_output_tokens: 最大输出 token 数 (默认: 8192)
                - top_p: Top-p sampling (默认: 1.0)
                - top_k: Top-k sampling (默认: 1)
                - media_resolution: 媒体分辨率 (默认: "media_resolution_medium")
                    可选值: "media_resolution_low", "media_resolution_medium", "media_resolution_high"
                    视频token使用: low=70/帧, medium=70/帧, high=280/帧
                - video_fps: 视频帧率 (默认: 1.0)
                    范围: 0.1 - 60.0 FPS
                    控制视频采样时的帧率，影响实际处理的帧数
            timeout: 请求超时时间（毫秒）
            retry: 最大重试次数
            wait: 重试等待时间（秒）
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google Gemini SDK is not available. "
                "Please install it with: uv add google-genai"
            )

        # Extract config
        self.project_id = api_config.get('project_id')
        if not self.project_id:
            raise ValueError("project_id is required in api_config")

        self.location = api_config.get('location', 'global')
        self.model_name = api_config.get('model_name', 'gemini-2.0-flash')
        self.credentials_path = api_config.get('credentials_path')

        # Generation parameters
        self.temperature = api_config.get('temperature', 0.0)
        self.max_output_tokens = api_config.get('max_output_tokens', 8192)
        self.top_p = api_config.get('top_p', 1.0)
        self.top_k = api_config.get('top_k', 1)

        # Media resolution parameter for Gemini 3
        # Options: 'media_resolution_low' (70 tokens/frame for video),
        #          'media_resolution_medium' (70 tokens/frame for video),
        #          'media_resolution_high' (280 tokens/frame for video)
        # Lower resolution reduces token usage and cost
        self.media_resolution = api_config.get('media_resolution', 'media_resolution_high')

        # Video FPS parameter (frames per second)
        # Controls the frame sampling rate when processing videos
        # Range: 0.1 - 60.0 FPS
        # - Low FPS (< 1): For long videos or static content (e.g., lectures)
        # - Default (1.0): Standard sampling rate
        # - High FPS (> 5): For fast-action videos or high temporal precision
        self.video_fps = api_config.get('video_fps', 1.0)

        # Retry configuration
        self.timeout = timeout
        self.max_retries = retry
        self.retry_delay = wait

        # Set up credentials if provided
        if self.credentials_path:
            if not os.path.exists(self.credentials_path):
                raise FileNotFoundError(
                    f"Credentials file not found: {self.credentials_path}"
                )
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_path
            logger.info(f"Set GOOGLE_APPLICATION_CREDENTIALS to: {self.credentials_path}")

        # Initialize Gemini client
        try:
            client_http_options = None
            if types is not None and hasattr(types, "HttpOptions"):
                client_http_options = types.HttpOptions(timeout=self.timeout)

            self.client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location,
                http_options=client_http_options,
            )
            logger.info(
                f"Gemini client initialized successfully: "
                f"project={self.project_id}, location={self.location}, "
                f"model={self.model_name}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise

        # Safety settings (turn off all safety filters)
        self.safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ]

        # Generation config
        self._update_generation_config()

        logger.info(
            f"Gemini KS wrapper initialized: "
            f"max_retries={self.max_retries}, retry_delay={self.retry_delay}s, "
            f"video_fps={self.video_fps}, media_resolution={self.media_resolution}"
        )

    def _update_generation_config(
        self,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None
    ):
        """
        更新生成配置

        参数:
            temperature: 温度参数（可选，使用实例默认值）
            max_output_tokens: 最大输出 token 数（可选）
            top_p: Top-p sampling（可选）
            top_k: Top-k sampling（可选）
        """
        generation_kwargs = {
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": top_p if top_p is not None else self.top_p,
            "top_k": top_k if top_k is not None else self.top_k,
            "max_output_tokens": (
                max_output_tokens if max_output_tokens is not None
                else self.max_output_tokens
            ),
            "safety_settings": self.safety_settings,
        }
        if types is not None and hasattr(types, "HttpOptions"):
            generation_kwargs["http_options"] = types.HttpOptions(timeout=self.timeout)
        self.generation_config = types.GenerateContentConfig(**generation_kwargs)

    def _process_images(self, images: Optional[List[Image.Image]]) -> List:
        """
        将 PIL 图像列表转换为 Gemini API 接受的 Part 对象列表

        参数:
            images: PIL 图像列表

        返回:
            Gemini Part 对象列表
        """
        if not images:
            return []

        parts = []
        for idx, img in enumerate(images):
            try:
                # Convert PIL Image to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()

                # Create Gemini Part with media_resolution
                # Note: Use Part() constructor for consistency with _process_video
                part = types.Part(
                    inline_data=types.Blob(
                        data=img_bytes,
                        mime_type="image/png"
                    ),
                    # Set media resolution for token control (Gemini 3 feature)
                    media_resolution=types.PartMediaResolution(
                        level=self.media_resolution
                    )
                )
                parts.append(part)

            except Exception as e:
                logger.error(f"Failed to process image {idx}: {e}")
                raise

        logger.debug(
            f"Processed {len(parts)} images for Gemini API "
            f"(media_resolution={self.media_resolution})"
        )
        return parts

    def _process_video(self, video_path: str) -> types.Part:
        """
        将视频文件转换为 Gemini API 接受的 Part 对象

        参数:
            video_path: 视频文件路径

        返回:
            Gemini Part 对象
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Read video file as bytes
        with open(video_path, 'rb') as f:
            video_bytes = f.read()

        # Determine MIME type based on file extension
        ext = os.path.splitext(video_path)[1].lower()
        mime_type_map = {
            '.mp4': 'video/mp4',
            '.mpeg': 'video/mpeg',
            '.mov': 'video/mov',
            '.avi': 'video/avi',
            '.flv': 'video/x-flv',
            '.mpg': 'video/mpg',
            '.webm': 'video/webm',
            '.wmv': 'video/wmv',
            '.3gpp': 'video/3gpp'
        }
        mime_type = mime_type_map.get(ext, 'video/mp4')  # Default to mp4

        # Create Gemini Part with video_metadata (FPS) and media_resolution
        # Note: Use Part() constructor instead of Part.from_bytes() to support video_metadata
        part = types.Part(
            inline_data=types.Blob(
                data=video_bytes,
                mime_type=mime_type
            ),
            # Set video metadata with FPS for frame sampling control
            video_metadata=types.VideoMetadata(fps=self.video_fps),
            # Set media resolution for token control (Gemini 3 feature)
            media_resolution=types.PartMediaResolution(
                level=self.media_resolution
            )
        )

        logger.debug(
            f"Processed video file: {video_path} "
            f"(size: {len(video_bytes)} bytes, mime: {mime_type}, "
            f"fps={self.video_fps}, media_resolution={self.media_resolution})"
        )
        return part

    def _retry_with_backoff(
        self,
        func,
        *args,
        **kwargs
    ) -> Any:
        """
        使用指数退避策略执行函数并重试

        参数:
            func: 要执行的函数
            *args: 函数位置参数
            **kwargs: 函数关键字参数

        返回:
            函数执行结果

        抛出:
            最后一次执行的异常
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries} failed: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"All {self.max_retries} attempts failed. Last error: {e}"
                    )

        raise last_exception

    def generate(
        self,
        text: str,
        images: Optional[List[Image.Image]] = None,
        videos: Optional[List] = None,
        video_metadata: Optional[List] = None,
        video_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        """
        执行单次推理

        参数:
            text: 输入文本查询
            images: PIL 图像列表（用于图像序列，与 video_path 互斥）
            videos: 视频输入（本实现中不使用，使用 images 或 video_path 代替）
            video_metadata: 视频元数据（本实现中不使用）
            video_path: 视频文件路径（直接传视频文件，与 images 互斥）
            system_prompt: 系统提示词（可选）
            max_new_tokens: 最大生成 token 数（覆盖默认值）
            temperature: 温度参数（覆盖默认值）
            top_p: Top-p sampling（覆盖默认值）

        返回:
            模型生成的文本响应
        """
        # Update generation config if parameters are overridden
        if (max_new_tokens is not None or temperature is not None or
            top_p is not None):
            self._update_generation_config(
                temperature=temperature,
                max_output_tokens=max_new_tokens,
                top_p=top_p
            )

        # Handle system prompt
        # Gemini API 支持 system_instruction 参数，但为了简单起见，我们将其拼接到 text 前面
        # 这样可以确保兼容性，且不需要修改 API 调用结构
        if system_prompt:
            formatted_text = f"{system_prompt}\n\n{text}"
            logger.info("Gemini KS mode: System prompt prepended to text")
        else:
            formatted_text = text

        # Build content parts
        content_parts = []

        # Priority: video_path > images
        # (video_path and images are mutually exclusive)
        if video_path:
            # Use direct video file input
            video_part = self._process_video(video_path)
            content_parts.append(video_part)
            logger.info(f"Added video file to request: {video_path}")
        elif images:
            # Use image sequence (frame extraction mode)
            image_parts = self._process_images(images)
            content_parts.extend(image_parts)
            logger.info(f"Added {len(image_parts)} images to request")

        # Add text prompt (with system prompt prepended if provided)
        text_part = types.Part.from_text(text=formatted_text)
        content_parts.append(text_part)

        # Build content
        request_content = types.Content(
            role="user",
            parts=content_parts
        )

        # Define the API call function
        def _call_api():
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=request_content,
                config=self.generation_config,
            )

            # Extract text from response
            # Try multiple ways to get text from response
            response_text = None

            # Method 1: Direct .text attribute
            if hasattr(response, 'text'):
                try:
                    response_text = response.text
                    if response_text:
                        return response_text.strip()
                except (ValueError, AttributeError) as e:
                    logger.debug(f"Failed to get response.text: {e}")

            # Method 2: Check candidates (更宽松的提取)
            if hasattr(response, 'candidates') and response.candidates:
                logger.debug(f"Found {len(response.candidates)} candidates")
                for i, candidate in enumerate(response.candidates):
                    logger.debug(f"Processing candidate {i}")
                    if hasattr(candidate, 'content') and candidate.content:
                        logger.debug(f"Candidate {i} has content")
                        if hasattr(candidate.content, 'parts') and candidate.content.parts:
                            logger.debug(f"Candidate {i} has {len(candidate.content.parts)} parts")
                            parts_text = []
                            for j, part in enumerate(candidate.content.parts):
                                # 打印 part 的类型和属性
                                logger.debug(f"Part {j} type: {type(part)}, attributes: {dir(part)[:10]}...")

                                # 尝试多种方式获取 text
                                text_content = None
                                if hasattr(part, 'text'):
                                    text_content = part.text
                                    logger.debug(f"Part {j} has text attribute: {type(text_content)}, value: {repr(text_content)[:100]}")
                                elif hasattr(part, 'get') and callable(part.get):
                                    text_content = part.get('text')
                                    logger.debug(f"Part {j} has get('text'): {type(text_content)}")

                                # 转换为字符串并添加（即使是空字符串也可能有意义）
                                if text_content is not None:
                                    str_content = str(text_content)
                                    parts_text.append(str_content)
                                    logger.debug(f"Part {j} added to parts_text (length: {len(str_content)})")

                            if parts_text:
                                response_text = ''.join(parts_text).strip()
                                if response_text:
                                    logger.info(f"Extracted text from candidate.content.parts (length: {len(response_text)})")
                                    return response_text
                                else:
                                    logger.warning(f"parts_text joined but empty after strip. parts_text: {parts_text}")
                            else:
                                logger.warning(f"Candidate {i} has parts but no text extracted")

            # Method 3: Check finish_reason and handle MAX_TOKENS
            finish_reason = None
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'finish_reason'):
                        finish_reason = candidate.finish_reason
                        logger.warning(f"Finish reason: {finish_reason}")

                        # Handle MAX_TOKENS case - may have partial content
                        if str(finish_reason) == "FinishReason.MAX_TOKENS":
                            logger.warning("Response truncated due to MAX_TOKENS. Attempting to extract partial content...")
                            # Try to get any available text even if incomplete
                            if hasattr(candidate, 'content'):
                                content = candidate.content
                                if hasattr(content, 'parts') and content.parts:
                                    partial_texts = []
                                    for part in content.parts:
                                        if hasattr(part, 'text'):
                                            partial_texts.append(str(part.text) if part.text else "")
                                    if partial_texts:
                                        partial_text = ''.join(partial_texts).strip()
                                        if partial_text:
                                            logger.warning(f"Extracted partial response (length: {len(partial_text)})")
                                            return partial_text

                    if hasattr(candidate, 'safety_ratings'):
                        logger.warning(f"Safety ratings: {candidate.safety_ratings}")

            # Method 4: Log response structure for debugging
            logger.warning(f"Empty response received. Response object: {response}")
            logger.warning(f"Response type: {type(response)}")

            # Provide more specific error message based on finish_reason
            if finish_reason and str(finish_reason) == "FinishReason.MAX_TOKENS":
                raise ValueError(
                    f"Gemini response exceeded max_output_tokens ({self.max_output_tokens}). "
                    "Consider increasing max_output_tokens in config or reducing input length."
                )
            else:
                raise ValueError(f"Received empty response from Gemini API (finish_reason: {finish_reason})")

        # Execute with retry
        logger.info(
            f"Calling Gemini API: model={self.model_name}, "
            f"text_length={len(text)}, num_images={len(images) if images else 0}"
        )

        try:
            response_text = self._retry_with_backoff(_call_api)
            logger.info(f"Gemini API call succeeded, response length: {len(response_text)}")
            return response_text

        except Exception as e:
            logger.error(f"Gemini API call failed after {self.max_retries} retries: {e}")
            raise

    def generate_stream(
        self,
        text: str,
        images: Optional[List[Image.Image]] = None,
        video_path: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        """
        执行流式推理（返回完整拼接后的文本）

        注意：此方法仅用于需要流式响应的场景，但最终返回完整文本。
        对于 temporal grounding 任务，通常使用 generate() 即可。

        参数:
            text: 输入文本查询
            images: PIL 图像列表（与 video_path 互斥）
            video_path: 视频文件路径（与 images 互斥）
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_p: Top-p sampling

        返回:
            模型生成的完整文本响应
        """
        # Update generation config
        if (max_new_tokens is not None or temperature is not None or
            top_p is not None):
            self._update_generation_config(
                temperature=temperature,
                max_output_tokens=max_new_tokens,
                top_p=top_p
            )

        # Build content parts
        content_parts = []

        # Priority: video_path > images
        if video_path:
            video_part = self._process_video(video_path)
            content_parts.append(video_part)
        elif images:
            image_parts = self._process_images(images)
            content_parts.extend(image_parts)

        text_part = types.Part.from_text(text=text)
        content_parts.append(text_part)

        request_content = types.Content(
            role="user",
            parts=content_parts
        )

        # Define streaming API call
        def _call_api_stream():
            chunks = []
            for chunk in self.client.models.generate_content_stream(
                model=self.model_name,
                contents=request_content,
                config=self.generation_config,
            ):
                # Try to extract text from chunk
                chunk_text = None

                # Method 1: Direct .text attribute
                if hasattr(chunk, 'text'):
                    try:
                        chunk_text = chunk.text
                        if chunk_text:
                            chunks.append(chunk_text)
                            continue
                    except (ValueError, AttributeError):
                        pass

                # Method 2: Check candidates.content.parts
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    for candidate in chunk.candidates:
                        if hasattr(candidate, 'content') and candidate.content:
                            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                for part in candidate.content.parts:
                                    if hasattr(part, 'text') and part.text:
                                        chunks.append(part.text)

            response = ''.join(chunks).strip()
            if not response:
                raise ValueError("Received empty response from Gemini streaming API")
            return response

        # Execute with retry
        logger.info(
            f"Calling Gemini streaming API: model={self.model_name}, "
            f"text_length={len(text)}, num_images={len(images) if images else 0}"
        )

        try:
            response_text = self._retry_with_backoff(_call_api_stream)
            logger.info(
                f"Gemini streaming API call succeeded, "
                f"response length: {len(response_text)}"
            )
            return response_text

        except Exception as e:
            logger.error(
                f"Gemini streaming API call failed after {self.max_retries} retries: {e}"
            )
            raise
