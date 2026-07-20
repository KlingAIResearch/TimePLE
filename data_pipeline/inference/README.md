# Inference backends

This directory contains the optional teacher-model adapters used by `train_building/step2_infer_models.py`.

- `gemini_backend.py` uses Vertex AI through `google-genai`; install it with `uv sync --extra data-gemini`.
- `vllm_backend.py` uses vLLM and `qwen-vl-utils`; install it with `uv sync --extra data-vllm` on a compatible GPU host.

Credentials and model checkpoints are supplied through a copied configuration template. Neither backend contains a fixed account, credential, proxy, model, or host path.
