# Transformers integration

This integration targets Transformers `4.57.6` exactly. `manifest.json` records the expected upstream and patched hashes; `patches/timeple.patch` adds the TimePLE Qwen3-VL model/processor integration during environment setup.

```bash
bash scripts/setup_env.sh eval
.venv/bin/python scripts/build_integration.py transformers --check
```
