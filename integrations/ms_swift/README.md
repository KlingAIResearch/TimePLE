# ms-swift integration

This integration targets ms-swift `3.12.6` exactly. It is applied only after the Transformers integration has been built.

```bash
bash scripts/setup_env.sh sft
.venv/bin/python scripts/build_integration.py ms-swift --check
```
