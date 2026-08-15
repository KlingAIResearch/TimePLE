# EasyR1 integration

This integration targets EasyR1/verl commit `07cae10c28d686a6604546617663d32e4f1089e6` exactly. The builder checks both the installed distribution version and VCS commit metadata before applying the patch.

```bash
bash scripts/setup_env.sh rl
.venv/bin/python scripts/build_integration.py easyr1 --check
```
