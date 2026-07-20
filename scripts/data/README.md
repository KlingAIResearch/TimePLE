# Data preparation

`prepare_sft.py` accepts common temporal-grounding JSONL records and reconstructs a public, minimal SFT schema. It deliberately discards source names, original IDs, media paths, provenance fields, and all unrecognized metadata.

`prepare_rl.py` converts that SFT schema into the `prompt`, `videos`, and `ground_truth` fields consumed by EasyR1.

```bash
python scripts/data/prepare_sft.py --input /path/to/private.jsonl --output data/sft.jsonl --count 100
python scripts/data/prepare_rl.py --input data/sft.jsonl --output data/rl_train.jsonl
```

The rewritten media names are placeholders. Video files are not distributed in the public repository.
