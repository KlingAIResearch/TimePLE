# VTG Deterministic Deduplication

This directory performs annotation-only preprocessing before the shared training-data pipeline:

1. Convert Charades-STA or ActivityNet-Captions to the common by-video schema.
2. Build high-overlap candidate components from annotation IoU.
3. Keep one deterministic representative per connected duplicate component.
4. Convert the result to training JSONL.

No model prompt is defined or invoked here. In particular, model outputs do not decide grouping, representative selection, query rewriting, or temporal-boundary rewriting. All model-assisted processing uses the canonical training-data prompt in `train_building/pipeline_core.py` after this stage.

```bash
bash data_pipeline/run_pipeline.sh vtg charades_sta all
bash data_pipeline/run_pipeline.sh vtg activitynet_captions all
```

Environment overrides:

- `CHARADES_INPUT_PATH`
- `ACTIVITYNET_INPUT_PATH`
- `VTG_OUTPUT_DIR`
- `VTG_DEDUP_IOU_THRESHOLD` (default `0.99`)
- `VTG_REPRESENTATIVE_POLICY` (`longest_query`, `longest_duration`, or `first`)
- `PYTHON_BIN`

The deduplicated JSON contains a complete `deduplication.audit` section recording every selected and removed query identifier.
