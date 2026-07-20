# TimePLE Evaluation

This directory contains the public TimePLE inference and evaluation suite. It
uses layered YAML profiles:

```text
suite -> dataset profile + model profile -> prompt profile
```

The suite supports sequential execution, distributed data sharding, resumable
JSONL predictions, shard merging, metric recomputation, and duration-stratified
mIoU for Short `(0, 10]s`, Medium `(10, 30]s`, and Long `(30, inf)` moments.

## Install

```bash
uv sync --extra eval
uv run python scripts/install_overlay.py transformers
```

The TimePLE vLLM adapter reads the public `timeple` package directly. No source
checkout outside this repository is required.

## Data layout

Place benchmark annotations and licensed videos under:

```text
data/benchmarks/
├── charades_sta/
│   ├── charades_sta_test.json
│   └── videos/
├── activitynet_captions/
│   ├── activitynet_captions_val1.json
│   └── videos/
└── qvhighlights/
    ├── highlight_val_release.json
    └── videos/
```

Annotation files may be JSON lists, `{ "data": [...] }` JSON objects, or JSONL.
The default schema is:

```json
{
  "sample_id": "sample-0001",
  "query": "the described event",
  "gt_timestamps": [[1.2, 4.8]],
  "video_path": "video-0001.mp4"
}
```

Edit a dataset profile when your field names or local layout differ. Keep
committed paths repository-relative.

## Model layout

The default profiles expect:

```text
checkpoints/base-model/   # Qwen3-VL-8B-Instruct
checkpoints/sft-stage2/   # exported TimePLE-8B checkpoint
```

The TimePLE checkpoint must contain its Hugging Face `config.json`, processor
files, model weights, and `timeple_codec.pth` when the codec is saved separately.

## Run

Render a suite without initializing a model:

```bash
uv run python evaluation/src/run_eval_suite.py \
  --suite evaluation/configs/suites/charades_sta.yaml
```

Execute the rendered jobs:

```bash
SUITE=charades_sta bash scripts/eval/run_suite.sh
SUITE=activitynet_captions bash scripts/eval/run_suite.sh
SUITE=qvhighlights bash scripts/eval/run_suite.sh
```

To evaluate only TimePLE in a suite containing the matched baseline:

```bash
SUITE=charades_sta bash scripts/eval/run_suite.sh --models timeple_8b
```

Outputs are written to:

```text
evaluation/outputs/<benchmark>/<job>/<run_timestamp>/
├── job_manifest.yaml
├── predictions.jsonl
├── failed_samples.jsonl
├── metrics.json
└── run.log
```

## Distributed execution

The worker automatically reads standard rank variables from torchrun, MPI, and
common cluster launchers when `distributed.auto_from_env=true`. Each rank writes
an independent prediction and metric shard. Merge a completed run with:

```bash
uv run python evaluation/src/merge_eval_shards.py \
  --run-dir evaluation/outputs/<benchmark>/<job>/<run_timestamp>
```

Use `evaluation/src/recompute_metrics.py` to recompute metrics from saved
predictions after changing only metric thresholds or reporting logic.

## Public inference settings

The default TimePLE profile follows the paper evaluation setting:

- 2 FPS video sampling;
- at most 200 sampled frames;
- at most 64 visual tokens per frame;
- deterministic decoding;
- duration groups `(0, 10]s`, `(10, 30]s`, and `(30, inf)`.

Adjust GPU memory and batching fields in
`evaluation/configs/models/timeple_8b.yaml` for the target hardware without
changing the benchmark protocol.
