# Local checkpoints

Model weights are intentionally excluded from Git. Environment setup should expose the selected artifacts through these stable local paths:

```text
checkpoints/
├── base-model/       # upstream Qwen3-VL base model
├── geometry/         # selected TimePLE geometry checkpoint
├── sft-stage1/       # selected stage-1 checkpoint
└── TimePLE-8B/       # final release-ready SFT model
```

Model directories placed here are ignored. Copy the final Hugging Face model package to `checkpoints/TimePLE-8B` without committing its weights to Git. The release directory contains weights and inference assets only; install and import the `timeple` source package to register its Transformers classes.
