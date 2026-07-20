# Third-party notices

TimePLE is distributed under the Apache License 2.0. Some integration files are modified snapshots of public Apache-2.0 projects and retain their upstream copyright headers.

| Component | Upstream project | Use in this archive | License |
|---|---|---|---|
| SFT integration | ModelScope ms-swift | Modified argument, model, dataset, template, and trainer overlays | Apache-2.0 |
| RL integration | EasyR1 / verl | Modified actor, trainer, rollout, dataset, and utility overlays | Apache-2.0 |
| VLM integration | Hugging Face Transformers and Qwen3-VL integration | Modified Qwen3-VL configuration, processing, and modeling overlays | Apache-2.0 |

The complete upstream libraries are not bundled. They are declared as environment dependencies in `pyproject.toml`; the TimePLE overlay installer applies only the files needed to expose the method.

No pretrained model weights, videos, or third-party datasets are distributed. Users must obtain those artifacts under their respective licenses and terms.
