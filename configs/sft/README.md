# SFT configurations

- `timeple_sft_stage1.yaml` freezes TimePLE geometry and trains the language-side interface.
- `timeple_sft_stage2.yaml` resumes from stage 1 and unfreezes the TimePLE encoder/decoder.

All data and checkpoint fields use stable project-relative paths. The full annotation package is expected at `data/TimePLE-Dataset`, which is intentionally ignored by Git.
The TimePLE launchers export `TIMEPLE_ROOT`, and the overlay resolves these
paths from that project root rather than from the `configs/sft` directory.
Both stages build their validation set through the configured deterministic
split of `data/TimePLE-Dataset/train/timeple_train.jsonl`.
