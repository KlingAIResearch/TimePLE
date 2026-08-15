# Example records

This directory contains 100 representative, text-only training records in both SFT and EasyR1 formats. Records were reconstructed using a strict output-field whitelist:

- no dataset/source field;
- no original sample identifier;
- no original filesystem or URL;
- no user, institution, host, or collection metadata.

The `videos/sample_XXXX.mp4` values document the expected media interface only. Corresponding videos are intentionally not included.

## Full local dataset

The complete training and benchmark annotation package is released as [KlingTeam/TimePLE-Dataset](https://huggingface.co/datasets/KlingTeam/TimePLE-Dataset) and is not tracked directly by Git. Place or link it at:

```text
data/TimePLE-Dataset/
├── manifest.json
├── README.md
├── train/timeple_train.jsonl
├── benchmark/charades_timeple_test.json
└── videos/                         # optional, user-provided licensed media
```

The public SFT configurations read `data/TimePLE-Dataset/train/timeple_train.jsonl`. Video paths inside the annotations are relative to `data/TimePLE-Dataset/`; licensed videos must be obtained separately from their original providers.
