# Third-party notices

TimePLE is distributed under the Apache License 2.0. Its optional integration patches target the following Apache-2.0 projects:

| Integration | Exact upstream | Repository |
|---|---|---|
| VLM | Transformers `4.57.6` | https://github.com/huggingface/transformers |
| SFT | ms-swift `3.12.6` | https://github.com/modelscope/ms-swift |
| RL | EasyR1/verl `07cae10c28d686a6604546617663d32e4f1089e6` | https://github.com/hiyouga/EasyR1 |

Complete upstream source trees are not distributed in this repository. They are installed from their official package or source repositories when a TimePLE environment is built. The integration manifests record source hashes, patched hashes, versions, and revisions. Applied patches retain upstream copyright notices and add a prominent TimePLE modification notice to every changed upstream file.

No pretrained model weights are distributed. The separately licensed TimePLE
dataset release, including its WebDataset video shards, is documented under
`data/TimePLE-Dataset/` and is made available under CC BY-NC-SA 4.0.
