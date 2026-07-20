from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_model_config_is_valid_json() -> None:
    config = json.loads((ROOT / "configs/model/timeple_codec.json").read_text())
    assert config["token_dim"] > 0
    assert config["decoder"]["duration_adaptive_residual"]["enabled"] is True


def test_rl_configs_use_distinct_public_algorithms() -> None:
    expected_flags = {
        "timeple_single_grpo_iou_dfl_format.yaml": (),
        "timeple_csdo.yaml": ("csdo_enabled",),
        "timeple_tr_spd.yaml": ("tr_spd_enabled",),
    }
    for filename, flags in expected_flags.items():
        config = yaml.safe_load((ROOT / "configs/rl" / filename).read_text())
        actor = config["worker"]["actor"]
        for flag in flags:
            assert actor[flag] is True
        assert str(actor["model"]["model_path"]).startswith("./")


def test_sft_configs_use_project_relative_paths_and_split_validation() -> None:
    for filename in ("timeple_sft_stage1.yaml", "timeple_sft_stage2.yaml"):
        config = yaml.safe_load((ROOT / "configs/sft" / filename).read_text())
        assert config["data"]["dataset_path"] == "./data/sft.jsonl"
        assert config["training"]["split_dataset_ratio"] > 0
        assert "val_dataset" not in config["training"]


def test_timeple_config_loader_uses_timeple_root() -> None:
    source = (ROOT / "integrations/ms_swift/overlay/swift/llm/argument/train_args.py").read_text()
    assert 'os.environ.get("TIMEPLE_ROOT", base_dir)' in source
