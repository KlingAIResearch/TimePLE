from __future__ import annotations

import json
from pathlib import Path
import re

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
        assert actor["model"]["model_path"] == "./checkpoints/TimePLE-8B"
        assert actor["model"]["trust_remote_code"] is False


def test_timeple_eval_uses_installed_package_instead_of_remote_code() -> None:
    config = yaml.safe_load((ROOT / "evaluation/configs/models/timeple_8b.yaml").read_text())
    assert config["model"]["eval"]["config"]["inference"]["trust_remote_code"] is False


def test_sft_configs_use_project_relative_paths_and_split_validation() -> None:
    for filename in ("timeple_sft_stage1.yaml", "timeple_sft_stage2.yaml"):
        config = yaml.safe_load((ROOT / "configs/sft" / filename).read_text())
        assert config["data"]["dataset_path"] == "./data/TimePLE-Dataset/train/timeple_train.jsonl"
        assert config["data"]["video_base_dir"] == "./data/TimePLE-Dataset"
        assert config["training"]["split_dataset_ratio"] > 0
        assert "val_dataset" not in config["training"]


def test_integration_manifests_are_exact_and_sources_are_not_vendored() -> None:
    expected = {
        "transformers": ("transformers", "4.57.6"),
        "ms_swift": ("ms-swift", "3.12.6"),
        "easyr1": ("verl", "0.3.3.dev0"),
    }
    sha256_pattern = re.compile(r"^[0-9a-f]{64}$")
    for integration, (distribution, version) in expected.items():
        integration_dir = ROOT / "integrations" / integration
        manifest = json.loads((integration_dir / "manifest.json").read_text())
        assert manifest["distribution"] == distribution
        assert manifest["version"] == version
        assert (integration_dir / manifest["patch"]).is_file()
        assert not list((integration_dir / "overlay").rglob("*.py"))
        assert not list((integration_dir / "files").rglob("*.py"))
        assert all(
            sha256_pattern.fullmatch(value)
            for hashes in manifest["modified_files"].values()
            for value in hashes.values()
        )
        assert all(sha256_pattern.fullmatch(value) for value in manifest["created_files"].values())


def test_integration_patches_declare_modifications_and_include_known_fix() -> None:
    for integration in ("transformers", "ms_swift", "easyr1"):
        integration_dir = ROOT / "integrations" / integration
        manifest = json.loads((integration_dir / "manifest.json").read_text())
        patch = (integration_dir / manifest["patch"]).read_text()
        assert patch.count("Modified by the TimePLE project") >= len(manifest["modified_files"])

    easyr1_patch = (ROOT / "integrations/easyr1/patches/timeple.patch").read_text()
    assert "+    sample_idx = int(sample_info.get(\"sample_idx\", 0))" in easyr1_patch


def test_build_metadata_allows_and_pins_direct_dependency() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert "allow-direct-references = true" in pyproject
    assert "EasyR1.git@07cae10c28d686a6604546617663d32e4f1089e6" in pyproject
    assert '"transformers==4.57.6"' in pyproject
    assert '"ms-swift==3.12.6"' in pyproject


def test_timeple_config_loader_uses_timeple_root() -> None:
    patch = (ROOT / "integrations/ms_swift/patches/timeple.patch").read_text()
    assert 'os.environ.get("TIMEPLE_ROOT", base_dir)' in patch
