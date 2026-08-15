#!/usr/bin/env python3
"""Convert the selected legacy CIS stage-2 checkpoint into a TimePLE model.

The legacy and TimePLE implementations use the same codec and interface-adapter
architectures, but their config fields, processor classes, and state-dict
prefixes differ.  This script performs that namespace migration without loading
the 8B base model into memory.  Safetensors payloads are copied as streams and
only their headers/offsets are rewritten.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


SOURCE_MODEL_TYPE = "qwen3_vl_cis_span_duration_adaptive_codec"
TARGET_MODEL_TYPE = "qwen3_vl_timeple"
SOURCE_ARCHITECTURE = "Qwen3VLForConditionalGenerationWithCISSpanDurationAdaptiveCodec"
TARGET_ARCHITECTURE = "Qwen3VLForConditionalGenerationWithTimePLECodec"
SOURCE_PROCESSOR = "Qwen3VLProcessorWithCISSpanDurationAdaptiveCodec"
TARGET_PROCESSOR = "Qwen3VLProcessorWithTimePLECodec"

SOURCE_CODEC_PREFIX = "cis_span_duration_adaptive_codec."
TARGET_CODEC_PREFIX = "timeple_codec."
SOURCE_ADAPTER_PREFIX = "cis_span_duration_adaptive_interface_adapter."
TARGET_ADAPTER_PREFIX = "timeple_interface_adapter."

# ``tokenizer.add_special_tokens`` replaces ``additional_special_tokens`` by
# default.  The legacy training loader added the TimePLE tokens that way, so
# its saved tokenizer retained the Qwen tokens in the vocabulary but lost them
# from the public special-token list.  A converted tokenizer must preserve the
# original Qwen3-VL list and append the two TimePLE tokens.
QWEN3_VL_ADDITIONAL_SPECIAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
)
TIMEPLE_ADDITIONAL_SPECIAL_TOKENS = ("<|TIMESTAMP|>", "<|TIMESPAN|>")
ALL_ADDITIONAL_SPECIAL_TOKENS = QWEN3_VL_ADDITIONAL_SPECIAL_TOKENS + TIMEPLE_ADDITIONAL_SPECIAL_TOKENS

RELEASE_METADATA_FILES = {
    ".gitattributes",
    "LICENSE",
    "README.md",
    "requirements.txt",
}

# These buffers came from an older adapter implementation.  The released
# TimePLE MLP projector does not define or consume them.
LEGACY_ADAPTER_STATS = {
    f"{TARGET_ADAPTER_PREFIX}stats.codec_mean",
    f"{TARGET_ADAPTER_PREFIX}stats.codec_std",
    f"{TARGET_ADAPTER_PREFIX}stats.input_ref_mean",
    f"{TARGET_ADAPTER_PREFIX}stats.input_ref_std",
}

INFERENCE_ASSETS = (
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "video_preprocessor_config.json",
    "vocab.json",
)


class ConversionError(RuntimeError):
    """Raised when a source checkpoint cannot be converted safely."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"Unable to read JSON from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConversionError(f"Expected a JSON object in {path}.")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _rename_weight(name: str) -> str:
    if name.startswith(SOURCE_CODEC_PREFIX):
        return TARGET_CODEC_PREFIX + name.removeprefix(SOURCE_CODEC_PREFIX)
    if name.startswith(SOURCE_ADAPTER_PREFIX):
        return TARGET_ADAPTER_PREFIX + name.removeprefix(SOURCE_ADAPTER_PREFIX)
    return name


def _keep_weight(name: str) -> bool:
    return name not in LEGACY_ADAPTER_STATS


def _replace_processor_names(value: Any) -> Any:
    if isinstance(value, dict):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            target_key = key.replace("use_cis_span_duration_adaptive_codec", "use_timeple_codec")
            converted[target_key] = _replace_processor_names(item)
        return converted
    if isinstance(value, list):
        return [_replace_processor_names(item) for item in value]
    if isinstance(value, str):
        return value.replace(SOURCE_PROCESSOR, TARGET_PROCESSOR)
    return value


def _convert_config(source: Path, destination: Path) -> None:
    config = _read_json(source)
    if config.get("model_type") != SOURCE_MODEL_TYPE:
        raise ConversionError(
            f"Expected source model_type={SOURCE_MODEL_TYPE!r}, got {config.get('model_type')!r}."
        )
    architectures = config.get("architectures")
    if architectures != [SOURCE_ARCHITECTURE]:
        raise ConversionError(f"Unexpected source architectures: {architectures!r}.")

    field_mapping = {
        "cis_span_duration_adaptive_codec_config": "timeple_codec_config",
        "cis_span_duration_adaptive_interface_adapter": "timeple_interface_adapter",
        "use_cis_span_duration_adaptive_codec": "use_timeple_codec",
        "use_cis_span_duration_adaptive_interface_adapter": "use_timeple_interface_adapter",
    }
    for source_name, target_name in field_mapping.items():
        if source_name not in config:
            raise ConversionError(f"Required source config field is missing: {source_name}")
        config[target_name] = config.pop(source_name)

    config["model_type"] = TARGET_MODEL_TYPE
    config["architectures"] = [TARGET_ARCHITECTURE]
    config.pop("auto_map", None)
    # Keep the serialization/training version recorded by the selected source
    # checkpoint.  This field describes the artifact; requirements.txt pins
    # the independently validated release runtime.
    config["transformers_version"] = "4.57.1"
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ConversionError("Source config has no valid text_config object.")
    # Training disables KV caching for gradient checkpointing.  The release is
    # inference-oriented and should retain the base model's efficient default.
    text_config["use_cache"] = True
    _write_json(destination, config)


def _write_external_package_release_metadata(output_dir: Path) -> None:
    """Write metadata for a weights-only Hub repository.

    The executable implementation is distributed by the separately installed
    ``timeple`` package and must not be duplicated in the model repository.
    """

    root = Path(__file__).resolve().parents[2]
    shutil.copy2(root / "LICENSE", output_dir / "LICENSE")
    (output_dir / ".gitattributes").write_text(
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    (output_dir / "requirements.txt").write_text(
        "transformers==4.57.6\n"
        "torch>=2.4\n"
        "torchvision>=0.19\n"
        "accelerate>=1.0\n"
        "numpy>=1.26\n"
        "pillow>=10\n"
        "qwen-vl-utils[decord]>=0.0.14\n",
        encoding="utf-8",
    )


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ConversionError(f"Invalid safetensors file (missing header length): {path}")
        header_length = struct.unpack("<Q", prefix)[0]
        header_bytes = handle.read(header_length)
    if len(header_bytes) != header_length:
        raise ConversionError(f"Invalid safetensors file (truncated header): {path}")
    try:
        header = json.loads(header_bytes.rstrip(b" "))
    except json.JSONDecodeError as exc:
        raise ConversionError(f"Invalid safetensors JSON header in {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ConversionError(f"Expected a JSON object in the safetensors header: {path}")
    return header_length, header


def _copy_exact(source: Any, destination: Any, byte_count: int, buffer_size: int = 16 * 1024 * 1024) -> None:
    remaining = byte_count
    while remaining:
        chunk = source.read(min(buffer_size, remaining))
        if not chunk:
            raise ConversionError(f"Unexpected end of file with {remaining} bytes left to copy.")
        destination.write(chunk)
        remaining -= len(chunk)


def _convert_safetensors(
    source: Path,
    destination: Path,
    rename: Callable[[str], str],
    keep: Callable[[str], bool],
) -> tuple[dict[str, str], dict[str, dict[str, Any]], list[str]]:
    """Stream a safetensors file while renaming and optionally dropping tensors."""

    source_header_length, source_header = _read_safetensors_header(source)
    source_data_start = 8 + source_header_length
    metadata = source_header.get("__metadata__")
    tensors: list[tuple[int, int, str, str, dict[str, Any]]] = []
    dropped: list[str] = []
    name_mapping: dict[str, str] = {}

    for source_name, raw_spec in source_header.items():
        if source_name == "__metadata__":
            continue
        if not isinstance(raw_spec, dict) or "data_offsets" not in raw_spec:
            raise ConversionError(f"Malformed tensor entry {source_name!r} in {source}.")
        offsets = raw_spec["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ConversionError(f"Malformed data offsets for {source_name!r} in {source}.")
        start, end = int(offsets[0]), int(offsets[1])
        target_name = rename(source_name)
        if target_name in name_mapping.values():
            raise ConversionError(f"Weight rename collision for {target_name!r}.")
        if not keep(target_name):
            dropped.append(target_name)
            continue
        name_mapping[source_name] = target_name
        tensors.append((start, end, source_name, target_name, raw_spec))

    tensors.sort(key=lambda item: item[0])
    target_header: dict[str, Any] = {}
    if metadata is not None:
        target_header["__metadata__"] = metadata
    target_offset = 0
    converted_specs: dict[str, dict[str, Any]] = {}
    for start, end, _source_name, target_name, raw_spec in tensors:
        if start < 0 or end < start:
            raise ConversionError(f"Invalid data offsets [{start}, {end}] in {source}.")
        spec = dict(raw_spec)
        tensor_size = end - start
        spec["data_offsets"] = [target_offset, target_offset + tensor_size]
        target_header[target_name] = spec
        converted_specs[target_name] = spec
        target_offset += tensor_size

    encoded_header = json.dumps(target_header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded_header += b" " * (-len(encoded_header) % 8)

    with source.open("rb") as source_handle, destination.open("wb") as destination_handle:
        destination_handle.write(struct.pack("<Q", len(encoded_header)))
        destination_handle.write(encoded_header)
        for start, end, _source_name, _target_name, _raw_spec in tensors:
            source_handle.seek(source_data_start + start)
            _copy_exact(source_handle, destination_handle, end - start)

    shutil.copystat(source, destination)
    expected_size = 8 + len(encoded_header) + target_offset
    actual_size = destination.stat().st_size
    if actual_size != expected_size:
        raise ConversionError(
            f"Converted safetensors size mismatch for {destination}: expected {expected_size}, got {actual_size}."
        )
    return name_mapping, converted_specs, dropped


def _convert_processor_assets(source_dir: Path, output_dir: Path) -> None:
    for filename in INFERENCE_ASSETS:
        source = source_dir / filename
        if not source.is_file():
            raise ConversionError(f"Required inference asset is missing: {source}")
        destination = output_dir / filename
        if source.suffix == ".json":
            data = _read_json(source)
            converted = _replace_processor_names(data)
            if not isinstance(converted, dict):
                raise AssertionError("Processor JSON conversion must preserve the top-level object.")
            if filename in {"special_tokens_map.json", "tokenizer_config.json"}:
                converted["additional_special_tokens"] = list(ALL_ADDITIONAL_SPECIAL_TOKENS)
            converted.pop("auto_map", None)
            if filename == "generation_config.json":
                converted["transformers_version"] = "4.57.1"
            _write_json(destination, converted)
        else:
            shutil.copy2(source, destination)


def _validate_tokenizer(output_dir: Path) -> None:
    added = _read_json(output_dir / "added_tokens.json")
    expected = {"<|TIMESTAMP|>": 151669, "<|TIMESPAN|>": 151670}
    for token, token_id in expected.items():
        if added.get(token) != token_id:
            raise ConversionError(f"Expected {token} to have token ID {token_id}, got {added.get(token)!r}.")

    special_map = _read_json(output_dir / "special_tokens_map.json")
    special_tokens = special_map.get("additional_special_tokens", [])
    if special_tokens != list(ALL_ADDITIONAL_SPECIAL_TOKENS):
        raise ConversionError(
            "special_tokens_map.json does not preserve the Qwen3-VL tokens followed by the TimePLE tokens."
        )

    tokenizer_config = _read_json(output_dir / "tokenizer_config.json")
    if tokenizer_config.get("processor_class") != TARGET_PROCESSOR:
        raise ConversionError("tokenizer_config.json does not reference the TimePLE processor.")
    if "auto_map" in tokenizer_config:
        raise ConversionError("tokenizer_config.json must not embed remote-code AutoProcessor mappings.")
    if tokenizer_config.get("additional_special_tokens") != list(ALL_ADDITIONAL_SPECIAL_TOKENS):
        raise ConversionError("tokenizer_config.json has an incomplete additional_special_tokens list.")


def _validate_output(output_dir: Path) -> None:
    config = _read_json(output_dir / "config.json")
    if config.get("model_type") != TARGET_MODEL_TYPE:
        raise ConversionError("Converted config has the wrong model_type.")
    if config.get("architectures") != [TARGET_ARCHITECTURE]:
        raise ConversionError("Converted config has the wrong architecture.")
    if config.get("timespan_token_id") != 151670 or config.get("timestamp_token_id") != 151669:
        raise ConversionError("Converted config has incorrect temporal token IDs.")
    if not config.get("use_timeple_codec") or not config.get("use_timeple_interface_adapter"):
        raise ConversionError("Converted config does not enable the TimePLE codec and adapter.")
    if "auto_map" in config:
        raise ConversionError("Converted config must not embed remote-code AutoClass mappings.")
    if (config.get("text_config") or {}).get("use_cache") is not True:
        raise ConversionError("Converted config must enable KV caching for release inference.")

    _validate_tokenizer(output_dir)

    for filename in INFERENCE_ASSETS:
        asset = output_dir / filename
        if asset.suffix == ".json" and "auto_map" in _read_json(asset):
            raise ConversionError(f"Release asset must not embed a remote-code mapping: {filename}")

    index = _read_json(output_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ConversionError("Converted model index has no weight_map.")
    indexed_names = set(weight_map)
    if not any(name.startswith(TARGET_CODEC_PREFIX) for name in indexed_names):
        raise ConversionError("Converted index has no TimePLE codec weights.")
    if not any(name.startswith(TARGET_ADAPTER_PREFIX) for name in indexed_names):
        raise ConversionError("Converted index has no TimePLE adapter weights.")
    if any(name.startswith((SOURCE_CODEC_PREFIX, SOURCE_ADAPTER_PREFIX)) for name in indexed_names):
        raise ConversionError("Converted index still contains legacy CIS weight names.")
    if indexed_names & LEGACY_ADAPTER_STATS:
        raise ConversionError("Converted index still contains obsolete adapter statistics.")

    header_names: set[str] = set()
    for shard_name in sorted(set(weight_map.values())):
        shard = output_dir / shard_name
        if not shard.is_file():
            raise ConversionError(f"Converted model shard is missing: {shard}")
        _header_length, header = _read_safetensors_header(shard)
        names = set(header) - {"__metadata__"}
        duplicate_names = header_names & names
        if duplicate_names:
            raise ConversionError(f"Duplicate tensor names across shards: {sorted(duplicate_names)[:5]}")
        header_names.update(names)
    if header_names != indexed_names:
        missing = sorted(indexed_names - header_names)[:5]
        extra = sorted(header_names - indexed_names)[:5]
        raise ConversionError(f"Index/header mismatch; missing={missing}, extra={extra}.")

    unsafe_weights = sorted(path.name for path in output_dir.iterdir() if path.suffix in {".bin", ".pt", ".pth"})
    if unsafe_weights:
        raise ConversionError(f"Release model contains unsafe pickle-based weights: {unsafe_weights}")
    embedded_python = sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*.py"))
    if embedded_python:
        raise ConversionError(f"Release model must use the external timeple package, not embedded code: {embedded_python}")
    missing_release_metadata = sorted(
        filename for filename in RELEASE_METADATA_FILES if not (output_dir / filename).is_file()
    )
    if missing_release_metadata:
        raise ConversionError(f"Release model is missing release metadata: {missing_release_metadata}")
    if any(path.name == "__pycache__" for path in output_dir.rglob("__pycache__")):
        raise ConversionError("Release model contains a Python bytecode cache directory.")

    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    required_model_card_fields = (
        "license: apache-2.0",
        "library_name: transformers",
        "pipeline_tag: image-text-to-text",
        "base_model: Qwen/Qwen3-VL-8B-Instruct",
        "## Requirements and loading",
        "## Training",
        "## Validation used for checkpoint selection",
        "## Intended use",
        "## Limitations",
        "import timeple",
    )
    missing_model_card_fields = [field for field in required_model_card_fields if field not in readme]
    if missing_model_card_fields:
        raise ConversionError(f"Release model card is incomplete: {missing_model_card_fields}")


def convert_checkpoint(source_dir: Path, output_dir: Path, model_card: Path | None = None) -> None:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if not source_dir.is_dir():
        raise ConversionError(f"Source checkpoint directory does not exist: {source_dir}")
    if output_dir.exists():
        raise ConversionError(f"Output already exists; refusing to overwrite it: {output_dir}")
    if source_dir == output_dir:
        raise ConversionError("Source and output directories must differ.")

    index_path = source_dir / "model.safetensors.index.json"
    source_index = _read_json(index_path)
    source_weight_map = source_index.get("weight_map")
    if not isinstance(source_weight_map, dict) or not source_weight_map:
        raise ConversionError(f"No model weight map found in {index_path}.")

    staging = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    if staging.exists():
        raise ConversionError(f"Staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        _convert_config(source_dir / "config.json", staging / "config.json")
        _convert_processor_assets(source_dir, staging)

        converted_weight_map: dict[str, str] = {}
        converted_total_size = 0
        dropped_names: list[str] = []
        for shard_name in sorted(set(source_weight_map.values())):
            source_shard = source_dir / shard_name
            if not source_shard.is_file():
                raise ConversionError(f"Model shard listed in the index is missing: {source_shard}")
            mapping, specs, dropped = _convert_safetensors(
                source_shard,
                staging / shard_name,
                _rename_weight,
                _keep_weight,
            )
            dropped_names.extend(dropped)
            for source_name, target_name in mapping.items():
                expected_shard = source_weight_map.get(source_name)
                if expected_shard != shard_name:
                    raise ConversionError(
                        f"Index/header shard mismatch for {source_name}: {expected_shard!r} vs {shard_name!r}."
                    )
                converted_weight_map[target_name] = shard_name
                start, end = specs[target_name]["data_offsets"]
                converted_total_size += int(end) - int(start)

        if set(dropped_names) != LEGACY_ADAPTER_STATS:
            raise ConversionError(
                f"Unexpected legacy adapter statistics: dropped={sorted(dropped_names)}, "
                f"expected={sorted(LEGACY_ADAPTER_STATS)}."
            )
        if len(converted_weight_map) != len(source_weight_map) - len(LEGACY_ADAPTER_STATS):
            raise ConversionError("Converted weight count does not match the expected source weight count.")

        converted_index = dict(source_index)
        metadata = dict(converted_index.get("metadata") or {})
        metadata["total_size"] = converted_total_size
        converted_index["metadata"] = metadata
        converted_index["weight_map"] = converted_weight_map
        _write_json(staging / "model.safetensors.index.json", converted_index)

        _write_external_package_release_metadata(staging)

        source_readme = model_card if model_card is not None else source_dir / "README.md"
        if not source_readme.is_file():
            raise ConversionError(
                "A complete release model card is required; pass it with --model-card when the source checkpoint "
                "has no README.md."
            )
        shutil.copy2(source_readme, staging / "README.md")

        _validate_output(staging)
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Legacy stage-2 checkpoint directory (for example checkpoint-145).")
    parser.add_argument("output", type=Path, help="New TimePLE model directory; it must not already exist.")
    parser.add_argument("--model-card", type=Path, help="Optional README.md to copy into the converted model.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        convert_checkpoint(args.source, args.output, args.model_card)
    except ConversionError as exc:
        print(f"conversion error: {exc}", file=sys.stderr)
        return 2
    print(f"Converted TimePLE model: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
