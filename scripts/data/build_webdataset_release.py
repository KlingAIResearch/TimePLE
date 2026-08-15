#!/usr/bin/env python3
"""Build deterministic, resumable WebDataset shards for the TimePLE release.

Each WebDataset sample represents one unique video and contains two members:

    <release-video-key>.<video-extension>
    <release-video-key>.json

The JSON member contains all annotations associated with that video.  Video
basenames are copied exactly from the public annotation files; source folder
names and local filesystem paths are never written into the shards.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import re
import tarfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "data" / "TimePLE-Dataset"
DEFAULT_TRAIN_MEDIA = REPO_ROOT / "data" / "videos" / "train"
DEFAULT_TRAIN_MEDIA_OVERLAY = REPO_ROOT / "data" / ".timeple_webdataset_mp4"
DEFAULT_TEST_MEDIA = REPO_ROOT / "data" / "videos" / "test"
SAFE_FILENAME = re.compile(r"^[A-Za-z0-9_@+.-]+$")


@dataclass(frozen=True)
class Sample:
    filename: str
    media_path: Path
    metadata: bytes

    @property
    def key(self) -> str:
        return Path(self.filename).stem


@dataclass(frozen=True)
class PlannedShard:
    split: str
    index: int
    samples: tuple[Sample, ...]

    @property
    def filename(self) -> str:
        return f"{self.split}-{self.index:05d}.tar"


class HashingWriter:
    """Hash bytes while tarfile writes them, avoiding a second full read."""

    def __init__(self, raw: BinaryIO) -> None:
        self.raw = raw
        self.hasher = hashlib.sha256()
        self.bytes_written = 0

    def write(self, data: bytes) -> int:
        written = self.raw.write(data)
        if written != len(data):
            raise OSError(f"short write: expected {len(data)}, wrote {written}")
        self.hasher.update(data)
        self.bytes_written += written
        return written

    def flush(self) -> None:
        self.raw.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=DATASET_ROOT, help="annotation package"
    )
    parser.add_argument(
        "--train-media-root",
        type=Path,
        default=DEFAULT_TRAIN_MEDIA,
        help="flat directory containing release-named training videos",
    )
    parser.add_argument(
        "--train-media-overlay",
        type=Path,
        default=DEFAULT_TRAIN_MEDIA_OVERLAY,
        help="flat directory containing normalized release media overrides",
    )
    parser.add_argument(
        "--test-media-root",
        type=Path,
        default=DEFAULT_TEST_MEDIA,
        help="directory containing Charades-TimePLE videos",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATASET_ROOT / "webdataset",
        help="output directory",
    )
    parser.add_argument(
        "--shard-size-gb",
        type=float,
        default=1.0,
        help="target maximum shard size in decimal GB (default: 1.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print the shard plan without writing",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of shards to write concurrently (default: 4)",
    )
    parser.add_argument(
        "--discard-incomplete",
        action="store_true",
        help="remove .incomplete shards left by an interrupted earlier run",
    )
    return parser.parse_args()


def public_filename(value: str) -> str:
    path = Path(value)
    name = path.name
    if not name or name in {".", ".."} or not SAFE_FILENAME.fullmatch(name):
        raise ValueError(f"unsafe release video filename: {value!r}")
    if name.count(".") != 1:
        raise ValueError(f"video filename must have one extension: {value!r}")
    return name


def build_media_index(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"media root is not a directory: {root}")
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for directory, _, filenames in os.walk(root):
        base = Path(directory)
        for filename in filenames:
            path = base / filename
            if filename in index:
                duplicates.setdefault(filename, [index[filename]]).append(path)
            else:
                index[filename] = path
    if duplicates:
        examples = ", ".join(sorted(duplicates)[:5])
        raise ValueError(f"ambiguous media basenames under {root}: {examples}")
    return index


def merge_media_index(index: dict[str, Path], overlay_root: Path) -> None:
    """Add normalized media without allowing ambiguous public basenames."""
    overlay = build_media_index(overlay_root)
    collisions = sorted(index.keys() & overlay.keys())
    if collisions:
        raise ValueError(
            f"media overlay collides with primary media: {', '.join(collisions[:5])}"
        )
    index.update(overlay)


def compact_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def resolve_groups(
    groups: OrderedDict[str, list[dict[str, object]]],
    media_index: dict[str, Path],
) -> list[Sample]:
    missing: list[str] = []
    empty: list[str] = []
    samples: list[Sample] = []
    for filename, annotations in groups.items():
        media_path = media_index.get(filename)
        if media_path is None:
            missing.append(filename)
            continue
        if media_path.stat().st_size == 0:
            empty.append(filename)
            continue
        metadata = compact_json(
            {"annotations": annotations, "video_filename": filename}
        )
        samples.append(Sample(filename, media_path, metadata))
    if missing or empty:
        details = []
        if missing:
            details.append(
                f"missing={len(missing)} (examples: {', '.join(missing[:5])})"
            )
        if empty:
            details.append(f"empty={len(empty)} (examples: {', '.join(empty[:5])})")
        raise FileNotFoundError("media preflight failed: " + "; ".join(details))
    return sorted(samples, key=lambda sample: sample.filename)


def load_train(dataset_root: Path, media_index: dict[str, Path]) -> tuple[list[Sample], int]:
    path = dataset_root / "train" / "timeple_train.jsonl"
    groups: OrderedDict[str, list[dict[str, object]]] = OrderedDict()
    annotation_count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            videos = row.get("videos")
            if not isinstance(videos, list) or len(videos) != 1:
                raise ValueError(f"{path}:{line_number}: expected exactly one video")
            filename = public_filename(videos[0])
            user_messages = [
                message["content"]
                for message in row["messages"]
                if message.get("role") == "user"
            ]
            assistant_messages = [
                message["content"]
                for message in row["messages"]
                if message.get("role") == "assistant"
            ]
            if len(user_messages) != 1 or len(assistant_messages) != 1:
                raise ValueError(
                    f"{path}:{line_number}: expected one user and one assistant message"
                )
            if not user_messages[0].startswith("<video>"):
                raise ValueError(f"{path}:{line_number}: user query must start with <video>")
            annotation = {
                "answer": assistant_messages[0],
                "query": user_messages[0][len("<video>") :],
                "sample_id": row["sample_id"],
                "timestamps": row["time_gt"],
            }
            groups.setdefault(filename, []).append(annotation)
            annotation_count += 1
    return resolve_groups(groups, media_index), annotation_count


def load_test(dataset_root: Path, media_index: dict[str, Path]) -> tuple[list[Sample], int]:
    path = dataset_root / "benchmark" / "charades_timeple_test.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    groups: OrderedDict[str, list[dict[str, object]]] = OrderedDict()
    for row in rows:
        filename = public_filename(row["video_path"])
        annotation = {
            "answer": "",
            "query": row["query"],
            "sample_id": row["sample_id"],
            "timestamps": row["gt_timestamps"],
        }
        groups.setdefault(filename, []).append(annotation)
    return resolve_groups(groups, media_index), len(rows)


def padded_tar_member_size(size: int) -> int:
    return 512 + ((size + 511) // 512) * 512


def plan_shards(
    split: str, samples: list[Sample], target_bytes: int
) -> list[PlannedShard]:
    shards: list[PlannedShard] = []
    current: list[Sample] = []
    current_bytes = 0
    # tarfile pads stream archives to a 10 KiB record boundary on close.
    closing_allowance = 10 * 1024
    for sample in samples:
        sample_bytes = padded_tar_member_size(sample.media_path.stat().st_size)
        sample_bytes += padded_tar_member_size(len(sample.metadata))
        if current and current_bytes + sample_bytes + closing_allowance > target_bytes:
            shards.append(PlannedShard(split, len(shards), tuple(current)))
            current = []
            current_bytes = 0
        current.append(sample)
        current_bytes += sample_bytes
    if current:
        shards.append(PlannedShard(split, len(shards), tuple(current)))
    return shards


def tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def expected_members(shard: PlannedShard) -> list[tuple[str, int]]:
    members: list[tuple[str, int]] = []
    for sample in shard.samples:
        members.append((sample.filename, sample.media_path.stat().st_size))
        members.append((f"{sample.key}.json", len(sample.metadata)))
    return members


def validate_existing_shard(path: Path, shard: PlannedShard) -> None:
    expected = expected_members(shard)
    with tarfile.open(path, mode="r:") as archive:
        actual = [(member.name, member.size) for member in archive]
    if actual != expected:
        raise ValueError(
            f"existing shard does not match the deterministic plan: {path}; "
            "remove only this shard and rerun"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_shard(
    output_dir: Path, shard: PlannedShard, discard_incomplete: bool = False
) -> dict[str, object]:
    split_dir = output_dir / shard.split
    split_dir.mkdir(parents=True, exist_ok=True)
    final_path = split_dir / shard.filename
    relative_path = final_path.relative_to(output_dir).as_posix()
    if final_path.exists():
        validate_existing_shard(final_path, shard)
        return {
            "file": relative_path,
            "samples": len(shard.samples),
            "bytes": final_path.stat().st_size,
            "sha256": sha256_file(final_path),
            "first_key": shard.samples[0].key,
            "last_key": shard.samples[-1].key,
        }

    partial_path = final_path.with_suffix(".tar.incomplete")
    if partial_path.exists():
        if not discard_incomplete:
            raise FileExistsError(
                f"incomplete shard exists: {partial_path}; rerun with "
                "--discard-incomplete to rebuild it"
            )
        partial_path.unlink()

    with partial_path.open("xb") as raw:
        writer = HashingWriter(raw)
        with tarfile.open(fileobj=writer, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for sample in shard.samples:
                media_size = sample.media_path.stat().st_size
                with sample.media_path.open("rb") as media:
                    archive.addfile(tar_info(sample.filename, media_size), media)
                archive.addfile(
                    tar_info(f"{sample.key}.json", len(sample.metadata)),
                    io.BytesIO(sample.metadata),
                )
        writer.flush()
        os.fsync(raw.fileno())
        digest = writer.hasher.hexdigest()
        bytes_written = writer.bytes_written
    partial_path.replace(final_path)
    return {
        "file": relative_path,
        "samples": len(shard.samples),
        "bytes": bytes_written,
        "sha256": digest,
        "first_key": shard.samples[0].key,
        "last_key": shard.samples[-1].key,
    }


def split_manifest(
    split: str,
    samples: list[Sample],
    annotation_count: int,
    shard_records: Iterable[dict[str, object]],
) -> dict[str, object]:
    media_bytes = sum(sample.media_path.stat().st_size for sample in samples)
    extensions: dict[str, int] = {}
    for sample in samples:
        extension = Path(sample.filename).suffix.lower().lstrip(".")
        extensions[extension] = extensions.get(extension, 0) + 1
    return {
        "annotations": annotation_count,
        "unique_videos": len(samples),
        "video_bytes": media_bytes,
        "video_extensions": dict(sorted(extensions.items())),
        "shards": list(shard_records),
    }


def write_manifest(output_dir: Path, manifest: dict[str, object]) -> None:
    path = output_dir / "webdataset_manifest.json"
    temporary = path.with_suffix(".json.incomplete")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.shard_size_gb <= 0:
        raise ValueError("--shard-size-gb must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    target_bytes = int(args.shard_size_gb * 1_000_000_000)

    print(f"Indexing training media: {args.train_media_root}", flush=True)
    train_index = build_media_index(args.train_media_root)
    print(f"Indexing normalized media: {args.train_media_overlay}", flush=True)
    merge_media_index(train_index, args.train_media_overlay)
    print(f"Indexing test media: {args.test_media_root}", flush=True)
    test_index = build_media_index(args.test_media_root)
    train_samples, train_annotations = load_train(args.dataset_root, train_index)
    test_samples, test_annotations = load_test(args.dataset_root, test_index)

    train_names = {sample.filename for sample in train_samples}
    test_names = {sample.filename for sample in test_samples}
    overlap = train_names & test_names
    if overlap:
        raise ValueError(f"train/test video basename collision: {sorted(overlap)[:5]}")

    plans = {
        "train": plan_shards("train", train_samples, target_bytes),
        "test": plan_shards("test", test_samples, target_bytes),
    }
    total_bytes = sum(
        sample.media_path.stat().st_size
        for sample in train_samples + test_samples
    )
    print(
        f"Preflight passed: train={len(train_samples)} videos/{train_annotations} annotations, "
        f"test={len(test_samples)} videos/{test_annotations} annotations, "
        f"media={total_bytes / 1_000_000_000:.3f} GB, "
        f"shards={len(plans['train']) + len(plans['test'])}",
        flush=True,
    )
    if args.dry_run:
        return

    args.output.mkdir(parents=True, exist_ok=True)
    records: dict[str, list[dict[str, object]]] = {"train": [], "test": []}
    for split in ("train", "test"):
        split_records: list[dict[str, object] | None] = [None] * len(plans[split])
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    write_shard, args.output, shard, args.discard_incomplete
                ): position
                for position, shard in enumerate(plans[split])
            }
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                position = futures[future]
                split_records[position] = future.result()
                completed += 1
                shard = plans[split][position]
                print(
                    f"[{split} {completed}/{len(plans[split])}] {shard.filename} "
                    f"complete ({len(shard.samples)} videos)",
                    flush=True,
                )
        if any(record is None for record in split_records):
            raise RuntimeError(f"not all {split} shards completed")
        records[split] = [record for record in split_records if record is not None]

    manifest = {
        "format": "WebDataset",
        "license": "cc-by-nc-sa-4.0",
        "sample_unit": "one unique video with all associated temporal annotations",
        "annotation_schema": {
            "answer": "string; empty when no reference response is released",
            "query": "string",
            "sample_id": "string",
            "timestamps": "list of [start_seconds, end_seconds]",
        },
        "member_naming": (
            "Video basenames are preserved exactly from the public annotations; "
            "the sidecar JSON uses the same basename stem."
        ),
        "shard_target_bytes": target_bytes,
        "splits": {
            "train": split_manifest(
                "train", train_samples, train_annotations, records["train"]
            ),
            "test": split_manifest(
                "test", test_samples, test_annotations, records["test"]
            ),
        },
    }
    write_manifest(args.output, manifest)
    print(f"Complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
