#!/usr/bin/env python3
"""Remux release-referenced MKV videos to MP4 and update annotation paths.

The operation copies encoded streams without re-encoding. Public filename stems
remain unchanged; only the container extension changes from ``.mkv`` to
``.mp4``. Output media is stored outside the release folder so it is not
accidentally uploaded as a second copy.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "data" / "TimePLE-Dataset"
DEFAULT_MEDIA_ROOT = REPO_ROOT / "data" / "videos" / "train"
DEFAULT_OUTPUT = REPO_ROOT / "data" / ".timeple_webdataset_mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def probe(path: Path) -> tuple[str, float]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,codec_name:format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"ffprobe failed for {path}: {completed.stderr.strip()}")
    result = json.loads(completed.stdout)
    video_codecs = [
        stream.get("codec_name")
        for stream in result.get("streams", [])
        if stream.get("codec_type") == "video"
    ]
    if len(video_codecs) != 1:
        raise ValueError(f"expected one video stream in {path}, found {video_codecs}")
    return video_codecs[0], float(result["format"]["duration"])


def remux(source: Path, target: Path) -> tuple[str, int]:
    source_codec, source_duration = probe(source)
    if target.exists():
        target_codec, target_duration = probe(target)
        if target.stat().st_size == 0 or target_codec != source_codec:
            raise ValueError(f"invalid existing remux output: {target}")
        if abs(target_duration - source_duration) > 0.05:
            raise ValueError(f"duration mismatch for existing remux output: {target}")
        return target.name, target.stat().st_size

    partial = target.with_suffix(".mp4.incomplete")
    if partial.exists():
        partial.unlink()
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-c",
        "copy",
        "-map_metadata",
        "-1",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        "-y",
        str(partial),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"ffmpeg failed for {source}: {completed.stderr.strip()}")
    target_codec, target_duration = probe(partial)
    if partial.stat().st_size == 0 or target_codec != source_codec:
        raise ValueError(f"invalid remux output: {partial}")
    if abs(target_duration - source_duration) > 0.05:
        raise ValueError(
            f"duration changed while remuxing {source}: "
            f"{source_duration} -> {target_duration}"
        )
    partial.replace(target)
    return target.name, target.stat().st_size


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    train_path = args.dataset_root / "train" / "timeple_train.jsonl"
    rows = []
    mkv_names: set[str] = set()
    with train_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            videos = row.get("videos")
            if not isinstance(videos, list) or len(videos) != 1:
                raise ValueError(f"{train_path}:{line_number}: expected one video")
            name = Path(videos[0]).name
            if name.lower().endswith(".mkv"):
                mkv_names.add(name)
            rows.append(row)

    if not mkv_names:
        print("No MKV references remain in the annotations.")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[Path, Path]] = []
    for name in sorted(mkv_names):
        source = args.media_root / name
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty source media: {source}")
        target = args.output / f"{Path(name).stem}.mp4"
        primary_mp4 = args.media_root / target.name
        if primary_mp4.exists():
            raise ValueError(f"MP4 basename collision in primary media: {primary_mp4}")
        jobs.append((source, target))

    completed_count = 0
    output_bytes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(remux, source, target) for source, target in jobs]
        for future in concurrent.futures.as_completed(futures):
            _, size = future.result()
            output_bytes += size
            completed_count += 1
            if completed_count % 100 == 0 or completed_count == len(jobs):
                print(f"Remuxed {completed_count}/{len(jobs)} videos", flush=True)

    temporary = train_path.with_suffix(".jsonl.normalizing")
    if temporary.exists():
        raise FileExistsError(f"temporary annotation file exists: {temporary}")
    changed_rows = 0
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            name = Path(row["videos"][0]).name
            if name.lower().endswith(".mkv"):
                name = f"{Path(name).stem}.mp4"
                row["videos"] = [f"videos/{name}"]
                changed_rows += 1
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(train_path)
    print(
        f"Complete: {len(jobs)} unique videos, {changed_rows} annotation rows, "
        f"{output_bytes} output bytes",
        flush=True,
    )


if __name__ == "__main__":
    main()
