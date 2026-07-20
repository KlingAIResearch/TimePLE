#!/usr/bin/env python3
"""Install TimePLE overlays into public packages in the active environment."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def package_root(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise SystemExit(f"Package {name!r} is not installed; run the matching `uv sync --extra ...` first.")
    return Path(spec.origin).resolve().parent


def copy_tree(source: Path, destination: Path, dry_run: bool) -> None:
    for item in sorted(source.rglob("*.py")):
        target = destination / item.relative_to(source)
        print(f"{item.relative_to(ROOT)} -> {target}")
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("integration", choices=("transformers", "ms-swift", "easyr1"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.integration == "easyr1":
        copy_tree(ROOT / "integrations/easyr1/overlay/verl", package_root("verl"), args.dry_run)
    elif args.integration == "transformers":
        copy_tree(
            ROOT / "integrations/transformers/overlay/models/qwen3_vl",
            package_root("transformers") / "models/qwen3_vl",
            args.dry_run,
        )
    else:
        copy_tree(ROOT / "integrations/ms_swift/overlay/swift", package_root("swift"), args.dry_run)


if __name__ == "__main__":
    main()
