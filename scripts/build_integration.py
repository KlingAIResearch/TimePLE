#!/usr/bin/env python3
"""Build a TimePLE integration on top of an exact upstream installation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INTEGRATIONS = {
    "transformers": ROOT / "integrations/transformers",
    "ms-swift": ROOT / "integrations/ms_swift",
    "easyr1": ROOT / "integrations/easyr1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(integration: str) -> tuple[Path, dict[str, Any]]:
    integration_dir = INTEGRATIONS[integration]
    manifest = json.loads((integration_dir / "manifest.json").read_text(encoding="utf-8"))
    return integration_dir, manifest


def installed_root(manifest: dict[str, Any]) -> tuple[importlib.metadata.Distribution, Path]:
    distribution_name = str(manifest["distribution"])
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise SystemExit(
            f"Package {distribution_name!r} is not installed; run the matching `scripts/setup_env.sh` profile first."
        ) from exc

    installed_version = distribution.version
    expected_version = str(manifest["version"])
    if installed_version != expected_version:
        raise SystemExit(
            f"{distribution_name} {installed_version} is installed, but this integration requires exactly {expected_version}."
        )

    expected_commit = manifest.get("commit")
    if expected_commit:
        direct_url_text = distribution.read_text("direct_url.json")
        direct_url = json.loads(direct_url_text) if direct_url_text else {}
        installed_commit = direct_url.get("vcs_info", {}).get("commit_id")
        if installed_commit != expected_commit:
            raise SystemExit(
                f"{distribution_name} must come from commit {expected_commit}; installed metadata reports "
                f"{installed_commit or 'no VCS commit'}."
            )

    return distribution, Path(distribution.locate_file("")).resolve()


def patch_state(site_packages: Path, modified_files: dict[str, Any]) -> str:
    states: set[str] = set()
    problems: list[str] = []
    for relative, expected in modified_files.items():
        target = site_packages / relative
        if not target.is_file():
            problems.append(f"{relative}: missing")
            continue
        actual = sha256(target)
        if actual == expected["upstream"]:
            states.add("upstream")
        elif actual == expected["patched"]:
            states.add("patched")
        else:
            problems.append(f"{relative}: unexpected sha256 {actual}")

    if problems:
        raise SystemExit("Integration source validation failed:\n  " + "\n  ".join(problems))
    if len(states) != 1:
        raise SystemExit("Integration is only partially applied; recreate the environment before retrying.")
    return states.pop()


def created_file_state(site_packages: Path, created_files: dict[str, str]) -> str:
    states: set[str] = set()
    problems: list[str] = []
    for relative, expected_hash in created_files.items():
        target = site_packages / relative
        if not target.exists():
            states.add("upstream")
        elif target.is_file() and sha256(target) == expected_hash:
            states.add("patched")
        else:
            problems.append(f"{relative}: existing file differs from the TimePLE integration")
    if problems:
        raise SystemExit("Integration file validation failed:\n  " + "\n  ".join(problems))
    if not states:
        return "patched"
    if len(states) != 1:
        raise SystemExit("Integration-created files are only partially installed; recreate the environment before retrying.")
    return states.pop()


def apply_integration(integration: str, *, dry_run: bool, check: bool) -> None:
    integration_dir, manifest = load_manifest(integration)
    _, site_packages = installed_root(manifest)
    modified_files = dict(manifest.get("modified_files", {}))
    modified_state = patch_state(site_packages, modified_files)
    created_state = created_file_state(site_packages, dict(manifest.get("created_files", {})))
    if modified_state != created_state:
        raise SystemExit("Integration is only partially applied; recreate the environment before retrying.")
    state = modified_state

    if check:
        if state != "patched":
            raise SystemExit(f"{integration} integration is not fully installed.")
        print(f"{integration}: installed and verified")
        return

    print(f"{integration}: upstream={manifest['version']} state={state}")
    if state == "upstream":
        patch_tool = shutil.which("patch")
        if patch_tool is None:
            raise SystemExit("The POSIX `patch` command is required to build integrations.")
        patch_file = integration_dir / str(manifest["patch"])
        command = [
            patch_tool,
            "--batch",
            "--forward",
            "--strip=1",
            "--directory",
            str(site_packages),
            "--input",
            str(patch_file),
        ]
        print("apply", patch_file.relative_to(ROOT))
        if not dry_run:
            subprocess.run(command, check=True)

    if not dry_run:
        if patch_state(site_packages, modified_files) != "patched":
            raise SystemExit("Patch application did not produce the expected files.")
        if created_file_state(site_packages, dict(manifest.get("created_files", {}))) != "patched":
            raise SystemExit("Patch application did not create the expected integration files.")
        print(f"{integration}: installed and verified")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("integration", choices=tuple(INTEGRATIONS))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    apply_integration(args.integration, dry_run=args.dry_run, check=args.check)


if __name__ == "__main__":
    main()
