from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Configuration root must be a mapping")
    return payload


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if file_path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("data", payload.get("samples", []))
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a list of JSON objects: {file_path}")
    return rows


def save_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.suffix.lower() == ".jsonl":
        file_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    else:
        file_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(asctime)s %(levelname)s %(message)s")


def normalize_intervals(raw: Any) -> list[list[float]]:
    from pipeline_core import normalize_intervals as normalize

    return normalize(raw)
