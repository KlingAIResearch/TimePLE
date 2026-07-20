#!/usr/bin/env python3
from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
import threading
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
TRAIN_DIR = CURRENT_DIR.parent / "train_building"
for import_path in (CURRENT_DIR, TRAIN_DIR, REPO_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from benchmark_utils import load_rows, load_yaml, normalize_intervals, save_rows, setup_logging
from step2_infer_models import GeminiKSInferencer, resolve_prompt_config


STATIC_DIR = CURRENT_DIR / "manual_review_web"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Human review for benchmark annotation correction.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", default="")
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def row_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("sample_id") or row.get("idx") or f"row_{index}")


class ReviewStore:
    def __init__(self, rows: list[dict[str, Any]], review_path: Path, export_path: Path, config: dict[str, Any]) -> None:
        self.rows = rows
        self.review_path = review_path
        self.export_path = export_path
        self.config = config
        self.ids = [row_id(row, index) for index, row in enumerate(rows)]
        self.index_by_id = {sample_id: index for index, sample_id in enumerate(self.ids)}
        self.allowed_videos = {str(Path(str(row.get("video_path"))).expanduser().resolve()) for row in rows if row.get("video_path")}
        self.lock = threading.Lock()
        self.reviews: dict[str, dict[str, Any]] = {}
        if review_path.exists():
            for record in load_rows(review_path):
                if record.get("sample_id"):
                    self.reviews[str(record["sample_id"])] = record
        self._inferencer: GeminiKSInferencer | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "total": len(self.rows),
            "reviewed": len(self.reviews),
            "samples": [
                {
                    "sample_id": sample_id,
                    "query": str(self.rows[index].get("query", "")),
                    "reviewed": sample_id in self.reviews,
                }
                for index, sample_id in enumerate(self.ids)
            ],
        }

    def sample(self, sample_id: str) -> dict[str, Any]:
        index = self.index_by_id[sample_id]
        row = self.rows[index]
        return {
            "sample_id": sample_id,
            "query": str(row.get("query", "")),
            "video_path": str(row.get("video_path", "")),
            "video_url": f"/api/video?path={quote(str(row.get('video_path', '')))}",
            "gt_timestamps": normalize_intervals(row.get("gt_timestamps") or row.get("time_gt")),
            "model_correction": row.get("model_correction"),
            "review": self.reviews.get(sample_id),
        }

    def inferencer(self) -> GeminiKSInferencer:
        if self._inferencer is None:
            self._inferencer = GeminiKSInferencer(dict(self.config.get("gemini", {})), resolve_prompt_config(self.config))
        return self._inferencer

    def assist(self, sample_id: str) -> dict[str, Any]:
        row = self.rows[self.index_by_id[sample_id]]
        result = self.inferencer().predict(str(row.get("video_path", "")), str(row.get("query", "")))
        correction = {
            "original_query": str(row.get("query", "")),
            "refined_query": result.get("refined_query", ""),
            "refined_segment": result.get("intervals", []),
            "reason": result.get("reason", ""),
            "raw_output": result.get("raw_text", ""),
            "error": result.get("error", ""),
        }
        row["model_correction"] = correction
        return correction

    def save_review(self, sample_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if sample_id not in self.index_by_id:
            raise KeyError(sample_id)
        record = {
            "sample_id": sample_id,
            "decision": str(payload.get("decision", "")),
            "final_query": str(payload.get("final_query", "")).strip(),
            "final_segment": normalize_intervals(payload.get("final_segment")),
            "notes": str(payload.get("notes", "")).strip(),
        }
        with self.lock:
            self.reviews[sample_id] = record
            save_rows(self.review_path, list(self.reviews.values()))
            exported: list[dict[str, Any]] = []
            for index, row in enumerate(self.rows):
                row_copy = dict(row)
                current_id = self.ids[index]
                if current_id in self.reviews:
                    row_copy["manual_review"] = self.reviews[current_id]
                exported.append(row_copy)
            save_rows(self.export_path, exported)
        return record


class Handler(BaseHTTPRequestHandler):
    store: ReviewStore

    def send_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path == "/api/bootstrap":
            self.send_json(self.store.summary())
            return
        if parsed.path == "/api/sample":
            sample_id = parse_qs(parsed.query).get("id", [""])[0]
            try:
                self.send_json(self.store.sample(sample_id))
            except KeyError:
                self.send_json({"error": "unknown sample"}, HTTPStatus.NOT_FOUND)
            return
        if parsed.path == "/api/video":
            raw_path = parse_qs(parsed.query).get("path", [""])[0]
            resolved = str(Path(raw_path).expanduser().resolve())
            if resolved not in self.store.allowed_videos or not Path(resolved).is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = Path(resolved).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(resolved)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        static_path = "index.html" if parsed.path in ("", "/") else parsed.path.lstrip("/")
        target = (STATIC_DIR / static_path).resolve()
        if STATIC_DIR.resolve() not in target.parents or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        parsed = urlsplit(self.path)
        payload = self.read_json()
        try:
            if parsed.path == "/api/assist":
                self.send_json(self.store.assist(str(payload.get("sample_id", ""))))
                return
            if parsed.path == "/api/review":
                self.send_json(self.store.save_review(str(payload.get("sample_id", "")), payload))
                return
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (KeyError, ValueError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    runtime = dict(config.get("runtime", {}))
    input_path = Path(args.input or runtime.get("input_jsonl", "")).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    review_path = Path(runtime.get("review_output_jsonl") or input_path.with_name(f"{input_path.stem}.reviews.jsonl")).expanduser().resolve()
    export_path = Path(runtime.get("review_export_jsonl") or input_path.with_name(f"{input_path.stem}.reviewed{input_path.suffix}")).expanduser().resolve()
    store = ReviewStore(load_rows(input_path), review_path, export_path, config)
    Handler.store = store
    host = args.host or str(runtime.get("review_host", "127.0.0.1"))
    port = args.port or int(runtime.get("review_port", 8765))
    print(f"Benchmark correction UI: http://{host}:{port}/")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
