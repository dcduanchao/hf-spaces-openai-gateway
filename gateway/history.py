"""Persistent per-request history in a PRIVATE Hugging Face bucket mounted at /data.

Requires a read-write Space volume: hf://buckets/duan0320/image-history:/data
No database locking or SQLite on object storage. Each request owns independent JSON.
"""
from __future__ import annotations
import json
import os
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException

ROOT = Path(os.getenv("HISTORY_BUCKET_PATH", "/data"))
ID_RE = re.compile(r"[a-f0-9]{32}")
MAX_ITEMS = 100
MAX_PROMPT = 4000


def _ready() -> Path:
    root = ROOT
    # A marker is explicitly created by the administrator after the bucket is
    # mounted. Never silently write history into an ephemeral /data directory.
    if not (root / ".history_bucket_ready").is_file():
        raise RuntimeError("History Bucket is not configured: mount duan0320/image-history read-write at /data and create .history_bucket_ready")
    if not os.access(root, os.W_OK):
        raise RuntimeError("History Bucket mount is not writable")
    return root / "gateway-history"


def _folder(record_id: str) -> Path:
    if not ID_RE.fullmatch(record_id):
        raise ValueError("Invalid history ID")
    return _ready() / record_id[:2] / record_id


def new_record(model: str, mode: str, prompt: str, params: dict) -> str:
    record_id = secrets.token_hex(16)
    directory = _folder(record_id)
    directory.mkdir(parents=True, exist_ok=False)
    record = {
        "id": record_id, "model": model, "mode": mode,
        "prompt": str(prompt)[:MAX_PROMPT],
        "parameters": {k: v for k, v in params.items() if k not in (
            "hf_token", "token", "authorization", "x_hf_token", "api_key", "image", "images")},
        "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
        "duration_ms": None, "error_code": None, "images": [],
        "input_images": [],
    }
    _write_json(directory / "record.json", record)
    return record_id


def _write_json(path: Path, value: dict) -> None:
    temp = path.with_name("record-" + secrets.token_hex(8) + ".tmp")
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _load(record_id: str) -> dict:
    path = _folder(record_id) / "record.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise KeyError(record_id) from exc


def store_image(record_id: str, image: bytes, *, input_image: bool = False) -> str:
    record = _load(record_id)
    key = "input_images" if input_image else "images"
    index = len(record[key])
    filename = f"{key}-{index}.png"
    directory = _folder(record_id)
    (directory / filename).write_bytes(image)
    record[key].append(filename)
    _write_json(directory / "record.json", record)
    return filename


def finish_record(record_id: str, status: str, duration_ms: int, error_code: str | None = None) -> None:
    record = _load(record_id)
    record.update(status=status, duration_ms=duration_ms, error_code=error_code)
    _write_json(_folder(record_id) / "record.json", record)


def list_records(*, model: str | None = None, status: str | None = None,
                 limit: int = 30, offset: int = 0) -> dict:
    limit = max(1, min(limit, MAX_ITEMS))
    root = _ready()
    rows = []
    for path in root.glob("*/*/record.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if model and data.get("model") != model:
            continue
        if status and data.get("status") != status:
            continue
        rows.append({k: data.get(k) for k in (
            "id", "model", "mode", "prompt", "status", "created_at",
            "duration_ms", "error_code", "images", "input_images")})
    rows.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return {"total": len(rows), "items": rows[offset:offset + limit]}


def get_record(record_id: str) -> dict:
    return _load(record_id)


def get_image(record_id: str, filename: str) -> Path:
    rec = _load(record_id)
    if filename not in rec.get("images", []) and filename not in rec.get("input_images", []):
        raise KeyError(filename)
    return _folder(record_id) / filename


def delete_record(record_id: str) -> None:
    from shutil import rmtree
    directory = _folder(record_id)
    if not (directory / "record.json").is_file():
        raise KeyError(record_id)
    rmtree(directory)
