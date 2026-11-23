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
    # Do not write into an ephemeral /data directory if the volume was not mounted.
    # Space bucket volumes are filesystem mounts. A marker may additionally be
    # used for explicitly configured nested/custom mounts.
    if not os.path.ismount(root) and not (root / ".history_bucket_ready").is_file():
        raise RuntimeError("History Bucket not mounted: attach duan0320/image-history read-write at /data")
    if not os.access(root, os.W_OK):
        raise RuntimeError("History Bucket mount is not writable")
    return root / "gateway-history"


def _folder(record_id: str) -> Path:
    if not ID_RE.fullmatch(record_id):
        raise ValueError("Invalid history ID")
    return _ready() / record_id[:2] / record_id


_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9_]+|Bearer\\s+[^\\s,;]+", re.I)


def _safe_value(value: Any, key: str = "", depth: int = 0):
    if any(x in key.lower() for x in ("token", "authorization", "secret", "password", "api_key", "cookie")):
        return "[REDACTED]"
    if depth > 5:
        return "[max-depth]"
    if isinstance(value, dict):
        return {str(k)[:80]: _safe_value(v, str(k), depth + 1) for k, v in list(value.items())[:40]}
    if isinstance(value, (list, tuple)):
        return [_safe_value(v, key, depth + 1) for v in value[:40]]
    if isinstance(value, str):
        return _TOKEN_RE.sub("[REDACTED]", value)[:MAX_PROMPT]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return "<unsupported>"


def new_record(model: str, mode: str, prompt: str, params: dict) -> str:
    record_id = secrets.token_hex(16)
    directory = _folder(record_id)
    directory.mkdir(parents=True, exist_ok=False)
    record = {
        "id": record_id, "model": model, "mode": mode,
        "prompt": _safe_value(str(prompt)[:MAX_PROMPT]),
        "parameters": _safe_value(params),
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


def delete_all_records() -> dict:
    """Permanently remove every gateway history record and its image files.

    Only touches gateway-history/<prefix>/<id> directories containing
    record.json; does not remove unrelated objects in the bucket.
    """
    from shutil import rmtree
    root = _ready()
    deleted = 0
    failed = 0
    for record_file in root.glob("*/*/record.json"):
        record_dir = record_file.parent
        if not ID_RE.fullmatch(record_dir.name):
            continue
        try:
            rmtree(record_dir)
            deleted += 1
        except OSError:
            failed += 1
    return {"deleted": deleted, "failed": failed}
