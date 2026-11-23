"""Structured, bounded, credential-safe logging for Gradio upstream calls."""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from pathlib import PurePath
from urllib.parse import urlsplit

logger = logging.getLogger("uvicorn.error")

_SECRET = re.compile(r"hf_[A-Za-z0-9_]+|Bearer\s+[^\s,]+", re.I)
_MAX_ITEMS = 12
_MAX_TEXT = 300


def _safe(obj, key="", depth=0):
    """Produce JSON-compatible log values, never binary blobs or credentials."""
    k = key.lower()
    if any(x in k for x in ("token", "authorization", "secret", "api_key", "password", "cookie")):
        return "[REDACTED]"
    if depth > 5:
        return "[max-depth]"
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return {"binary_bytes": len(obj)}
    if obj is None or isinstance(obj, (int, float, bool)):
        return obj
    if isinstance(obj, str):
        if k in ("prompt", "original_prompt", "negative_prompt", "text", "rewritten_prompt") and os.getenv("LOG_UPSTREAM_PROMPTS", "").lower() not in ("1", "true", "yes"):
            return {"text_chars": len(obj), "preview": "[REDACTED: set LOG_UPSTREAM_PROMPTS=true to log prompt text]"}
        if k in ("path", "name", "orig_name") or obj.startswith(("/tmp/", "/app/", "/home/")):
            return {"file": PurePath(obj).name}
        if obj.startswith(("https://", "http://")):
            parsed = urlsplit(obj)
            return {"url_host": parsed.hostname, "url_path": parsed.path[:120]}
        return _SECRET.sub("[REDACTED]", obj)[:_MAX_TEXT]
    if isinstance(obj, dict):
        return {str(k): _safe(v, str(k), depth + 1) for k, v in list(obj.items())[:_MAX_ITEMS]}
    if isinstance(obj, (list, tuple)):
        values = [_safe(v, key, depth + 1) for v in obj[:_MAX_ITEMS]]
        if len(obj) > _MAX_ITEMS:
            values.append({"more_items": len(obj) - _MAX_ITEMS})
        return values
    return f"<{type(obj).__name__}>"


def _dump(obj):
    return json.dumps(_safe(obj), ensure_ascii=False, default=str)


def predict_logged(client, *, space_id, api_name, kwargs, request_id=None):
    request_id = request_id or uuid.uuid4().hex[:12]
    start = time.monotonic()
    logger.info("[hf-upstream] request id=%s space=%s endpoint=%s params=%s",
                request_id, space_id, api_name, _dump(kwargs))
    try:
        response = client.predict(api_name=api_name, **kwargs)
    except Exception as exc:
        logger.error("[hf-upstream] failure id=%s endpoint=%s duration_ms=%d error_type=%s error=%s",
                     request_id, api_name, int((time.monotonic() - start) * 1000),
                     type(exc).__name__, _dump(str(exc)))
        raise
    logger.info("[hf-upstream] response id=%s endpoint=%s duration_ms=%d result=%s",
                request_id, api_name, int((time.monotonic() - start) * 1000),
                _dump(response))
    return response
