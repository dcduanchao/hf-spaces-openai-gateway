"""Short-lived public image links, without exposing upstream Gradio file paths.

Files are ephemeral: Hugging Face Spaces may restart and discard local storage.
"""
from __future__ import annotations

import os
import re
import tempfile
import time
import uuid
from pathlib import Path

IMAGE_TTL_SECONDS = 30 * 60
IMAGE_DIR = Path(tempfile.gettempdir()) / "hf_spaces_openai_gateway_images"
ID_PATTERN = re.compile(r"[0-9a-f]{32}\.png")


def cleanup_expired(now: float | None = None) -> None:
    now = time.time() if now is None else now
    if not IMAGE_DIR.is_dir():
        return
    for path in IMAGE_DIR.glob("*.png"):
        if not ID_PATTERN.fullmatch(path.name):
            continue
        try:
            if now - path.stat().st_mtime >= IMAGE_TTL_SECONDS:
                path.unlink(missing_ok=True)
        except OSError:
            pass


def save_temp_image(data: bytes) -> str:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_expired()
    name = uuid.uuid4().hex + ".png"
    path = IMAGE_DIR / name
    with path.open("xb") as out:
        out.write(data)
    return name


def get_temp_image(name: str) -> Path | None:
    if not ID_PATTERN.fullmatch(name):
        return None
    path = IMAGE_DIR / name
    try:
        if not path.is_file():
            return None
        if time.time() - path.stat().st_mtime >= IMAGE_TTL_SECONDS:
            path.unlink(missing_ok=True)
            return None
        return path
    except OSError:
        return None


def public_image_url(name: str) -> str:
    # Fixed, controlled origin avoids Host header injection into JSON image URLs.
    base = os.getenv(
        "GATEWAY_PUBLIC_BASE_URL",
        "https://duan0320-hf-spaces-openai-gateway.hf.space",
    ).rstrip("/")
    if not base.startswith("https://") and not base.startswith("http://localhost:"):
        raise ValueError("GATEWAY_PUBLIC_BASE_URL must use https (or localhost for development)")
    return base + "/v1/files/" + name
