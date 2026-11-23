from __future__ import annotations
import base64
import io
import ipaddress
import os
from pathlib import Path
from urllib.parse import urlsplit
import httpx
from gradio_client import Client, handle_file
from PIL import Image
from .models import Profile, validate_extra
from .upstream_logging import predict_logged
from .upstream_auth import client_kwargs


def parse_size(size: str, profile: Profile) -> tuple[int, int]:
    """Presets are suggestions, not an exhaustive allowlist.

    Custom width/height values are accepted within safe bounds. Upstream
    Spaces may impose tighter limits and can reject custom dimensions.
    """
    import re
    if not isinstance(size, str) or not re.fullmatch(r"[1-9][0-9]{1,4}[xX][1-9][0-9]{1,4}", size.strip()):
        raise ValueError("size must be WIDTHxHEIGHT (e.g. 1024x1536)")
    width, height = map(int, re.split("[xX]", size.strip()))
    if not (256 <= width <= 4096 and 256 <= height <= 4096):
        raise ValueError("size width and height must be between 256 and 4096")
    if width * height > 16_777_216:
        raise ValueError("size exceeds 16777216 pixels")
    return width, height


def make_kwargs(profile: Profile, payload: dict, image_paths: list[str] | None = None):
    width, height = parse_size(payload.get("size") or profile.default_size, profile)
    reserved = {"model", "prompt", "size", "n", "response_format", "seed", "extra_body", "user", "quality", "background", "output_format", "moderation", "style"}
    extras = dict(payload.get("extra_body") or {})
    if not isinstance(extras, dict):
        raise ValueError("extra_body must be an object")
    for key, val in payload.items():
        if key not in reserved:
            if key in extras:
                raise ValueError(f"duplicate extra field: {key}")
            extras[key] = val
    unknown = set(extras) - set(profile.extras)
    if unknown:
        raise ValueError(f"unsupported fields: {sorted(unknown)}")
    std = {"prompt": payload["prompt"], "width": width, "height": height, "seed": payload.get("seed", -1)}
    kwargs = dict(profile.fixed)
    for standard, target in profile.bindings.items():
        if standard == "images":
            if image_paths:
                kwargs[target] = [handle_file(p) for p in image_paths]
        elif standard == "image":
            if image_paths:
                kwargs[target] = handle_file(image_paths[0])
        elif standard in std:
            kwargs[target] = std[standard]
        else:
            raise ValueError(f"unknown binding: {standard}")
    for name, spec in profile.extras.items():
        raw = extras.get(name, spec.default)
        try:
            value = validate_extra(raw, spec)
        except ValueError as e:
            raise ValueError(f"{name}: {e}") from e
        if value is not None:
            kwargs[spec.target] = value
    return kwargs


def select_result(value, path):
    for key in path:
        value = value[key]
    # Gallery results often wrap one image inside a list/tuple.
    for _ in range(4):
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        else:
            break
    if isinstance(value, dict):
        if "path" in value or "url" in value:
            return value
        if "image" in value:
            return select_result(value["image"], [])
    return value


def _trusted_upstream_url(url: str, profile: Profile) -> bool:
    parsed = urlsplit(url)
    expected = profile.space_id.lower().replace("/", "-") + ".hf.space"
    return parsed.scheme == "https" and parsed.hostname == expected and not parsed.username and not parsed.password and parsed.port in (None, 443)


def result_bytes(value, profile: Profile) -> bytes:
    if isinstance(value, dict):
        path, url = value.get("path"), value.get("url")
    else:
        path, url = value if isinstance(value, str) else None, None
    if path and Path(path).is_file():
        raw = Path(path).read_bytes()
    else:
        url = url or (path if isinstance(path, str) and path.startswith("https://") else None)
        if not url or not _trusted_upstream_url(url, profile):
            raise ValueError("upstream did not return a trusted image")
        with httpx.Client(timeout=60, follow_redirects=False) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                chunks = []
                size = 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > 20 * 1024 * 1024:
                        raise ValueError("upstream image too large")
                    chunks.append(chunk)
                raw = b"".join(chunks)
    with Image.open(io.BytesIO(raw)) as im:
        if im.width * im.height > 16_777_216:
            raise ValueError("upstream image dimensions too large")
        output = io.BytesIO()
        im.convert("RGB").save(output, "PNG")
        return output.getvalue()


def invoke(profile: Profile, payload: dict, token: str | None, image_paths=None, ip_token=None) -> bytes:
    if profile.adapter == "qwen_image":
        from .qwen_image import run
        return run(profile, payload, token, image_paths, ip_token=ip_token)
    if profile.adapter == "z_image":
        from .z_image import run
        return run(profile, payload, token, image_paths)
    kwargs = make_kwargs(profile, payload, image_paths)
    client = Client(profile.space_id, **client_kwargs(token, ip_token))
    try:
        result = predict_logged(client, space_id=profile.space_id, api_name=profile.api_name, kwargs=kwargs)
        return result_bytes(select_result(result, profile.result_path), profile)
    finally:
        client.close()
