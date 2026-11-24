"""Adapter for linoyts/qwen-image-2-1-realtime's 2-step /generate API.

The upstream generate endpoint accepts prompt, gallery, seed, resolution
(integer choices 512/1024/1536), aspect_ratio (Auto or seven fixed ratios).
No prepare or realtime session is required for a single image request.
"""
from __future__ import annotations

import math
import secrets
from gradio_client import Client, handle_file
from .adapter import parse_size, result_bytes, select_result
from .upstream_auth import client_kwargs
from .upstream_logging import predict_logged

BASE_SIZES = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}
RESOLUTIONS = (512, 1024, 1536)
MAX_REFERENCES = 4
MAX_SEED = 2**31 - 1


def preset_sizes():
    """Represent all 21 upstream size-band/ratio combinations as API size hints.

    These are display/request hints derived from the supplied upstream
    ASPECT_RATIOS reference sizes, not guaranteed pixel-exact outputs.
    """
    return {
        f"{round(w * res / 2048)}x{round(h * res / 2048)}": (res, ratio)
        for res in RESOLUTIONS
        for ratio, (w, h) in BASE_SIZES.items()
    }


def choose_parameters(width: int, height: int) -> tuple[int, str]:
    known = preset_sizes().get(f"{width}x{height}")
    if known:
        return known
    # Preserve generic API custom size support. Compare aspect ratios first,
    # then select the closest area band for the selected aspect ratio.
    target_ratio = width / height
    ratio = min(BASE_SIZES, key=lambda r: abs(math.log(target_ratio / (BASE_SIZES[r][0] / BASE_SIZES[r][1]))))
    base_w, base_h = BASE_SIZES[ratio]
    target_area = width * height
    resolution = min(RESOLUTIONS, key=lambda res: abs(math.log(target_area / (base_w * base_h * (res / 2048)**2))))
    return resolution, ratio


def build_kwargs(profile, payload, image_paths=None):
    requested_size = payload.get("size") or profile.default_size
    if requested_size == "auto":
        resolution, aspect_ratio = 1536, "Auto"
    else:
        width, height = parse_size(requested_size, profile)
        if requested_size in {f"{w}x{h}" for w, h in BASE_SIZES.values()}:
            ratio = next(k for k, (w, h) in BASE_SIZES.items()
                         if requested_size == f"{w}x{h}")
            resolution, aspect_ratio = 1536, ratio
        else:
            resolution, aspect_ratio = choose_parameters(width, height)
    paths = image_paths or []
    if len(paths) > MAX_REFERENCES:
        raise ValueError("Qwen Image 2.1 2-step 网关最多支持 4 张参考图")
    extras = payload.get("extra_body") or {}
    if not isinstance(extras, dict):
        raise ValueError("extra_body 必须为 JSON 对象")
    reserved = {
        "model", "prompt", "size", "n", "response_format", "seed",
        "extra_body", "user", "quality", "background", "output_format",
        "moderation", "style",
    }
    extra = dict(extras)
    for key, value in payload.items():
        if key not in reserved:
            if key in extra:
                raise ValueError(f"重复扩展参数：{key}")
            extra[key] = value
    if extra:
        raise ValueError(f"此模型不支持扩展参数：{sorted(extra)}")

    raw_seed = int(payload.get("seed", -1))
    if not (-1 <= raw_seed <= MAX_SEED):
        raise ValueError("seed 只能是 -1 或 0～2147483647")
    seed = secrets.randbelow(MAX_SEED + 1) if raw_seed == -1 else raw_seed

    return {
        "prompt": payload["prompt"],
        "input_images": [{"image": handle_file(path), "caption": None} for path in paths],
        "seed": seed,
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
    }


def run(profile, payload, token, image_paths=None, ip_token=None):
    kwargs = build_kwargs(profile, payload, image_paths)
    client = Client(profile.space_id, **client_kwargs(token, ip_token))
    try:
        result = predict_logged(
            client, space_id=profile.space_id, api_name=profile.api_name, kwargs=kwargs
        )
        return result_bytes(select_result(result, profile.result_path), profile)
    finally:
        client.close()
