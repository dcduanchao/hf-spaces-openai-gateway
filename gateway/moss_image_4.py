"""Adapter for hugging-apps/moss-image-4step-lora's Gradio /generate endpoint.

Contract: (prompt: str, resolution: one of 12 exact dropdown strings, seed: int)
          -> (image_filepath, markdown_text).
"""
from __future__ import annotations

import secrets

from gradio_client import Client

from .adapter import parse_size, result_bytes, select_result
from .upstream_auth import client_kwargs
from .upstream_logging import predict_logged


RESOLUTIONS = (
    "1024x1024", "1280x1280", "1440x1440",
    "768x512", "1024x768", "1280x960",
    "1440x1280", "1824x1024",
    "512x768", "768x1024", "960x1280", "1024x1824",
)
MAX_SEED = 2**32 - 1


def build_kwargs(profile, payload, image_paths=None):
    if image_paths:
        raise ValueError("MOSS Image 4-Step 仅支持文生图，不支持图片编辑")

    size = payload.get("size") or profile.default_size
    width, height = parse_size(size, profile)
    normalized_size = f"{width}x{height}"
    if normalized_size not in RESOLUTIONS:
        raise ValueError("moss-image-4 只支持 12 个固定的 resolution 尺寸")

    extras = payload.get("extra_body") or {}
    if not isinstance(extras, dict):
        raise ValueError("extra_body 必须是 JSON 对象")
    if extras:
        raise ValueError("moss-image-4 不支持额外参数")

    reserved = {
        "model", "prompt", "size", "n", "response_format", "seed",
        "extra_body", "user", "quality", "background", "output_format",
        "moderation", "style",
    }
    unexpected = set(payload) - reserved
    if unexpected:
        raise ValueError(f"moss-image-4 不支持额外参数：{sorted(unexpected)}")

    user_seed = payload.get("seed", -1)
    if isinstance(user_seed, bool) or not isinstance(user_seed, int) or not -1 <= user_seed <= MAX_SEED:
        raise ValueError("seed 应为 -1（随机）或 0～4294967295")
    seed = secrets.randbelow(MAX_SEED + 1) if user_seed == -1 else user_seed
    return {
        "prompt": payload["prompt"],
        "resolution": f"{width} × {height}",
        "seed": seed,
    }


def run(profile, payload, token, image_paths=None, ip_token=None):
    kwargs = build_kwargs(profile, payload, image_paths)
    client = Client(profile.space_id, **client_kwargs(token, ip_token))
    try:
        result = predict_logged(
            client, space_id=profile.space_id, api_name=profile.api_name, kwargs=kwargs,
        )
        if not isinstance(result, (list, tuple)) or len(result) != 2:
            raise ValueError("MOSS Image 4-Step 返回结果应为 (image, markdown_text)")
        return result_bytes(select_result(result[0], []), profile)
    finally:
        client.close()
