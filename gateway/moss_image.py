"""Adapter for hugging-apps/moss-image-8-step-dmd.

Published Gradio /generate signature:
(prompt: str, height: float, width: float, num_steps: float,
 seed: int, randomize_seed: bool, lora_strength: float)
 -> (image_filepath, used_seed)
"""
from __future__ import annotations

import math

from gradio_client import Client

from .adapter import parse_size, result_bytes, select_result
from .models import validate_extra
from .upstream_auth import client_kwargs
from .upstream_logging import predict_logged


def build_kwargs(profile, payload, image_paths=None):
    if image_paths:
        raise ValueError("MOSS Image 8-Step 仅支持文生图，不支持图片编辑")

    width, height = parse_size(payload.get("size") or profile.default_size, profile)
    extras = payload.get("extra_body") or {}
    if not isinstance(extras, dict):
        raise ValueError("extra_body 必须是 JSON 对象")
    extras = dict(extras)
    reserved = {
        "model", "prompt", "size", "n", "response_format", "seed",
        "extra_body", "user", "quality", "background", "output_format",
        "moderation", "style",
    }
    for key, value in payload.items():
        if key not in reserved:
            if key in extras:
                raise ValueError(f"重复的扩展参数：{key}")
            extras[key] = value
    unknown = set(extras) - set(profile.extras)
    if unknown:
        raise ValueError(f"未知模型参数：{sorted(unknown)}")

    values = {}
    for key in ("steps", "lora_strength"):
        spec = profile.extras[key]
        try:
            values[key] = validate_extra(extras.get(key, spec.default), spec)
        except ValueError as exc:
            raise ValueError(f"{key}: {exc}") from exc
        if values[key] is None or not math.isfinite(float(values[key])):
            raise ValueError(f"{key}: must be a finite number")

    steps = float(values["steps"])
    lora_strength = float(values["lora_strength"])
    if steps <= 0:
        raise ValueError("steps 必须为正数")
    if not 0 <= lora_strength <= 1:
        raise ValueError("lora_strength 应在 0～1 之间")

    raw_seed = payload.get("seed", -1)
    if isinstance(raw_seed, bool) or not isinstance(raw_seed, int) or not -1 <= raw_seed <= 2**32 - 1:
        raise ValueError("seed 应为 -1（随机）或 0～4294967295")

    return {
        "prompt": payload["prompt"],
        "height": float(height),
        "width": float(width),
        "num_steps": steps,
        "seed": 0 if raw_seed == -1 else raw_seed,
        "randomize_seed": raw_seed == -1,
        "lora_strength": lora_strength,
    }


def run(profile, payload, token, image_paths=None, ip_token=None):
    kwargs = build_kwargs(profile, payload, image_paths)
    client = Client(profile.space_id, **client_kwargs(token, ip_token))
    try:
        result = predict_logged(
            client, space_id=profile.space_id,
            api_name=profile.api_name, kwargs=kwargs,
        )
        if not isinstance(result, (list, tuple)) or len(result) != 2:
            raise ValueError("MOSS Image 8-Step 返回结果应为 (image, used_seed)")
        image = select_result(result[0], [])
        return result_bytes(image, profile)
    finally:
        client.close()
