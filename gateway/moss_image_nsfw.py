"""Adapter for jmendelblatt/moss-image-nsfw-lora's Gradio /generate API.

Upstream contract: prompt, negative_prompt, width, height, steps,
guidance_scale, lora_scale, realistic_scale, add_trigger, seed.
The result is a one-item list containing an image filepath.
"""
from __future__ import annotations

import math
import secrets

from gradio_client import Client

from .adapter import parse_size, result_bytes, select_result
from .models import validate_extra
from .upstream_auth import client_kwargs
from .upstream_logging import predict_logged


MAX_SEED = 2**32 - 1
EXTRA_NAMES = (
    "negative_prompt", "steps", "guidance_scale",
    "lora_scale", "realistic_scale", "add_trigger",
)


def build_kwargs(profile, payload, image_paths=None):
    if image_paths:
        raise ValueError("MOSS Image NSFW LoRA 仅支持文生图，不支持图片编辑")

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

    unknown = set(extras) - set(EXTRA_NAMES)
    if unknown:
        raise ValueError(f"未知模型参数：{sorted(unknown)}")

    values = {}
    for key in EXTRA_NAMES:
        spec = profile.extras[key]
        try:
            value = validate_extra(extras.get(key, spec.default), spec)
        except ValueError as exc:
            raise ValueError(f"{key}: {exc}") from exc
        if value is None:
            raise ValueError(f"{key}: cannot be null")
        if key in ("steps", "guidance_scale", "lora_scale", "realistic_scale"):
            if not math.isfinite(float(value)):
                raise ValueError(f"{key}: must be a finite number")
            value = float(value)
        values[key] = value

    user_seed = payload.get("seed", -1)
    if isinstance(user_seed, bool) or not isinstance(user_seed, int) or not -1 <= user_seed <= MAX_SEED:
        raise ValueError("seed 应为 -1（随机）或 0～4294967295")
    seed = secrets.randbelow(MAX_SEED + 1) if user_seed == -1 else user_seed

    return {
        "prompt": payload["prompt"],
        "negative_prompt": values["negative_prompt"],
        "width": float(width),
        "height": float(height),
        "steps": values["steps"],
        "guidance_scale": values["guidance_scale"],
        "lora_scale": values["lora_scale"],
        "realistic_scale": values["realistic_scale"],
        "add_trigger": values["add_trigger"],
        "seed": seed,
    }


def run(profile, payload, token, image_paths=None, ip_token=None):
    kwargs = build_kwargs(profile, payload, image_paths)
    client = Client(profile.space_id, **client_kwargs(token, ip_token))
    try:
        result = predict_logged(
            client, space_id=profile.space_id, api_name=profile.api_name, kwargs=kwargs,
        )
        # A single-output Gradio endpoint is unwrapped by gradio_client:
        # predict() can return the image filepath directly, even though the
        # HTTP API documentation describes a one-element result array.
        if isinstance(result, (list, tuple)):
            if len(result) != 1:
                raise ValueError("MOSS Image NSFW LoRA 返回结果应为单张图片")
            result = result[0]
        image = select_result(result, [])
        if not isinstance(image, (str, dict)):
            raise ValueError("MOSS Image NSFW LoRA 未返回有效的图片路径")
        return result_bytes(image, profile)
    finally:
        client.close()
