"""Adapter for mrfakename/Z-Image-Turbo.

Published Gradio API:
  /generate_image(prompt, height, width, num_inference_steps, seed,
                  randomize_seed) -> (image, used_seed)
"""
from __future__ import annotations

import secrets

from gradio_client import Client

from .adapter import parse_size, result_bytes, select_result
from .models import validate_extra
from .upstream_logging import predict_logged
from .upstream_auth import client_kwargs


def build_kwargs(profile, payload, image_paths=None):
    if image_paths:
        raise ValueError("Z-Image-Turbo 仅支持文生图，不支持上传图片编辑")
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

    steps_spec = profile.extras["steps"]
    try:
        steps = validate_extra(extras.get("steps", steps_spec.default), steps_spec)
    except ValueError as exc:
        raise ValueError(f"steps: {exc}") from exc

    user_seed = int(payload.get("seed", -1))
    if user_seed < -1 or user_seed > 2**32 - 1:
        raise ValueError("seed 应为 -1（随机）或 0～4294967295")
    seed = secrets.randbelow(2**32) if user_seed == -1 else user_seed
    return {
        "prompt": payload["prompt"],
        "height": height,
        "width": width,
        "num_inference_steps": steps,
        "seed": seed,
        "randomize_seed": False,
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
            raise ValueError("Z-Image-Turbo 返回结果应为 (image, used_seed)")
        image = select_result(result[0], [])
        return result_bytes(image, profile)
    finally:
        client.close()
