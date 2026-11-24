"""MiniMax H3 Image MysticXXX V5 Gradio adapter.

Published /generate contract (12 fields): prompt, images (Gallery list),
aspect, resolution, seed, randomize, enhance, reasoning_effort, and four
LoRA scale controls. Gallery supports up to 8 image references.
"""
from __future__ import annotations

import math

from gradio_client import Client, handle_file

from .adapter import parse_size, result_bytes, select_result
from .models import validate_extra
from .upstream_auth import client_kwargs
from .upstream_logging import predict_logged

ASPECTS = ("auto", "1:1", "5:4", "4:3", "3:2", "16:9", "21:9",
           "4:5", "3:4", "2:3", "9:16")
MAX_REFERENCES = 8
MAX_SEED = 2**31 - 1
FIELDS = ("aspect", "resolution", "enhance", "reasoning_effort",
          "lora_scale", "lora_attn_scale", "lora_mlp_scale", "lora_early_scale")
RESERVED = {"model", "prompt", "size", "n", "response_format", "seed",
            "extra_body", "user", "quality", "background", "output_format",
            "moderation", "style"}


def build_kwargs(profile, payload, image_paths=None):
    paths = image_paths or []
    if len(paths) > MAX_REFERENCES:
        raise ValueError("MiniMax H3 最多允许 8 张编辑参考图")
    raw_size = payload.get("size") or profile.default_size
    if raw_size == "auto":
        inferred_aspect = "auto"
    else:
        width, height = parse_size(raw_size, profile)
        ratio = width / height
        fixed = ASPECTS[1:]
        inferred_aspect = min(
            fixed,
            key=lambda name: abs(math.log(
                ratio / (int(name.split(":")[0]) / int(name.split(":")[1]))
            )),
        )

    raw_extras = payload.get("extra_body") or {}
    if not isinstance(raw_extras, dict):
        raise ValueError("extra_body 必须是 JSON 对象")
    extras = dict(raw_extras)
    for key, value in payload.items():
        if key not in RESERVED:
            if key in extras:
                raise ValueError(f"重复扩展参数：{key}")
            extras[key] = value
    unknown = set(extras) - set(FIELDS)
    if unknown:
        raise ValueError(f"未知 MiniMax H3 参数：{sorted(unknown)}")

    values = {}
    for key in FIELDS:
        spec = profile.extras[key]
        try:
            value = validate_extra(extras.get(key, spec.default), spec)
        except ValueError as exc:
            raise ValueError(f"{key}: {exc}") from exc
        if value is None:
            raise ValueError(f"{key}: 不能是 null")
        if key.startswith("lora_"):
            if not math.isfinite(float(value)):
                raise ValueError(f"{key}: 必须是有限数值")
            value = float(value)
        values[key] = value

    aspect = values["aspect"]
    if aspect == "auto":
        aspect = inferred_aspect
    if aspect not in ASPECTS:
        raise ValueError("aspect 必须是上游支持的 11 个比例之一")

    user_seed = payload.get("seed", -1)
    if isinstance(user_seed, bool) or not isinstance(user_seed, int) or not -1 <= user_seed <= MAX_SEED:
        raise ValueError("seed 必须是 -1 或 0～2147483647")
    # Leave random seed generation to the upstream checkbox.
    randomize = user_seed == -1

    return {
        "prompt": payload["prompt"],
        "images": [{"image": handle_file(path), "caption": None} for path in paths],
        "aspect": aspect,
        "resolution": values["resolution"],
        "seed": float(0 if randomize else user_seed),
        "randomize": randomize,
        "enhance": values["enhance"],
        "reasoning_effort": values["reasoning_effort"],
        "lora_scale": values["lora_scale"],
        "lora_attn_scale": values["lora_attn_scale"],
        "lora_mlp_scale": values["lora_mlp_scale"],
        "lora_early_scale": values["lora_early_scale"],
    }


def run(profile, payload, token, image_paths=None, ip_token=None):
    kwargs = build_kwargs(profile, payload, image_paths)
    client = Client(profile.space_id, **client_kwargs(token, ip_token))
    try:
        result = predict_logged(client, space_id=profile.space_id,
                                api_name=profile.api_name, kwargs=kwargs)
        if not isinstance(result, (tuple, list)) or len(result) != 3:
            raise RuntimeError("MiniMax H3 上游结果应为 (image, seed, prompt_used)")
        return result_bytes(select_result(result[0], []), profile)
    finally:
        client.close()
