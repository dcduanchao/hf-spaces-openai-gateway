"""Exact Qwen-Image-2.1 adapter, based on its published Gradio API contract.

Endpoints:
  /prepare_request(original_prompt, input_images, enable_extend, custom_size,
                   quality, seed, randomize_seed) -> [image, seed, rewritten_prompt]
  /generate_request(original_prompt, enable_extend, custom_size, log_dir,
                    seed, height, width, negative_prompt) -> [image]
"""
from __future__ import annotations

import logging
import secrets
from gradio_client import Client, handle_file
from .models import validate_extra
from .upstream_logging import predict_logged
from .upstream_auth import client_kwargs

logger = logging.getLogger(__name__)


def normalize_seed(raw):
    seed = int(raw)
    if seed < -1:
        raise ValueError("seed 只能是 -1（随机）或非负整数")
    return secrets.randbelow(2**31 - 1) if seed == -1 else seed


def build_requests(profile, payload, image_paths=None):
    from .adapter import parse_size

    width, height = parse_size(payload.get("size") or profile.default_size, profile)
    extras = payload.get("extra_body") or {}
    if not isinstance(extras, dict):
        raise ValueError("extra_body 必须是 JSON 对象")
    extras = dict(extras)
    for name, val in payload.items():
        if name not in {
            "model", "prompt", "size", "n", "response_format", "seed",
            "extra_body", "user", "quality", "background", "output_format",
            "moderation", "style",
        }:
            if name in extras:
                raise ValueError(f"重复扩展参数：{name}")
            extras[name] = val
    # OpenAI SDK extra_body merges its properties into the top-level JSON.
    # quality may also be passed as a standard Images API property.
    if "quality" in payload and payload["quality"] in ("speed", "quality"):
        extras.setdefault("quality", payload["quality"])

    unknown = set(extras) - set(profile.extras)
    if unknown:
        raise ValueError(f"未知模型扩展参数：{sorted(unknown)}")

    values = {}
    for name, spec in profile.extras.items():
        value = extras.get(name, spec.default)
        try:
            values[name] = validate_extra(value, spec)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc

    actual_seed = normalize_seed(payload.get("seed", -1))
    paths = image_paths or []
    if len(paths) > 10:
        raise ValueError("Qwen 上游最多接受 10 张参考图")

    # Gallery input is a list[dict(image=FileData,caption=str|None)].
    gallery = [{"image": handle_file(path), "caption": None} for path in paths]

    prepare = {
        "input_images": gallery,
        "original_prompt": payload["prompt"],
        "enable_extend": False,  # 禁用提示词扩展
        "custom_size": values["custom_size"],
        "quality": values["quality"],
        "seed": actual_seed,
        "randomize_seed": False,
    }
    generate = {
        "original_prompt": payload["prompt"],
        "enable_extend": False,  # 禁用提示词扩展
        "custom_size": values["custom_size"],
        "log_dir": "./generation_logs_paper_case",
        "seed": actual_seed,
        "height": height,
        "width": width,
        "negative_prompt": values["negative_prompt"],
    }
    return prepare, generate


def run(profile, payload, token, image_paths=None, ip_token=None):
    from .adapter import select_result, result_bytes

    prepare, generate = build_requests(profile, payload, image_paths)
    client = Client(profile.space_id, **client_kwargs(token, ip_token))
    try:
        logger.info("Qwen call /prepare_request (seed nonnegative; %d image(s))", len(prepare["input_images"]))
        prepared = predict_logged(client, space_id=profile.space_id, api_name="/prepare_request", kwargs=prepare)
        # The documented response is exactly [result_image, effective_seed, rewritten_prompt].
        if not isinstance(prepared, (list, tuple)) or len(prepared) != 3:
            raise ValueError("上游 /prepare_request 结果格式不符合 API 文档：应返回 3 个元素")
        effective_seed = prepared[1]
        if effective_seed is None or int(effective_seed) < 0:
            raise ValueError("上游 /prepare_request 返回了无效 seed")
        generate["seed"] = int(effective_seed)
        logger.info("Qwen call /generate_request")
        result = predict_logged(client, space_id=profile.space_id, api_name="/generate_request", kwargs=generate)
        # /generate_request returns a list of one filepath.
        if isinstance(result, (list, tuple)) and len(result) != 1:
            raise ValueError("上游 /generate_request 结果格式不符合 API 文档：应返回 1 个元素")
        return result_bytes(select_result(result, profile.result_path), profile)
    finally:
        client.close()
