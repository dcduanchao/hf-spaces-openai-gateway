"""Runtime API-discovered adapter for third-party Qwen Image 2.1 Gradio Spaces.

Gradio Spaces are not a stable standard: discover endpoint names and argument
names instead of silently hardcoding the official app's API for an unknown fork.
"""
from __future__ import annotations
import json
import logging
import secrets
from gradio_client import Client, handle_file


logger = logging.getLogger(__name__)


def api_endpoints(client):
    schema = client.view_api(print_info=False, return_format="dict")
    if not isinstance(schema, dict):
        raise ValueError("上游 Gradio API Schema 不可用")
    endpoints = schema.get("named_endpoints", {})
    logger.info("Qwen discovered API endpoints: %s", list(endpoints))
    return endpoints


def parameters(api, endpoint):
    info = api.get(endpoint, {})
    return {x.get("parameter_name") for x in info.get("parameters", [])
            if isinstance(x, dict) and x.get("parameter_name")}


def endpoint(api, candidates):
    for name in candidates:
        if name in api:
            return name
    return None


def arguments(api, name, values):
    names = parameters(api, name)
    if not names:
        raise ValueError(f"接口 {name} 未提供可识别的具名参数定义")
    return {k: v for k, v in values.items() if k in names}


def run(profile, payload, token, image_paths=None):
    # Delayed import avoids an import cycle.
    from .adapter import parse_size, select_result, result_bytes
    w, h = parse_size(payload.get("size") or profile.default_size, profile)
    extras = dict(payload.get("extra_body") or {})
    for k, v in payload.items():
        if k not in ("model", "prompt", "size", "n", "response_format", "seed", "extra_body"):
            if k in extras:
                raise ValueError(f"重复扩展参数：{k}")
            extras[k] = v
    unknown = set(extras) - set(profile.extras)
    if unknown:
        raise ValueError(f"当前模型不支持扩展参数：{sorted(unknown)}")
    for key, spec in profile.extras.items():
        from .models import validate_extra
        if key in extras:
            try:
                validate_extra(extras[key], spec)
            except ValueError as e:
                raise ValueError(f"{key}: {e}") from e

    requested_seed = int(payload.get("seed", -1))
    if requested_seed < -1:
        raise ValueError("seed 必须是 -1（随机）或非负整数")
    actual_seed = secrets.randbelow(2**31 - 1) if requested_seed == -1 else requested_seed
    logger.info("Qwen seed normalized: random=%s", requested_seed == -1)

    values = {
        "prompt": payload["prompt"], "original_prompt": payload["prompt"],
        "text": payload["prompt"], "input_prompt": payload["prompt"],
        "width": w, "height": h,
        "seed": actual_seed,
        "randomize_seed": False,
        "negative_prompt": extras.get("negative_prompt", ""),
        "enable_extend": extras.get("enable_extend", False),
        "quality": extras.get("quality", "speed"),
        "custom_size": extras.get("custom_size", True),
        "input_images": [handle_file(p) for p in (image_paths or [])],
        "image_paths": [handle_file(p) for p in (image_paths or [])],
        "images": [handle_file(p) for p in (image_paths or [])],
        "image": handle_file(image_paths[0]) if image_paths else None,
        "log_dir": "",
    }
    client = Client(profile.space_id, token=token or False, verbose=False)
    try:
        api = api_endpoints(client)
        # Official Qwen-style two-stage UI: CPU prepare, then GPU generation.
        prepare = endpoint(api, ["/prepare_request", "/prepare"])
        generate = endpoint(api, ["/generate_request"])
        if prepare and generate:
            logger.info("Qwen adapter two-stage call: %s -> %s", prepare, generate)
            prepared = client.predict(api_name=prepare, **arguments(api, prepare, values))
            if not isinstance(prepared, (list, tuple)) or len(prepared) < 4:
                raise ValueError(f"上游 {prepare} 未返回预期的 request_state")
            values.update({
                "request_state": prepared[3],
                "seed": prepared[1],
            })
            result = client.predict(api_name=generate, **arguments(api, generate, values))
        else:
            direct = endpoint(api, ["/generate", "/infer", "/predict", "/text_to_image"])
            if not direct:
                raise ValueError(
                    "目标 Space 未暴露已支持的 Gradio 生图端点。请通过 inspect_space.py "
                    "检查其 API，并为该 Space 编写专用适配器。"
                )
            if image_paths and not ({"image", "images", "input_images", "image_paths"} & parameters(api, direct)):
                raise ValueError("当前生图接口不接受参考图")
            logger.info("Qwen adapter direct endpoint: %s", direct)
            result = client.predict(api_name=direct, **arguments(api, direct, values))
        return result_bytes(select_result(result, profile.result_path), profile)
    finally:
        client.close()
