"""Viggle Qwen Image 2.1 Turbo (v0.3) Space adapter.

The Space exposes a single Gradio /generate event with ten *positional*
parameters. It is not compatible with the original Qwen /prepare_request /
generate_request protocol. Keep the user's HF token and history in the shared
gateway pipeline.
"""
from __future__ import annotations

import secrets

from gradio_client import Client, handle_file

from .adapter import parse_size, result_bytes, select_result
from .models import validate_extra
from .upstream_auth import client_kwargs
from .upstream_logging import logger

# Exact preset label documented by the upstream /generate API.
# Other valid dimensions take its "Custom" path with explicit width/height.
SQUARE_PRESET = "1:1 · 1024×1024 (1024² area)"
CUSTOM_PRESET = "Custom · set width and height below"
MAX_REFERENCES = 4
MAX_SEED = 2**31 - 1  # Matches Viggle Gradio Seed widget upper bound


def build_args(profile, payload, image_paths=None):
    width, height = parse_size(payload.get("size") or profile.default_size, profile)
    paths = image_paths or []
    if len(paths) > MAX_REFERENCES:
        raise ValueError("Viggle Turbo 最多接受 4 张参考图")

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
                raise ValueError(f"重复扩展参数：{key}")
            extras[key] = value
    unknown = set(extras) - set(profile.extras)
    if unknown:
        raise ValueError(f"未知 Viggle Turbo 参数：{sorted(unknown)}")

    steps_spec = profile.extras["steps"]
    steps = validate_extra(extras.get("steps", steps_spec.default), steps_spec)

    user_seed = int(payload.get("seed", -1))
    if user_seed < -1 or user_seed > MAX_SEED:
        raise ValueError("seed 应为 -1（随机）或 0～2147483647")
    seed = secrets.randbelow(MAX_SEED + 1) if user_seed == -1 else user_seed

    # Gradio Gallery input: list of objects wrapping the FileData.
    gallery = [{"image": handle_file(path), "caption": None} for path in paths]
    # Gradio upstream sliders require dimensions >= 512 in multiples of 32.
    if (width != 1024 or height != 1024) and (
        width < 512 or height < 512 or width % 32 or height % 32
    ):
        raise ValueError("Viggle 自定义宽高需至少 512，且为 32 的倍数")
    preset = SQUARE_PRESET if (width, height) == (1024, 1024) else CUSTOM_PRESET
    return [
        payload["prompt"],  # Prompt
        gallery,            # References (optional)
        preset,             # Output size
        seed,               # Seed
        False,              # Randomize seed: seed has already been chosen
        "Off",              # Enhance prompt disabled as requested
        steps,              # Steps (6 by default)
        width,              # Custom width
        height,             # Custom height
        "v0.3",             # Pinned model version
    ]


def run(profile, payload, token, image_paths=None, ip_token=None):
    args = build_args(profile, payload, image_paths)
    try:
        client = Client(profile.space_id, **client_kwargs(token, ip_token))
    except ValueError:
        logger.error("[hf-upstream] viggle_client_init_validation_failure")
        raise RuntimeError("Viggle Turbo upstream client initialization rejected request") from None
    try:
        # predict_logged assumes named parameters. Viggle requires positional
        # parameters, so log only non-sensitive operational metadata.
        logger.info(
            "[hf-upstream] viggle space=%s endpoint=%s references=%d size=%sx%s steps=%d version=v0.3",
            profile.space_id, profile.api_name, len(args[1]), args[7], args[8], args[6],
        )
        try:
            result = client.predict(*args, api_name=profile.api_name)
            if not isinstance(result, (list, tuple)) or len(result) != 3:
                raise RuntimeError("Viggle Turbo upstream result format invalid")
            return result_bytes(select_result(result[0], []), profile)
        except ValueError as exc:
            # A gradio_client ValueError after request construction is an
            # upstream protocol/preprocessing failure, not necessarily a
            # caller's invalid input. Never expose upstream exception text:
            # it can contain prompt text or private file metadata.
            logger.error("[hf-upstream] viggle_client_validation_failure type=%s", type(exc).__name__)
            raise RuntimeError("Viggle Turbo upstream client rejected prepared request") from None
    finally:
        client.close()
