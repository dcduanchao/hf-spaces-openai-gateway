"""BFS Best Face Swap: two-image head swap using its Gradio /face_swap API."""
from __future__ import annotations

import logging

from gradio_client import Client, handle_file

logger = logging.getLogger("uvicorn.error")

from .adapter import result_bytes, select_result
from .upstream_auth import client_kwargs
from .upstream_logging import predict_logged

TURBO = "Turbo · Viggle 6-step"
BASE = "Base · Qwen-Image-2.1"
BASE_MODELS = (TURBO, BASE)
RESOLUTIONS = (1024, 1536, 2048)
DEFAULT_PROMPT = (
    "head_swap: start with <image1> as the base image, keeping its lighting, "
    "environment, and background. remove the head from <image1> completely "
    "and replace it with the head from <image2>, strictly preserving the hair, "
    "eye color, nose structure from <image2>. copy the direction of the eye, "
    "head rotation, micro expressions from <image1>, high quality, sharp details, 4k"
)
MAX_SEED = 2**31 - 1


def build_kwargs(profile, payload, image_paths, steps):
    """image_paths ordered as [reference_face, target_image]."""
    paths = image_paths or []
    if len(paths) != 2:
        raise ValueError("面部交换必须恰好上传两张图：参考人脸、目标身体/场景")
    extras = payload.get("extra_body") or {}
    if not isinstance(extras, dict):
        raise ValueError("extra_body 必须是 JSON 对象")
    if set(extras) - {"base_model"}:
        raise ValueError("面部交换只允许配置 base_model")
    base_model = extras.get("base_model", TURBO)
    if base_model not in BASE_MODELS:
        raise ValueError("base_model 必须是 Turbo · Viggle 6-step 或 Base · Qwen-Image-2.1")
    raw_seed = payload.get("seed", -1)
    if isinstance(raw_seed, bool) or not isinstance(raw_seed, int) or not -1 <= raw_seed <= MAX_SEED:
        raise ValueError("seed 必须是 -1 或 0～2147483647")
    if isinstance(steps, bool) or not isinstance(steps, (int, float)) or not (0 < steps <= 100):
        raise ValueError("上游推理步数无效")
    return {
        "reference_face": handle_file(paths[0]),
        "target_image": handle_file(paths[1]),
        "base_model": base_model,
        "resolution": 1536,
        "num_inference_steps": float(steps),
        "lora_scale": 1.0,
        "seed": float(0 if raw_seed == -1 else raw_seed),
        "randomize_seed": raw_seed == -1,
        "prompt": DEFAULT_PROMPT,
    }


def image_pair(result):
    """Unpack ImageSlider (before, after), preserving both outputs in order."""
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise RuntimeError("BFS 上游输出格式异常：应为 [before/after, used_seed]")
    pair = result[0]
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise RuntimeError("BFS 上游未返回有效的 Before / After 图片对")
    images = [select_result(value, []) if value is not None else None for value in pair]
    if any(not isinstance(image, (str, dict)) for image in images):
        raise RuntimeError("BFS 上游返回的 Before / After 图片路径无效")
    return images


def image_after(result):
    """Imageslider returns (before, after) or None. Return the swapped result."""
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise RuntimeError("BFS 上游输出格式异常：应为 [before/after, used_seed]")
    pair = result[0]
    if not isinstance(pair, (tuple, list)) or len(pair) != 2 or pair[1] is None:
        raise RuntimeError("BFS 上游未返回交换后的图片")
    image = select_result(pair[1], [])
    if not isinstance(image, (str, dict)):
        raise RuntimeError("BFS 上游生成的图片路径无效")
    return image


def run(profile, payload, token, image_paths=None, ip_token=None):
    paths = image_paths or []
    extras = payload.get("extra_body") or {}
    if not isinstance(extras, dict):
        raise ValueError("extra_body 必须是 JSON 对象")
    base_model = extras.get("base_model", TURBO)
    if base_model not in BASE_MODELS:
        raise ValueError("无效的面部交换基础模型")
    if len(paths) != 2:
        raise ValueError("面部交换需要恰好两张图片")
    # Keep local validation separate from Gradio Client validation so that
    # upstream ValueErrors are not incorrectly reported as HTTP 400.
    try:
        client = Client(profile.space_id, **client_kwargs(token, ip_token))
    except ValueError as exc:
        logger.error("[hf-upstream] bfs_failure phase=client_init type=%s", type(exc).__name__)
        raise RuntimeError("BFS 上游 Gradio Client 初始化失败") from None
    phase = "prepare"
    try:
        # The Turbo model uses six steps; the Base model uses the native
        # Qwen-Image-2.1 30-step default. Do not call /on_mode_change here:
        # Gradio may return a component-update payload instead of a number,
        # which previously caused the Base request to fail before inference.
        steps = 6.0 if base_model == TURBO else 30.0
        logger.info("[hf-upstream] bfs_parameters mode=%s steps=%s", "turbo" if base_model == TURBO else "base", steps)
        kwargs = build_kwargs(profile, payload, paths, steps)
        phase = "face_swap"
        response = predict_logged(client, space_id=profile.space_id,
                                  api_name=profile.api_name, kwargs=kwargs)
        phase = "result"
        return [result_bytes(image, profile) for image in image_pair(response)]
    except ValueError as exc:
        # Gradio endpoint validation, remote errors and response file handling
        # should never be categorized as malformed local user parameters.
        logger.error("[hf-upstream] bfs_failure phase=%s type=%s", phase, type(exc).__name__)
        raise RuntimeError(f"BFS 上游调用失败（阶段：{phase}）") from None
    finally:
        client.close()
