"""Wan 2.2 I2V adapter; separate from image conversion pipeline."""
from __future__ import annotations

import math
import logging
from pathlib import Path
from gradio_client import Client, handle_file
from .upstream_auth import client_kwargs

logger = logging.getLogger("uvicorn.error")

MODEL_ID = "wan2.2-i2v-v3"
SPACE_ID = "duan0320/wan2-2-i2v-v3"
API_NAME = "/generate_video"
SCHEDULERS = ("FlowMatchEulerDiscrete", "SASolver", "DEISMultistep",
              "DPMSolverMultistepInverse", "UniPCMultistep",
              "DPMSolverMultistep", "DPMSolverSinglestep")
DEFAULTS = dict(steps=6, negative_prompt="blurry, low quality, static",
                guidance_scale=1.0, guidance_scale_2=1.0, quality=6,
                scheduler="UniPCMultistep", flow_shift=3.0,
                frame_multiplier=16, safe_mode=True)

def build_kwargs(payload, image_paths):
    if len(image_paths or []) not in (1, 2):
        raise ValueError("Wan requires one or two input images")
    if payload.get("model", MODEL_ID) != MODEL_ID:
        raise ValueError("unknown video model")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt is required")
    extras = payload.get("extra_body", {})
    if not isinstance(extras, dict) or set(extras) - set(DEFAULTS):
        raise ValueError("unsupported video extra_body fields")
    options = {**DEFAULTS, **extras}
    def number(name, value, low, high, integer=False):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not low <= value <= high
                or (integer and not isinstance(value, int))):
            raise ValueError(f"invalid {name}")
        return value
    for name, lo, hi, integer in (
        ("steps", 1, 30, True), ("quality", 1, 10, True),
        ("guidance_scale", 0, 10, False), ("guidance_scale_2", 0, 10, False),
        ("flow_shift", .5, 15, False),
    ):
        number(name, options[name], lo, hi, integer)
    if options["scheduler"] not in SCHEDULERS:
        raise ValueError("invalid scheduler")
    fps = options["frame_multiplier"]
    if type(fps) is not int or fps not in (16, 32, 64, 128):
        raise ValueError("invalid frame_multiplier")
    if type(options["safe_mode"]) is not bool or not isinstance(options["negative_prompt"], str):
        raise ValueError("invalid video options")
    seed = number("seed", payload.get("seed", -1), -1, 2147483647, True)
    seconds = payload.get("seconds", 3.5)
    if isinstance(seconds, str):
        try:
            seconds = float(seconds)
        except ValueError:
            raise ValueError("invalid seconds") from None
    number("seconds", seconds, .5, 20.1)
    return dict(input_image=handle_file(image_paths[0]),
                last_image=handle_file(image_paths[1]) if len(image_paths) == 2 else None,
                prompt=prompt, duration_seconds=float(seconds), seed=42 if seed == -1 else seed,
                randomize_seed=seed == -1, lora_groups=[], auto_lora_enabled=False,
                video_component=True, **options)

def unpack_result(result):
    if not isinstance(result, (tuple, list)) or len(result) != 3:
        raise RuntimeError("invalid Wan output")
    path = result[1]
    if not isinstance(path, (str, Path)) or not Path(path).is_file():
        raise RuntimeError("Wan did not return a downloaded video file")
    file = Path(path)
    if file.suffix.lower() != ".mp4" or not 0 < file.stat().st_size <= 128 * 1024 * 1024:
        raise RuntimeError("invalid MP4 file")
    with file.open("rb") as stream:
        if stream.read(8)[4:8] != b"ftyp":
            raise RuntimeError("invalid MP4 header")
    seed = result[2]
    if (isinstance(seed, bool) or not isinstance(seed, (int, float))
            or not math.isfinite(seed) or not 0 <= seed <= 2147483647 or int(seed) != seed):
        raise RuntimeError("invalid Wan seed")
    return file, int(seed)

class VideoUpstreamError(RuntimeError):
    def __init__(self, phase, exc):
        self.phase = phase
        # Only fixed, public diagnostic messages. Never return or log raw
        # exception text (may contain credentials, prompt or signed URLs).
        message = str(exc).lower()
        if any(word in message for word in ("quota", "zerogpu runs", "gpu seconds", "rate limit", "429")):
            self.code = "upstream_quota_exceeded"
            self.public_message = "ZeroGPU quota or rate limit reached; check HF Token and remaining quota."
        elif isinstance(exc, TimeoutError):
            self.code = "upstream_timeout"
            self.public_message = "Upstream video job timed out."
        elif "frame_multiplier" in message or ("16" in message and "choice" in message):
            self.code = "upstream_fps_invalid"
            self.public_message = "Upstream rejected the FPS parameter; check Dropdown value type."
        elif "last_image" in message or ("nonetype" in message and "image" in message):
            self.code = "upstream_last_image_invalid"
            self.public_message = "Upstream rejected the optional last image."
        elif "401" in message or "403" in message or "unauthorized" in message:
            self.code = "upstream_auth_failed"
            self.public_message = "Upstream access denied; verify HF Token and Space permissions."
        else:
            self.code = {"client_init": "upstream_client_failed", "submit": "upstream_submit_failed",
                         "wait": "upstream_inference_failed", "result": "upstream_result_invalid",
                         "prepare": "upstream_input_failed"}.get(phase, "upstream_failed")
            self.public_message = "Video generation failed at " + phase + " (" + type(exc).__name__ + ")."
        super().__init__(self.public_message)

def run(payload, token, image_paths, ip_token=None, timeout=900):
    """Return MP4 bytes and actual seed; intended for a background worker."""
    phase = "prepare"
    client = None
    try:
        kwargs = build_kwargs(payload, image_paths)
        logger.info("[hf-video] space=%s endpoint=%s images=%d fps=%s fps_type=%s seconds=%s steps=%s",
                    SPACE_ID, API_NAME, len(image_paths), kwargs["frame_multiplier"],
                    type(kwargs["frame_multiplier"]).__name__, kwargs["duration_seconds"], kwargs["steps"])
        phase = "client_init"
        client = Client(SPACE_ID, **client_kwargs(token, ip_token))
        phase = "submit"
        job = client.submit(api_name=API_NAME, **kwargs)
        phase = "wait"
        try:
            result = job.result(timeout=timeout)
        except TimeoutError:
            try:
                job.cancel()
            except Exception:
                pass
            raise
        phase = "result"
        logger.info("[hf-video] result_type=%s item_types=%s", type(result).__name__,
                    [type(item).__name__ for item in result] if isinstance(result, (tuple, list)) else [])
        path, seed = unpack_result(result)
        return path.read_bytes(), seed
    except Exception as exc:
        error = VideoUpstreamError(phase, exc)
        logger.error("[hf-video] phase=%s code=%s error_type=%s", phase, error.code, type(exc).__name__)
        raise error from None
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                logger.warning("[hf-video] client close failed")
