"""MiniMax H3 workflow adapter for ordinary video generation."""
import math
import secrets
from pathlib import Path
from gradio_client import Client, handle_file
from .upstream_auth import client_kwargs
from .wan2_2_i2v_v3 import VideoUpstreamError

MODEL_ID = "minimax-h3-turbo"
SPACE_ID = "kazufusa/MiniMax-H3-Turbo-Lora-UNCENSORED-Workflow"
API_NAME = "/output_video"
CANVASES = ("960x544 · 16:9 fast", "1024x576 · 16:9 fast",
            "1152x640 · 16:9", "1280x704 · 16:9", "1344x768 · 16:9 full",
            "544x960 · 9:16 fast", "640x1152 · 9:16", "768x1344 · 9:16 full",
            "544x544 · 1:1 fast", "768x768 · 1:1 full",
            "768x576 · 4:3 fast", "1024x768 · 4:3 full",
            "576x768 · 3:4 fast", "768x1024 · 3:4 full",
            "1152x512 · 21:9 fast", "1536x672 · 21:9 full")

def build_args(payload, image_paths):
    if len(image_paths) > 1:
        raise ValueError("Workflow accepts at most one first frame")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt required")
    extra = payload.get("extra_body", {})
    if not isinstance(extra, dict) or set(extra) - {"canvas", "steps", "upsample", "lora"}:
        raise ValueError("unsupported workflow options")
    canvas = extra.get("canvas", CANVASES[0])
    steps = extra.get("steps", 6)
    upsample = extra.get("upsample", False)
    if canvas not in CANVASES or type(steps) is not int or steps not in (4, 6, 28):
        raise ValueError("invalid canvas or steps")
    if type(upsample) is not bool or extra.get("lora", "off") != "off":
        raise ValueError("invalid workflow options; LoRA is disabled")
    seconds = float(payload.get("seconds", 5))
    # Conservative gateway limits, not a claim about upstream limits.
    if not math.isfinite(seconds) or not 1 <= seconds <= 10:
        raise ValueError("seconds must be 1–10")
    seed = payload.get("seed", -1)
    if type(seed) is not int or not -1 <= seed <= 2147483647:
        raise ValueError("invalid seed")
    return (prompt, handle_file(image_paths[0]) if image_paths else None,
            canvas, seconds, steps, seed, upsample, "off")

def resolve_seed(payload):
    if payload.get("seed", -1) == -1:
        return {**payload, "seed": secrets.randbelow(2**31)}
    return payload

def run(payload, token, image_paths, ip_token=None, timeout=900):
    phase = "prepare"
    client = None
    try:
        payload = resolve_seed(payload)
        args = build_args(payload, image_paths)
        phase = "client_init"
        client = Client(SPACE_ID, **client_kwargs(token, ip_token), oauth_token=token)
        phase = "submit"
        job = client.submit(*args, api_name=API_NAME)
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
        if not isinstance(result, (tuple, list)) or len(result) != 3:
            raise ValueError("invalid workflow result")
        video, report, refined = result
        if isinstance(video, dict):
            video = video.get("video") or video.get("path")
        if not isinstance(video, (str, Path)):
            raise ValueError("invalid video path")
        path = Path(video)
        if not path.is_file() or path.suffix.lower() != ".mp4" or not 0 < path.stat().st_size <= 128*1024*1024:
            raise ValueError("invalid MP4")
        with path.open("rb") as stream:
            if stream.read(8)[4:8] != b"ftyp":
                raise ValueError("invalid MP4 header")
        # Keep user text and credentials out of diagnostics and public task metadata.
        return path.read_bytes(), payload["seed"]
    except Exception as exc:
        raise VideoUpstreamError(phase, exc) from None
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
