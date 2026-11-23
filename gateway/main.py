from __future__ import annotations
import base64
import logging
import uuid
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel, Field, ConfigDict
from .models import load_profiles
from .adapter import invoke, parse_size
from .token_quota import quota_status
from .temp_images import get_temp_image, save_temp_image, public_image_url

app = FastAPI(title="HF Spaces OpenAI Image Gateway", version="0.1.0")

access_logger = logging.getLogger("uvicorn.error")


@app.middleware("http")
async def request_duration_log(request: Request, call_next):
    """Log complete HTTP latency (including ZeroGPU queue); never log token or prompt."""
    start = time.monotonic()
    request_id = uuid.uuid4().hex[:12]
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Request-Duration-Ms"] = str(round((time.monotonic() - start) * 1000))
        return response
    finally:
        access_logger.info(
            "[gateway-request] id=%s method=%s path=%s status=%s duration_ms=%d",
            request_id, request.method, request.url.path, status,
            round((time.monotonic() - start) * 1000),
        )


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    prompt: str = Field(min_length=1, max_length=4000)
    size: str | None = None
    n: int = Field(default=1, ge=1, le=4)
    response_format: str = "url"
    seed: int = -1
    extra_body: dict[str, Any] = Field(default_factory=dict)


def err(status, code, message):
    raise HTTPException(status_code=status, detail={"error": {"code": code, "message": message, "type": "invalid_request_error"}})

@app.exception_handler(HTTPException)
async def error_handler(_, exc):
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"error": {"type": "invalid_request_error", "code": "bad_request", "message": str(exc.detail)}})


def request_hf_token(authorization: str | None, x_hf_token: str | None) -> str | None:
    """Optional per-request HF credential. Never fall back to the server owner's token."""
    explicit = (x_hf_token or "").strip()
    auth = (authorization or "").strip()
    if auth.lower().startswith("bearer "):
        candidate = auth[7:].strip()
        if candidate.startswith("hf_"):
            if explicit and explicit != candidate:
                err(400, "conflicting_hf_tokens", "Provide HF Token in only one header")
            explicit = candidate
    if explicit and not re.fullmatch(r"hf_[A-Za-z0-9_]+", explicit):
        err(400, "invalid_hf_token", "HF token must start with hf_; otherwise omit it for anonymous access")
    return explicit or None


def upstream_error_message(exc: Exception) -> tuple[int, str, str]:
    """Return a small non-sensitive error and an explicit no-quota hint when recognized."""
    raw = str(exc)
    message = re.sub(r"hf_[A-Za-z0-9_]+", "[HF_TOKEN_REDACTED]", raw)
    message = re.sub(r"Bearer\\s+[^\\s,;]+", "Bearer [REDACTED]", message, flags=re.I)
    msg = message.lower()
    if any(term in msg for term in ("quota", "zero gpu", "zerogpu", "gpu seconds", "daily limit", "rate limit", "too many requests", "429", "no gpu time")):
        return (429, "upstream_quota_exceeded",
                "Upstream ZeroGPU quota or rate limit exhausted. Retry later or provide your own HF Token via X-HF-Token.")
    # Avoid returning upstream exception text, which can include private headers or URLs.
    return (502, "upstream_failed",
            "Upstream Space failed (" + type(exc).__name__ + "). Check Space status and its Container Logs.")



def profile_for(model: str, mode: str):
    try:
        profile = load_profiles().get(model)
    except Exception:
        err(503, "catalog_invalid", "Model catalog configuration invalid")
    if not profile:
        err(404, "model_not_found", "Unknown or disabled model")
    if mode not in profile.modes:
        err(400, "unsupported_mode", f"{model} does not support {mode}")
    return profile


def produce(payload: dict, token: str | None, mode="generations", paths=None):
    p = profile_for(payload["model"], mode)
    response_format = payload.get("response_format", "url")
    if response_format not in ("url", "b64_json"):
        err(400, "unsupported_format", "Supported response_format values: url, b64_json")
    try:
        parse_size(payload.get("size") or p.default_size, p)
        data = []
        for i in range(payload.get("n", 1)):
            item = dict(payload)
            if item.get("seed", -1) >= 0:
                item["seed"] += i
            png = invoke(p, item, token, image_paths=paths)
            if response_format == "url":
                data.append({"url": public_image_url(save_temp_image(png))})
            else:
                data.append({"b64_json": base64.b64encode(png).decode("ascii")})
    except ValueError as ex:
        err(400, "invalid_parameters", str(ex))
    except Exception as ex:
        status, code, message = upstream_error_message(ex)
        err(status, code, message)
    return {"created": int(time.time()), "data": data, "model": p.id}

@app.get("/health")
def health():
    return {"ok": True, "auth_required": False, "models": len(load_profiles())}

# Process-local publication timestamp, not the upstream model's release date.
CATALOG_PUBLISHED_AT = int(time.time())


@app.get("/v1/models")
def models():
    return {"data": [{
        "id": p.id,
        "object": "model",
        "created": CATALOG_PUBLISHED_AT,
        "owned_by": p.space_id.split("/", 1)[0],
        "type": "model",
        "display_name": p.display_name or p.id,
    } for p in load_profiles().values()]}


@app.get("/v1/models/detail")
def models_detail():
    return {"object": "list", "data": [{
        "id": p.id,
        "object": "model",
        "owned_by": p.space_id,
        "description": p.description,
        "modes": p.modes,
        "default_size": p.default_size,
        "sizes": p.allowed_sizes,
        "extra_parameters": {k: v.model_dump() for k, v in p.extras.items()},
    } for p in load_profiles().values()]}

@app.get("/v1/hf/status")
def hf_status(authorization: str | None = Header(default=None), x_hf_token: str | None = Header(default=None)):
    """Optional caller HF Token status + ZeroGPU remaining quota / reset time."""
    token = request_hf_token(authorization, x_hf_token)
    try:
        return quota_status(token)
    except ValueError as exc:
        err(400, "invalid_hf_token", str(exc))
    except PermissionError:
        err(401, "hf_token_invalid", "HF Token validation failed; check token permissions or expiration")
    except ConnectionError:
        err(502, "hf_quota_unavailable", "HF quota unavailable; ensure Billing read permission and retry later")


@app.get("/v1/files/{image_name}")
def serve_temp_image(image_name: str):
    path = get_temp_image(image_name)
    if path is None:
        err(404, "image_not_found", "Temporary image missing or expired")
    return FileResponse(path, media_type="image/png", filename=image_name,
                        content_disposition_type="inline",
                        headers={"Cache-Control": "private, max-age=60", "X-Content-Type-Options": "nosniff"})


@app.post("/v1/images/generations")
def generate(body: GenerateRequest, authorization: str | None = Header(default=None), x_hf_token: str | None = Header(default=None)):
    token = request_hf_token(authorization, x_hf_token)
    return produce(body.model_dump(exclude_none=True, exclude_unset=False), token)

@app.post("/v1/images/edits")
async def edit(model: str = Form(...), prompt: str = Form(...), image: list[UploadFile] = File(...), size: str | None = Form(None), n: int = Form(1), response_format: str = Form("url"), seed: int = Form(-1), extra_body: str = Form("{}"), authorization: str | None = Header(default=None), x_hf_token: str | None = Header(default=None)):
    import json
    token = request_hf_token(authorization, x_hf_token)
    profile = profile_for(model, "edits")
    if profile.adapter != "qwen_image" and not ("images" in profile.bindings or "image" in profile.bindings):
        err(400, "unsupported_edit", "Model has no image input binding")
    try:
        extras = json.loads(extra_body)
        if not isinstance(extras, dict):
            raise ValueError("extra_body must be object")
        body = GenerateRequest(model=model, prompt=prompt, size=size, n=n, response_format=response_format, seed=seed, extra_body=extras)
    except (ValueError, TypeError) as ex:
        err(400, "invalid_parameters", str(ex))
    if not 1 <= len(image) <= 4:
        err(400, "invalid_image_count", "1-4 images required")
    paths = []
    with tempfile.TemporaryDirectory() as temp:
        for i, upload in enumerate(image):
            if upload.content_type not in ("image/jpeg", "image/png", "image/webp"):
                err(400, "invalid_image", "Only PNG, JPEG and WebP supported")
            content = await upload.read(10 * 1024 * 1024 + 1)
            if len(content) > 10 * 1024 * 1024:
                err(413, "image_too_large", "Image exceeds 10 MB")
            from PIL import Image
            import io
            try:
                with Image.open(io.BytesIO(content)) as im:
                    im.verify()
            except Exception:
                err(400, "invalid_image", "Invalid image content")
            dest = Path(temp) / f"{i}.png"
            # preserve original encoding, with correct suffix for Gradio
            ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[upload.content_type]
            dest = dest.with_suffix(ext)
            dest.write_bytes(content)
            paths.append(str(dest))
        return produce(body.model_dump(), token, mode="edits", paths=paths)

# Serve Gradio studio on / while preserving the FastAPI image endpoints.
import gradio as gr
from .ui import create_ui, CSS
app = gr.mount_gradio_app(app, create_ui(), path="/", css=CSS)
