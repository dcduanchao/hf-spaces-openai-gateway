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
from fastapi import FastAPI, Header, HTTPException, Request
from starlette.datastructures import UploadFile
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel, Field, ConfigDict
from .models import load_profiles
from .adapter import invoke, parse_size
from .token_quota import quota_status
from .temp_images import get_temp_image, save_temp_image, public_image_url
from . import history
from .admin import router as admin_router
from .image_inputs import download_public_image, validate_png_bytes, ImageInputError

app = FastAPI(title="HF Spaces OpenAI Image Gateway", version="0.1.0")
app.include_router(admin_router)

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
    except ValueError as ex:
        err(400, "invalid_parameters", str(ex))

    started = time.monotonic()
    try:
        record_id = history.new_record(p.id, mode, payload["prompt"], {
            "size": payload.get("size") or p.default_size,
            "n": payload.get("n", 1),
            "seed": payload.get("seed", -1),
            "response_format": response_format,
            "extra_body": payload.get("extra_body") or {},
        })
    except (RuntimeError, OSError) as ex:
        err(503, "history_storage_unavailable", str(ex))

    data = []
    try:
        if paths:
            from PIL import Image
            import io
            for path in paths:
                with Image.open(path) as source:
                    target = io.BytesIO()
                    source.convert("RGB").save(target, format="PNG")
                history.store_image(record_id, target.getvalue(), input_image=True)
        for i in range(payload.get("n", 1)):
            item = dict(payload)
            if item.get("seed", -1) >= 0:
                item["seed"] += i
            png = invoke(p, item, token, image_paths=paths)
            history.store_image(record_id, png)
            if response_format == "url":
                data.append({"url": public_image_url(save_temp_image(png))})
            else:
                data.append({"b64_json": base64.b64encode(png).decode("ascii")})
    except ValueError as ex:
        history.finish_record(record_id, "failed", round((time.monotonic() - started) * 1000), "invalid_parameters")
        err(400, "invalid_parameters", str(ex))
    except Exception as ex:
        status, code, message = upstream_error_message(ex)
        history.finish_record(record_id, "failed", round((time.monotonic() - started) * 1000), code)
        err(status, code, message)
    history.finish_record(record_id, "success", round((time.monotonic() - started) * 1000))
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

def _edit_json_images(payload: dict) -> list[tuple[str, str]]:
    """Accept image/image_urls/images, preserving order and rejecting ambiguous mixes."""
    provided = [key for key in ("image", "images", "image_urls") if key in payload]
    if len(provided) > 1:
        err(400, "invalid_image", "Use only one of image, images, image_urls")
    if not provided:
        err(400, "invalid_image_count", "1-4 images required")
    raw = payload[provided[0]]
    values = [raw] if isinstance(raw, str) else raw
    if not isinstance(values, list) or not 1 <= len(values) <= 4:
        err(400, "invalid_image_count", "1-4 image URLs required")
    if not all(isinstance(url, str) for url in values):
        err(400, "invalid_image", "JSON image inputs must be public HTTPS URL strings")
    return [("url", url) for url in values]


@app.post("/v1/images/edits")
async def edit(request: Request, authorization: str | None = Header(default=None),
               x_hf_token: str | None = Header(default=None)):
    """OpenAI multipart edits + JSON public HTTPS URLs + mixed multipart URLs/files."""
    import asyncio
    import json
    from pydantic import ValidationError

    token = request_hf_token(authorization, x_hf_token)
    content_type = request.headers.get("content-type", "").lower()
    sources: list[tuple[str, Any]] = []
    try:
        if content_type.startswith("application/json"):
            raw = await request.json()
            if not isinstance(raw, dict):
                err(400, "invalid_parameters", "JSON body must be an object")
            sources = _edit_json_images(raw)
            reserved = {"model", "prompt", "size", "n", "seed", "response_format",
                        "extra_body", "image", "images", "image_urls"}
            # Accept SDK-style top-level extra fields just as generations does.
            extras = raw.get("extra_body", {})
            if not isinstance(extras, dict):
                err(400, "invalid_parameters", "extra_body must be a JSON object")
            for k, v in raw.items():
                if k not in reserved:
                    if k in extras:
                        err(400, "invalid_parameters", f"duplicate extra parameter: {k}")
                    extras[k] = v
            fields = {k: raw[k] for k in ("model", "prompt", "size", "n", "seed", "response_format") if k in raw}
            fields["extra_body"] = extras
        elif content_type.startswith("multipart/form-data"):
            form = await request.form()
            fields = {}
            for key in ("model", "prompt", "size", "n", "seed", "response_format"):
                if key in form:
                    fields[key] = form.get(key)
            for key in ("n", "seed"):
                if key in fields:
                    try:
                        fields[key] = int(fields[key])
                    except (ValueError, TypeError) as exc:
                        err(400, "invalid_parameters", f"{key} must be an integer")
            extras = json.loads(str(form.get("extra_body") or "{}"))
            if not isinstance(extras, dict):
                err(400, "invalid_parameters", "extra_body must be a JSON object")
            fields["extra_body"] = extras
            # FormData.multi_items preserves input order for repeated image/image_url.
            for key, value in form.multi_items():
                if key == "image":
                    if isinstance(value, UploadFile):
                        sources.append(("upload", value))
                    elif isinstance(value, str):
                        sources.append(("url", value))
                    else:
                        err(400, "invalid_image", "image must be an uploaded file or public HTTPS URL")
                elif key in ("image_url", "image_urls"):
                    if not isinstance(value, str):
                        err(400, "invalid_image", "image_url must be a public HTTPS URL")
                    sources.append(("url", value))
        else:
            err(415, "unsupported_media_type", "Use application/json or multipart/form-data")

        if not 1 <= len(sources) <= 4:
            err(400, "invalid_image_count", "1-4 images required (file upload or public HTTPS URLs)")
        try:
            body = GenerateRequest(**fields)
        except ValidationError as ex:
            err(400, "invalid_parameters", "Invalid edit parameters: " + "; ".join(
                str(e.get("loc")) + " " + e.get("msg", "") for e in ex.errors())[:400])
    except (ValueError, json.JSONDecodeError) as ex:
        err(400, "invalid_parameters", "Invalid JSON or image parameter")
    except HTTPException:
        raise

    profile = profile_for(body.model, "edits")
    if profile.adapter != "qwen_image" and not ("images" in profile.bindings or "image" in profile.bindings):
        err(400, "unsupported_edit", "Model has no image input binding")

    with tempfile.TemporaryDirectory(prefix="hf_edit_") as folder:
        paths = []
        for index, (kind, value) in enumerate(sources):
            try:
                if kind == "upload":
                    uploaded = value
                    if uploaded.content_type not in ("image/jpeg", "image/png", "image/webp"):
                        err(400, "invalid_image", "Only PNG, JPEG and WebP supported")
                    content = await uploaded.read(10 * 1024 * 1024 + 1)
                    png = await asyncio.to_thread(validate_png_bytes, content)
                else:
                    png = await asyncio.to_thread(download_public_image, value)
            except ImageInputError as ex:
                err(400, "invalid_image", f"Image {index+1}: {ex}")
            target = Path(folder) / f"input_{index}.png"
            target.write_bytes(png)
            paths.append(str(target))
        return await asyncio.to_thread(produce, body.model_dump(), token, "edits", paths)

# Serve Gradio studio on / while preserving the FastAPI image endpoints.
import gradio as gr
from .ui import create_ui, CSS
app = gr.mount_gradio_app(app, create_ui(), path="/", css=CSS)
