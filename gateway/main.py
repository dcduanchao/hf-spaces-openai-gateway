from __future__ import annotations
import base64
import os
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field, ConfigDict
from .models import load_profiles
from .adapter import invoke, parse_size

app = FastAPI(title="HF Spaces OpenAI Image Gateway", version="0.1.0")

class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    prompt: str = Field(min_length=1, max_length=4000)
    size: str | None = None
    n: int = Field(default=1, ge=1, le=4)
    response_format: str = "b64_json"
    seed: int = -1
    extra_body: dict[str, Any] = Field(default_factory=dict)


def err(status, code, message):
    raise HTTPException(status_code=status, detail={"error": {"code": code, "message": message, "type": "invalid_request_error"}})

@app.exception_handler(HTTPException)
async def error_handler(_, exc):
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"error": {"type": "invalid_request_error", "code": "bad_request", "message": str(exc.detail)}})


def authorize(authorization: str | None):
    gateway_key = os.getenv("GATEWAY_API_KEY", "").strip()
    if not gateway_key:
        err(503, "gateway_unconfigured", "Set GATEWAY_API_KEY in Space Secrets")
    provided = (authorization or "").removeprefix("Bearer ").strip()
    if not secrets.compare_digest(provided, gateway_key):
        err(401, "invalid_api_key", "Invalid gateway API key")
    return os.getenv("UPSTREAM_HF_TOKEN", "").strip() or None


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
    if payload.get("response_format", "b64_json") != "b64_json":
        err(400, "unsupported_format", "Only b64_json is supported in this stateless version")
    try:
        parse_size(payload.get("size") or p.default_size, p)
        data = []
        for i in range(payload.get("n", 1)):
            item = dict(payload)
            if item.get("seed", -1) >= 0:
                item["seed"] += i
            png = invoke(p, item, token, image_paths=paths)
            data.append({"b64_json": base64.b64encode(png).decode("ascii")})
    except ValueError as ex:
        err(400, "invalid_parameters", str(ex))
    except Exception:
        err(502, "upstream_failed", "Upstream Space failed; check Space status, permissions and API schema")
    return {"created": int(time.time()), "data": data, "model": p.id}

@app.get("/health")
def health():
    return {"ok": True, "configured": bool(os.getenv("GATEWAY_API_KEY")), "models": len(load_profiles())}

@app.get("/v1/models")
def models(authorization: str | None = Header(default=None)):
    authorize(authorization)
    return {"object": "list", "data": [{"id": p.id, "object": "model", "owned_by": p.space_id, "description": p.description, "modes": p.modes, "default_size": p.default_size, "sizes": p.allowed_sizes, "extra_parameters": {k: v.model_dump() for k, v in p.extras.items()}} for p in load_profiles().values()]}

@app.post("/v1/images/generations")
def generate(body: GenerateRequest, authorization: str | None = Header(default=None)):
    token = authorize(authorization)
    return produce(body.model_dump(exclude_none=True, exclude_unset=False), token)

@app.post("/v1/images/edits")
async def edit(model: str = Form(...), prompt: str = Form(...), image: list[UploadFile] = File(...), size: str | None = Form(None), n: int = Form(1), response_format: str = Form("b64_json"), seed: int = Form(-1), extra_body: str = Form("{}"), authorization: str | None = Header(default=None)):
    import json
    token = authorize(authorization)
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
