"""OpenAI-style asynchronous video endpoints (Wan image-to-video)."""
import asyncio
import json
import tempfile
from pathlib import Path
from fastapi import APIRouter, Request, Header, HTTPException
from fastapi.responses import FileResponse
from starlette.datastructures import UploadFile
from . import video_tasks
from .image_inputs import validate_png_bytes, ImageInputError, MAX_IMAGE_BYTES
from .wan2_2_i2v_v3 import MODEL_ID, build_kwargs

router = APIRouter(prefix="/v1/videos")

def fail(status, code, message):
    raise HTTPException(status, detail={"error": {"type": "invalid_request_error", "code": code, "message": message}})

@router.post("")
async def create_video(request: Request, authorization: str | None = Header(None),
                       x_hf_token: str | None = Header(None)):
    from .main import request_hf_token
    token = request_hf_token(authorization, x_hf_token)
    if not request.headers.get("content-type", "").startswith("multipart/form-data"):
        fail(415, "unsupported_media_type", "Use multipart/form-data")
    try:
        async with request.form(max_files=2, max_fields=12) as form:
            allowed = {"model", "prompt", "seconds", "seed", "extra_body", "input_reference", "last_image"}
            if set(form) - allowed or any(len(form.getlist(k)) != 1 for k in form):
                raise ValueError("Unknown or repeated field")
            if form.get("model") != MODEL_ID:
                fail(404, "model_not_found", "Unknown video model")
            prompt = form.get("prompt")
            if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 4000:
                raise ValueError("prompt must contain 1-4000 characters")
            payload = dict(model=MODEL_ID, prompt=prompt, seconds=float(form.get("seconds", "4")),
                           seed=int(form.get("seed", "-1")), extra_body=json.loads(form.get("extra_body", "{}")))
            images = []
            for key in ("input_reference", "last_image"):
                item = form.get(key)
                if item is None and key == "last_image":
                    continue
                if not isinstance(item, UploadFile):
                    raise ValueError(f"{key} must be an uploaded image")
                images.append(await asyncio.to_thread(validate_png_bytes, await item.read(MAX_IMAGE_BYTES + 1)))
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, data in enumerate(images):
                path = Path(tmp) / f"{i}.png"
                path.write_bytes(data)
                paths.append(str(path))
            build_kwargs(payload, paths)
        return await asyncio.to_thread(video_tasks.create, payload, images, token, request.headers.get("x-ip-token"))
    except (ValueError, TypeError, ImageInputError):
        fail(400, "invalid_parameters", "Invalid video parameters or image")
    except OverflowError:
        fail(429, "queue_full", "Video queue is full")
    except (OSError, RuntimeError):
        fail(503, "storage_unavailable", "Video storage unavailable")

@router.get("/{video_id}")
def get_video(video_id: str):
    try:
        return video_tasks.get(video_id)
    except KeyError:
        fail(404, "video_not_found", "Video not found")
    except (OSError, RuntimeError):
        fail(503, "storage_unavailable", "Video storage unavailable")

@router.get("/{video_id}/content")
def content(video_id: str):
    record = get_video(video_id)
    if record["status"] != "completed":
        fail(409, "video_not_ready", "Video is not completed")
    path = video_tasks.folder(video_id) / "output.mp4"
    if not path.is_file():
        fail(404, "video_not_found", "Video file missing")
    return FileResponse(path, media_type="video/mp4", filename="video.mp4",
                        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})

@router.delete("/{video_id}")
def delete_video(video_id: str):
    try:
        return video_tasks.delete(video_id)
    except KeyError:
        fail(404, "video_not_found", "Video not found")
    except ValueError:
        fail(409, "video_active", "Cannot delete an active video")
    except (OSError, RuntimeError):
        fail(503, "storage_unavailable", "Video storage unavailable")
