"""Single-process bounded video worker with persistent task records."""
import json
import logging
import re
import secrets
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore, Lock
from . import history
from .wan2_2_i2v_v3 import run, VideoUpstreamError

logger = logging.getLogger("uvicorn.error")

POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video")
SLOTS = BoundedSemaphore(4)
LOCK = Lock()
ACTIVE = set()

def folder(video_id):
    if not re.fullmatch(r"video_[a-f0-9]{32}", video_id):
        raise KeyError(video_id)
    return history._ready().parent / "gateway-videos" / video_id

def write(path, record):
    history._write_json(path / "record.json", record)

def read(video_id):
    path = folder(video_id)
    try:
        record = json.loads((path / "record.json").read_text())
    except FileNotFoundError:
        raise KeyError(video_id) from None
    if record["status"] in ("queued", "in_progress") and video_id not in ACTIVE:
        record.update(status="failed", error={"code": "worker_interrupted", "message": "Worker restarted; submit a new task."})
        write(path, record)
    return record

def worker(path, record, payload, token, ip_token, paths):
    try:
        record.update(status="in_progress")
        write(path, record)
        raw, seed = run(payload, token, paths, ip_token=ip_token)
        (path / "output.mp4").write_bytes(raw)
        record.update(status="completed", progress=100, completed_at=int(time.time()), seed=seed)
        write(path, record)
    except Exception as exc:
        phase = exc.phase if isinstance(exc, VideoUpstreamError) else "worker"
        code = exc.code if isinstance(exc, VideoUpstreamError) else "video_worker_failed"
        message = exc.public_message if isinstance(exc, VideoUpstreamError) else "Video worker failed; check storage and service logs."
        logger.error("[video-task] id=%s phase=%s code=%s error_type=%s", record["id"], phase, code, type(exc).__name__)
        record.update(status="failed", error={"code": code, "message": message, "phase": phase})
        write(path, record)
    finally:
        with LOCK:
            ACTIVE.discard(record["id"])
        SLOTS.release()

def create(payload, images, token, ip_token):
    if not SLOTS.acquire(blocking=False):
        raise OverflowError("Video queue is full")
    path = None
    video_id = "video_" + secrets.token_hex(16)
    try:
        path = folder(video_id)
        path.mkdir(parents=True, exist_ok=False)
        paths = []
        for i, image in enumerate(images):
            target = path / f"input-{i}.png"
            target.write_bytes(image)
            paths.append(str(target))
        record = dict(id=video_id, object="video", model=payload["model"],
                      status="queued", progress=0, created_at=int(time.time()),
                      seconds=str(payload["seconds"]), error=None)
        write(path, record)
        history._write_json(path / "parameters.json", history._safe_value(payload))
        with LOCK:
            ACTIVE.add(video_id)
        response = dict(record)
        POOL.submit(worker, path, record, payload, token, ip_token, paths)
        return response
    except Exception:
        with LOCK:
            ACTIVE.discard(video_id)
        if path is not None and path.exists():
            shutil.rmtree(path)
        SLOTS.release()
        raise

def get(video_id):
    with LOCK:
        return read(video_id)

def delete(video_id):
    with LOCK:
        record = read(video_id)
        if record["status"] in ("queued", "in_progress"):
            raise ValueError("Cannot delete an active video")
        shutil.rmtree(folder(video_id))
    return {"id": video_id, "object": "video.deleted", "deleted": True}
