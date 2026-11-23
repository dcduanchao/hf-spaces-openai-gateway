"""Explicit two-step reference confirmation for the Gradio image editor.

Gallery clicks are previews only. On confirmation, download one validated
reference into a temporary PNG and append it to the same gr.File input used
for manual uploads. The generation path processes files exactly once.
"""
from __future__ import annotations

import os
import secrets
import tempfile
import time
from pathlib import Path

from .image_inputs import download_public_image
from .reference_gallery import load_references

CACHE_DIR = Path(tempfile.gettempdir()) / "hf_gateway_confirmed_references"
CACHE_TTL = 24 * 60 * 60
MAX_INPUT_IMAGES = 4


def normalize_paths(files):
    if not files:
        return []
    return [str(item) for item in files]


def preview_candidate(results, index):
    if isinstance(index, tuple):
        index = index[0]
    if not isinstance(index, int) or index < 0 or index >= len(results or []):
        return None, None, "请点击一张图库图片进行预览。"
    row = results[index]
    candidate = {key: row.get(key, "") for key in ("id", "kind", "title", "url", "detail") }
    return candidate, row["url"], (
        f"正在预览：**{candidate['kind']} · {candidate['title']}**。"
        "这张图尚未加入上传图片框；确认后才会添加。"
    )


def confirm_reference(candidate, files, added):
    """Return (gr.File value, confirmed paths, hint).

    Do not rely on a browser-provided URL alone. Recheck the reference
    against the filtered catalog before fetching any bytes.
    """
    if not isinstance(candidate, dict) or candidate.get("kind") not in ("人像", "服装", "姿势"):
        raise ValueError("请先点击图库图片进行预览")
    kind = candidate["kind"]
    allowed = {item["url"]: item for item in load_references(kind)}
    if candidate.get("url") not in allowed:
        raise ValueError("图库参考图无效，请重新预览并选择")
    paths = normalize_paths(files)
    entries = [x for x in (added or []) if x.get("path") in paths]
    existing = next((x for x in entries if x["url"] == candidate["url"]), None)
    if existing:
        return paths, entries, "该图库图片已经在上传图片框中，无需重复添加。"
    if len(paths) >= MAX_INPUT_IMAGES:
        raise ValueError("上传图片框中最多放 4 张图片，请先删除不需要的图片")
    png = download_public_image(candidate["url"])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for file in CACHE_DIR.glob("ref-*.png"):
        try:
            if now - file.stat().st_mtime > CACHE_TTL:
                file.unlink(missing_ok=True)
        except OSError:
            continue
    dest = CACHE_DIR / ("ref-" + secrets.token_hex(16) + ".png")
    with dest.open("xb") as target:
        target.write(png)
    paths.append(str(dest))
    entries.append({"url": candidate["url"], "path": str(dest), "kind": kind,
                    "title": allowed[candidate["url"]].get("title", ""),
                    "detail": allowed[candidate["url"]].get("detail", "")})
    return paths, entries, (
        f"已添加 **{kind}** 参考图至上传图片框（共 {len(paths)}/{MAX_INPUT_IMAGES} 张）。"
        "可继续上传、从其他图库选图，或直接开始编辑。"
    )


def clear_gallery_files(files, added):
    """Remove confirmed gallery files without deleting manually uploaded files."""
    gallery_paths = {x["path"] for x in (added or [])}
    kept = [p for p in normalize_paths(files) if p not in gallery_paths]
    return kept, [], f"已移除图库添加的图片；保留 {len(kept)} 张手动上传图片。"
