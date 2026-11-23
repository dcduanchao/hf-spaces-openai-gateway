"""Opt-in reference selection from the external AI Pose Gallery Chinese edition.

Metadata stays in memory; image assets remain on the original public GitHub CDN.
Third-party images are NOT MIT licensed. Respect their non-commercial usage terms.
"""
from __future__ import annotations

import re
import time
from threading import Lock
from urllib.parse import quote

import httpx

SOURCE = "https://github.com/anhao/ai-pose-gallery-zh"
DATA_URL = "https://raw.githubusercontent.com/anhao/ai-pose-gallery-zh/main/data/pose-zh.json"
ASSET_BASE = "https://raw.githubusercontent.com/anhao/ai-pose-gallery-zh/main/"
TTL_SECONDS = 3600
MAX_RESULTS = 48
_ALLOWED_PATH = re.compile(r"^img/[A-Za-z0-9_./-]+\.(?:png|webp|jpe?g)$", re.I)
_cache: list[dict] = []
_expires_at = 0.0
_lock = Lock()


def safe_pose_path(path):
    if not isinstance(path, str) or not _ALLOWED_PATH.fullmatch(path):
        raise ValueError("invalid pose image path")
    if ".." in path.split("/"):
        raise ValueError("invalid pose image path")
    return path


def public_asset_url(path):
    return ASSET_BASE + quote(safe_pose_path(path), safe="/")


def load_pose_catalog():
    global _cache, _expires_at
    with _lock:
        if _cache and time.monotonic() < _expires_at:
            return _cache
        response = httpx.get(DATA_URL, timeout=15, follow_redirects=False)
        response.raise_for_status()
        rows = response.json()
        if not isinstance(rows, list):
            raise ValueError("pose catalog is not a list")
        result = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            # The upstream gallery explicitly marks these as sensitive. Keep
            # the integrated picker SFW-only, without an unlock switch.
            section = str(row.get("体系", ""))
            category = str(row.get("分类", ""))
            name = str(row.get("姿势名称", ""))
            if ("性感" in section or "双人" in section or
                    any(term in (section + category + name).lower()
                        for term in ("nsfw", "r18", "色情", "性爱", "内衣", "胸部"))):
                continue
            try:
                asset = public_asset_url(row.get("原图"))
            except ValueError:
                continue
            item = {
                "id": str(row.get("id", ""))[:90],
                "category": category[:50],
                "title": name[:90],
                "angle": str(row.get("角度", ""))[:40],
                "system": section[:50],
                "url": asset,
                "keywords": str(row.get("检索", ""))[:280],
            }
            if item["id"] and item["title"]:
                result.append(item)
        if not result:
            raise ValueError("pose catalog contains no usable entries")
        _cache = result
        _expires_at = time.monotonic() + TTL_SECONDS
        return result


def search_poses(keyword="", category="全部", limit=MAX_RESULTS):
    term = (keyword or "").strip().casefold()[:80]
    result = []
    for item in load_pose_catalog():
        if category not in ("", "全部") and item["category"] != category:
            continue
        text = " ".join(str(item[k]) for k in ("id", "category", "title", "angle", "system", "keywords")).casefold()
        if term and term not in text:
            continue
        result.append(item)
        if len(result) >= min(max(1, limit), MAX_RESULTS):
            break
    return result


def categories():
    return ["全部"] + sorted({item["category"] for item in load_pose_catalog() if item["category"]})


def gallery_items(items):
    return [(x["url"], f'{x["title"]} · {x["angle"]}') for x in items]


def add_selected_pose(items, selected, index):
    if isinstance(index, tuple):
        index = index[0]
    if not isinstance(index, int) or index < 0 or index >= len(items):
        return selected or [], "请先搜索并选择一张姿势图"
    urls = list(selected or [])
    new = items[index]["url"]
    if new not in urls:
        if len(urls) >= 4:
            return urls, "最多只能选择 4 张姿势参考图（含上传图片合计最多 4 张）"
        urls.append(new)
    return urls, f"已选择 {len(urls)} 张姿势参考图；生成时还会计算上传的图片数量。"
