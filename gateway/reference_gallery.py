"""Read-only, source-attributed references from AI Pose Gallery Chinese edition.

Metadata is retrieved on demand from upstream. Image files are NOT redistributed
or included in our repository. Original image terms and copyright still apply.
"""
from __future__ import annotations

import re
import threading
import time
from urllib.parse import quote

import httpx

from .pose_gallery import load_pose_catalog

SOURCE = "https://github.com/anhao/ai-pose-gallery-zh"
RAW_BASE = "https://raw.githubusercontent.com/anhao/ai-pose-gallery-zh/main/"
FILES = {
    "人像": "data/portrait-zh.json",
    "服装": "data/costume-zh.json",
}
_CACHE_TTL = 3600
_LOCK = threading.Lock()
_CACHE: dict[str, tuple[float, list[dict]]] = {}
_IMAGE_RE = re.compile(r"^img/[A-Za-z0-9_./-]+\.(?:png|webp|jpe?g)$", re.I)
MAX_RESULTS = 48


def _file_url(path):
    if not isinstance(path, str) or not _IMAGE_RE.fullmatch(path) or ".." in path.split("/"):
        raise ValueError("Invalid upstream image path")
    return RAW_BASE + quote(path, safe="/")


def _is_safe_costume(item):
    text = " ".join(str(item.get(k, "")) for k in ("分类", "服装名称", "关键词")).lower()
    blocked = ("内衣", "睡衣", "性感", "泳装", "比基尼", "lingerie",
               "sexy", "underwear", "bikini", "nsfw")
    return not any(term in text for term in blocked)


def _normalise(kind, item):
    if not isinstance(item, dict):
        return None
    if kind == "服装" and not _is_safe_costume(item):
        return None
    try:
        url = _file_url(item.get("原图"))
    except ValueError:
        return None
    if kind == "人像":
        category = str(item.get("分区") or "其他")
        name = str(item.get("标题") or item.get("发型名称") or "人像参考")
        detail = str(item.get("发长") or "")
    else:
        category = str(item.get("分类") or "其他")
        name = str(item.get("服装名称") or "服装参考")
        detail = str(item.get("关键词") or "")
    return {
        "id": str(item.get("id", ""))[:90], "kind": kind,
        "category": category[:60], "title": name[:120],
        "detail": detail[:250], "url": url,
        "keywords": str(item.get("检索") or "")[:450],
    }


def load_references(kind):
    if kind == "姿势":
        return [
            {"id": row["id"], "kind": kind, "category": row["category"],
             "title": row["title"], "detail": row["angle"],
             "url": row["url"], "keywords": row["keywords"]}
            for row in load_pose_catalog()
        ]
    if kind not in FILES:
        raise ValueError("Unsupported reference kind")
    with _LOCK:
        now = time.monotonic()
        cached = _CACHE.get(kind)
        if cached and cached[0] > now:
            return cached[1]
        response = httpx.get(RAW_BASE + FILES[kind], timeout=20, follow_redirects=False)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("Invalid reference metadata")
        items = []
        for record in payload:
            item = _normalise(kind, record)
            if item and item["id"]:
                items.append(item)
        if not items:
            raise ValueError("Empty reference catalog")
        _CACHE[kind] = (time.monotonic() + _CACHE_TTL, items)
        return items


def categories(kind):
    return ["全部"] + sorted({row["category"] for row in load_references(kind)})


def search_references(kind, keyword="", category="全部", limit=MAX_RESULTS):
    query = str(keyword or "").strip().casefold()[:80]
    matches = []
    for item in load_references(kind):
        if category not in ("", "全部") and item["category"] != category:
            continue
        searchable = " ".join(str(item[k]) for k in (
            "id", "title", "category", "detail", "keywords")).casefold()
        if query and query not in searchable:
            continue
        matches.append(item)
        if len(matches) >= min(MAX_RESULTS, max(1, int(limit))):
            break
    return matches


def gallery_items(rows):
    return [(item["url"], f'{item["kind"]} · {item["title"]}') for item in rows]


def choose_reference(results, selected, index):
    if isinstance(index, tuple):
        index = index[0]
    selected = list(selected or [])
    if not isinstance(index, int) or index < 0 or index >= len(results):
        return selected, "请点击图库中的参考图"
    item = results[index]
    if any(old["url"] == item["url"] for old in selected):
        return selected, f'已选择 {len(selected)} 张参考图（不重复添加）'
    if len(selected) >= 4:
        return selected, "最多可选择 4 张参考图（与手动上传合计最多 4 张）"
    selected.append({k: item[k] for k in ("id", "kind", "title", "url")})
    return selected, format_selection(selected)


def format_selection(items):
    if not items:
        return "当前未选择任何图库参考图。"
    return "已选 " + str(len(items)) + " 张：" + " / ".join(
        f'{item["kind"]}：{item["title"]}' for item in items)


def validate_selection(selected):
    """Validate server-side metadata against catalogs; never trust Gradio State alone."""
    selected = list(selected or [])
    if len(selected) > 4:
        raise ValueError("最多选择 4 张图库参考图")
    catalogs = {}
    for row in selected:
        if not isinstance(row, dict):
            raise ValueError("参考图选择无效")
        kind = row.get("kind")
        if kind not in ("人像", "服装", "姿势"):
            raise ValueError("参考图库类型无效")
        if kind not in catalogs:
            catalogs[kind] = {item["url"] for item in load_references(kind)}
        if row.get("url") not in catalogs[kind]:
            raise ValueError("参考图片不属于允许的公开图库")
    return selected
