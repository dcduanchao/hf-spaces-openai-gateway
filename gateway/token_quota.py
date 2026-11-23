"""Validate a visitor HF Token and report official ZeroGPU quota.

Only the explicitly supplied visitor token is used. No token is persisted or logged.
huggingface_hub<2 is required by this app, so the quota REST endpoint is used.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx
from huggingface_hub import HfApi

TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]+")
QUOTA_ENDPOINT = "https://huggingface.co/api/spaces/zero-gpu/quota"
TOKYO = ZoneInfo("Asia/Tokyo")


def format_seconds(value: float) -> str:
    total = max(0, int(value))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h} 小时 {m} 分 {s} 秒" if h else f"{m} 分 {s} 秒"


def format_date(value: str | None) -> str:
    if not value:
        return "未提供（无法确定何时重置）"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(TOKYO).strftime("%Y-%m-%d %H:%M:%S JST") + "（东京时间）"
    except (ValueError, TypeError):
        return "上游返回了无法解析的重置时间"


def parse_quota(payload: object) -> tuple[float, float, str | None]:
    if not isinstance(payload, dict):
        raise ValueError("ZeroGPU 额度响应不是 JSON 对象")
    try:
        remaining = float(payload["current"])
        base = float(payload["base"])
        reset = payload.get("resetsAt")
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("ZeroGPU 额度响应缺少必要字段") from exc
    if not all(map(math.isfinite, (remaining, base))) or remaining < 0 or base < 0:
        raise ValueError("ZeroGPU 额度数据无效")
    return remaining, base, reset


def token_quota_info(token: str) -> str:
    token = (token or "").strip()
    if not token:
        return "匿名模式：未提供 HF Token，无法查询个人 ZeroGPU 额度。"
    if not TOKEN_PATTERN.fullmatch(token):
        return "HF Token 格式无效，请检查是否为 hf_ 开头的访问令牌。"

    try:
        identity = HfApi(token=token).whoami()
        username = identity.get("name") or "Hugging Face 用户"
    except Exception:
        return "HF Token 验证失败，请检查是否过期、已撤销或权限不足。"

    checked_at = datetime.now(timezone.utc).astimezone(TOKYO).strftime("%Y-%m-%d %H:%M:%S JST")
    # Avoid injecting an unexpected HF account name as Markdown.
    username = re.sub(r"[^A-Za-z0-9_.-]", "", str(username))[:80] or "HF 用户"
    try:
        response = httpx.get(
            QUOTA_ENDPOINT,
            headers={"Authorization": f"Bearer {token}"},
            timeout=12.0,
            follow_redirects=False,
        )
        response.raise_for_status()
        remaining, base, reset = parse_quota(response.json())
    except Exception:
        return (
            f"**Token 有效：{username}**\n\n"
            f"查询时间：{checked_at}\n\n"
            "ZeroGPU 个人配额暂时无法获取。请检查 Fine-grained Token 是否具有 "
            "**Billing → Read billing usage and payment method status** 权限，"
            "或稍后重试 Hugging Face 配额服务。"
        )

    used = max(0, base - remaining)
    return (
        f"**Token 有效：{username}**\n\n"
        f"**ZeroGPU 剩余可用时长：{format_seconds(remaining)}**\n\n"
        f"基础额度：{format_seconds(base)}\n\n"
        f"已使用（按基础额度估算）：{format_seconds(used)}\n\n"
        f"**下次额度重置时间：{format_date(reset)}**\n\n"
        f"查询时间：{checked_at}\n\n"
        "说明：以上是 Hugging Face 当前返回的账号级 GPU 额度，"
        "不代表每次生成任务允许持续运行的时间；如有额外额度，已用时长仅供参考。"
    )


def quota_status(token: str | None) -> dict:
    """Structured account-level ZeroGPU quota, using only a caller-provided token.

    All timestamps use ISO 8601 UTC; never return or log the credential.
    """
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if not token:
        return {
            "authenticated": False, "status": "anonymous", "username": None,
            "remaining_seconds": None, "base_seconds": None, "used_seconds": None,
            "resets_at": None, "checked_at": now,
            "message": "Anonymous mode has no account quota information. Provide your HF Token using X-HF-Token.",
        }
    if not TOKEN_PATTERN.fullmatch(token):
        raise ValueError("Invalid HF Token format")
    try:
        identity = HfApi(token=token).whoami()
    except Exception as exc:
        raise PermissionError("HF Token validation failed; check token permissions or expiration") from exc
    username = re.sub(r"[^A-Za-z0-9_.-]", "", str(identity.get("name") or ""))[:80] or None
    try:
        response = httpx.get(
            QUOTA_ENDPOINT,
            headers={"Authorization": f"Bearer {token}"},
            timeout=12.0,
            follow_redirects=False,
        )
        response.raise_for_status()
        remaining, base, reset = parse_quota(response.json())
    except Exception as exc:
        raise ConnectionError(
            "HF ZeroGPU quota unavailable; check Billing read permission or retry later"
        ) from exc
    reset_iso = None
    if isinstance(reset, str) and reset:
        try:
            dt = datetime.fromisoformat(reset.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            reset_iso = dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except ValueError:
            pass
    return {
        "authenticated": True, "status": "valid", "username": username,
        "remaining_seconds": remaining, "base_seconds": base,
        "used_seconds": max(0.0, base - remaining),
        "resets_at": reset_iso, "checked_at": now,
        "message": "Account-level ZeroGPU quota; not per-request GPU execution time.",
    }
