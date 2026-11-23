"""Resolve public HTTPS reference images without exposing the Space to SSRF.

DNS is validated for public destinations and the TCP connection is pinned to
the validated IP. TLS verifies the original hostname. Redirects are rechecked.
"""
from __future__ import annotations

import http.client
import base64
import binascii
import re
import ipaddress
import io
import socket
import ssl
from urllib.parse import urljoin, urlsplit

from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_777_216
MAX_REDIRECTS = 2


class ImageInputError(ValueError):
    pass


DATA_URL_PATTERN = re.compile(r"^data:image/(png|jpeg|webp);base64,([A-Za-z0-9+/]*={0,2})$", re.I)
MAX_DATA_URL_LENGTH = ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 128


def decode_image_data_url(value: str) -> bytes:
    """Decode an inline PNG/JPEG/WebP safely without leaking its contents.

    Enforce encoded-length bound before decoding and normalize to PNG through
    the same image validator used by uploads and public HTTPS URLs.
    """
    if not isinstance(value, str) or not value.startswith("data:"):
        raise ImageInputError("图片必须为支持的 Base64 Data URL")
    if len(value) > MAX_DATA_URL_LENGTH:
        raise ImageInputError("Base64 图片超过 10 MB")
    matched = DATA_URL_PATTERN.fullmatch(value)
    if not matched or not matched.group(2):
        raise ImageInputError("仅支持 data:image/png|jpeg|webp;base64,... 图片")
    try:
        decoded = base64.b64decode(matched.group(2), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ImageInputError("Base64 图片编码无效") from exc
    return validate_png_bytes(decoded)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, hostname: str, ip_address: str):
        super().__init__(hostname, port=443, timeout=15, context=ssl.create_default_context())
        self._pinned_ip = ip_address

    def connect(self):
        connection = socket.create_connection((self._pinned_ip, 443), timeout=self.timeout)
        self.sock = self._context.wrap_socket(connection, server_hostname=self.host)


def _validate_url(url: str) -> tuple[str, str, str]:
    if not isinstance(url, str) or not url or len(url) > 2048:
        raise ImageInputError("图片 URL 无效或过长")
    try:
        parsed = urlsplit(url)
        hostname = (parsed.hostname or "").rstrip(".").lower()
        port = parsed.port
    except ValueError as exc:
        raise ImageInputError("图片 URL 无效") from exc
    if (parsed.scheme != "https" or not hostname or parsed.username is not None
            or parsed.password is not None or port not in (None, 443)
            or hostname.endswith((".local", ".internal", ".localhost"))
            or hostname in ("localhost",)):
        raise ImageInputError("仅支持公开的 HTTPS 图片地址（443 端口、不可带账号密码）")
    try:
        hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ImageInputError("图片域名无效") from exc
    try:
        records = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
        ips = sorted({record[4][0] for record in records})
    except (socket.gaierror, OSError) as exc:
        raise ImageInputError("无法解析图片域名") from exc
    if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
        raise ImageInputError("图片 URL 不能指向内网、保留地址或本地服务")
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    return hostname, ips[0], path


def validate_png_bytes(content: bytes) -> bytes:
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise ImageInputError("单张图片不能为空或超过 10 MB")
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format not in ("PNG", "JPEG", "WEBP"):
                raise ImageInputError("仅支持 PNG、JPEG、WebP 图片")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ImageInputError("图片分辨率超过 16777216 像素")
            image.load()
            out = io.BytesIO()
            image.convert("RGB").save(out, format="PNG")
            converted = out.getvalue()
            if len(converted) > MAX_IMAGE_BYTES:
                raise ImageInputError("图片转换为 PNG 后超过 10 MB")
            return converted
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageInputError("文件不是有效的 PNG、JPEG 或 WebP 图片") from exc


def download_public_image(url: str) -> bytes:
    """Return normalized PNG bytes. Never forward the caller's HF token."""
    current = url
    for redirects in range(MAX_REDIRECTS + 1):
        hostname, ip, path = _validate_url(current)
        conn = _PinnedHTTPSConnection(hostname, ip)
        try:
            conn.request("GET", path, headers={
                "Host": hostname, "Accept": "image/png,image/jpeg,image/webp",
                "User-Agent": "HF-Image-Gateway/1.0",
            })
            response = conn.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader("Location")
                if not location or redirects >= MAX_REDIRECTS:
                    raise ImageInputError("图片 URL 重定向过多或无效")
                current = urljoin(current, location)
                continue
            if response.status != 200:
                raise ImageInputError(f"图片 URL 返回 HTTP {response.status}")
            content_length = response.getheader("Content-Length")
            if content_length:
                try:
                    if int(content_length) > MAX_IMAGE_BYTES:
                        raise ImageInputError("远程图片超过 10 MB")
                except ValueError as exc:
                    raise ImageInputError("远程图片长度无效") from exc
            content_type = (response.getheader("Content-Type") or "").split(";", 1)[0].lower().strip()
            if content_type not in ("image/png", "image/jpeg", "image/webp", "application/octet-stream"):
                raise ImageInputError("远程资源不是支持的图片类型")
            chunks = []
            size = 0
            while True:
                data = response.read(min(65536, MAX_IMAGE_BYTES + 1 - size))
                if not data:
                    break
                size += len(data)
                if size > MAX_IMAGE_BYTES:
                    raise ImageInputError("远程图片超过 10 MB")
                chunks.append(data)
            return validate_png_bytes(b"".join(chunks))
        except (OSError, ssl.SSLError, TimeoutError, http.client.HTTPException) as exc:
            raise ImageInputError("远程图片下载失败或超时") from exc
        finally:
            conn.close()
    raise ImageInputError("图片 URL 重定向过多")
