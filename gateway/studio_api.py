"""Internal loopback HTTP adapter: the Gradio Studio uses the public OpenAI API.

No separate Gradio -> ZeroGPU path. The API performs token propagation, validation,
request history and image persistence. Loopback only, never user-configured URLs.
"""
from __future__ import annotations

import base64
import binascii
import json
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:7860"
TIMEOUT_SECONDS = 360.0
MAX_IMAGE_BYTES = 15 * 1024 * 1024


class StudioAPIError(Exception):
    def __init__(self, code: str, status: int):
        self.code = code
        self.status = status
        super().__init__(code)


def api_generate(payload: dict, token: str | None, *, image_paths=None) -> list[bytes]:
    """Call the same API that works in Postman, using b64_json for local output.

    Never log, persist, include in URLs or mirror tokens into body fields.
    Do not forward browser X-IP-Token when the user provides a personal HF Token.
    """
    headers = {"X-HF-Token": token} if token else {}
    body = {**payload, "response_format": "b64_json"}
    try:
        with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT_SECONDS, trust_env=False) as client:
            if image_paths:
                fields = {k: str(v) for k, v in body.items() if k != "extra_body"}
                fields["extra_body"] = json.dumps(body.get("extra_body") or {}, ensure_ascii=False)
                # Order is meaningful: reference 1/2/3 corresponds to prompt.
                from contextlib import ExitStack
                with ExitStack() as stack:
                    files = [
                        ("image", (Path(path).name, stack.enter_context(open(path, "rb")), "image/png"))
                        for path in image_paths
                    ]
                    response = client.post("/v1/images/edits", data=fields, files=files, headers=headers)
            else:
                response = client.post("/v1/images/generations", json=body, headers=headers)
    except (httpx.RequestError, OSError) as exc:
        raise StudioAPIError("api_connection_failed", 502) from exc

    if response.status_code != 200:
        # Do not put a potentially untrusted body (or token) into UI errors.
        try:
            code = response.json().get("error", {}).get("code") or "upstream_failed"
        except (ValueError, AttributeError):
            code = "upstream_failed"
        if not isinstance(code, str) or len(code) > 80:
            code = "upstream_failed"
        raise StudioAPIError(code, response.status_code)

    try:
        images = []
        for item in response.json()["data"]:
            binary = base64.b64decode(item["b64_json"], validate=True)
            if not binary or len(binary) > MAX_IMAGE_BYTES:
                raise ValueError("invalid image size")
            images.append(binary)
        if not images:
            raise ValueError("empty image response")
        return images
    except (KeyError, TypeError, ValueError, binascii.Error) as exc:
        raise StudioAPIError("invalid_api_image_response", 502) from exc
