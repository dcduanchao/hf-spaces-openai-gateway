"""Forward the signed Hugging Face identity header between Spaces.

HF's edge injects X-IP-Token on incoming Space requests. A Gradio Client
created inside our adapter must explicitly forward this header when it is
not created directly inside a Gradio event callback. See:
https://www.gradio.app/docs/python-client/using-zero-gpu-spaces

Never persist or log the header value. Authorization remains caller-owned.
"""
from __future__ import annotations

import re
import logging

logger = logging.getLogger("uvicorn.error")

# Keep an upper bound and reject embedded control characters before forwarding.
SAFE_HEADER = re.compile(r"^[\x21-\x7e]{1,2048}$")


def token_tail(token: str | None) -> str:
    """Diagnostic display only: never log complete HF credentials."""
    if not token:
        return "none"
    return "****" + token[-4:]


def forward_headers(ip_token: str | None) -> dict[str, str]:
    if not isinstance(ip_token, str) or not SAFE_HEADER.fullmatch(ip_token):
        return {}
    return {"x-ip-token": ip_token}


def client_kwargs(token: str | None, ip_token: str | None) -> dict:
    """Use caller-owned HF token OR the HF edge identity, never competing identities.

    An incognito browser may have an anonymous X-IP-Token while the user
    explicitly entered a Pro HF Token in the studio. Passing both can make
    the upstream quota engine attribute the call to the anonymous identity.
    API calls carrying only X-HF-Token already work, so mirror that behavior.
    """
    kwargs = {"token": token or False, "verbose": False}
    # Explicit per-request HF Token takes precedence over a browser edge token.
    # Forward the edge token only when no HF Token was supplied.
    if not token:
        headers = forward_headers(ip_token)
        if headers:
            kwargs["headers"] = headers
    logger.info("[hf-auth] upstream_client hf_token=%s auth_mode=%s",
                token_tail(token), "hf_token" if token else ("edge_identity" if "headers" in kwargs else "anonymous"))
    return kwargs
