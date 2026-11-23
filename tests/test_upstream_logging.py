"""Verify upstream log traces without sending tokens, full prompts, or binary data."""
import io
import logging
import os
import unittest
from unittest.mock import patch
from gateway.upstream_logging import predict_logged, _safe


class FakeClient:
    def predict(self, *, api_name, **kwargs):
        return [{"path": "/tmp/generated/secret-name.png"}, 42, "rewritten secret prompt"]


class UpstreamLoggingTests(unittest.TestCase):
    def test_request_and_response(self):
        with self.assertLogs("uvicorn.error", level="INFO") as captured:
            value = predict_logged(FakeClient(), space_id="owner/example",
                api_name="/prepare_request",
                kwargs={"original_prompt": "a private portrait",
                        "input_images": [{"image": {"path": "/tmp/photo.jpg"}}],
                        "seed": 42, "hf_token": "hf_abc123"})
        self.assertEqual(value[1], 42)
        logs = "\n".join(captured.output)
        self.assertIn("endpoint=/prepare_request", logs)
        self.assertIn("params=", logs)
        self.assertIn("result=", logs)
        self.assertIn("seed", logs)
        self.assertNotIn("hf_abc123", logs)
        self.assertNotIn("a private portrait", logs)
        self.assertIn("secret-name.png", logs)

    def test_prompt_can_be_opted_in(self):
        with patch.dict(os.environ, {"LOG_UPSTREAM_PROMPTS": "true"}):
            self.assertEqual(_safe("a small cat", "original_prompt"), "a small cat")

    def test_binary_and_url_filtered(self):
        self.assertEqual(_safe(b"some-private-binary"), {"binary_bytes": 19})
        self.assertEqual(_safe({"token": "hf_topsecret"})["token"], "[REDACTED]")
        self.assertNotIn("signature=abc", str(_safe("https://example.com/file?signature=abc")))


if __name__ == "__main__":
    unittest.main()
