"""Regression tests for public model listings and caller-owned HF tokens."""
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from gateway.main import app, request_hf_token, upstream_error_message


class PublicApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_models_public_without_secrets(self):
        with patch.dict(os.environ, {}, clear=True):
            response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        ids = {x["id"] for x in response.json()["data"]}
        self.assertIn("qwen-image-2.1", ids)
        self.assertIn("z-image-turbo", ids)

    def test_anonymous_request_does_not_use_server_secret(self):
        with patch.dict(os.environ, {"UPSTREAM_HF_TOKEN": "hf_OWNER_SECRET"}):
            self.assertIsNone(request_hf_token(None, None))
            self.assertIsNone(request_hf_token("Bearer anonymous", None))

    def test_caller_hf_header(self):
        self.assertEqual(request_hf_token(None, "hf_CALLER123"), "hf_CALLER123")
        self.assertEqual(request_hf_token("Bearer hf_CALLER123", None), "hf_CALLER123")

    def test_quota_error_is_explicit(self):
        status, code, message = upstream_error_message(RuntimeError("ZeroGPU quota exceeded"))
        self.assertEqual((status, code), (429, "upstream_quota_exceeded"))
        self.assertIn("HF Token", message)

    def test_unexpected_upstream_errors_redact_contents(self):
        status, code, message = upstream_error_message(RuntimeError("Backend hf_SUPERSECRET broken"))
        self.assertEqual(status, 502)
        self.assertNotIn("hf_SUPERSECRET", message)

    def test_generation_is_public_and_passes_no_owner_token(self):
        with patch.dict(os.environ, {"UPSTREAM_HF_TOKEN": "hf_OWNERSECRET"}, clear=True):
            with patch("gateway.main.invoke", return_value=b"abc") as fake:
                response = self.client.post("/v1/images/generations", json={
                    "model": "z-image-turbo", "prompt": "cat", "size": "1024x1024"
                })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNone(fake.call_args.args[2])

    def test_generation_can_use_per_request_token(self):
        with patch("gateway.main.invoke", return_value=b"abc") as fake:
            response = self.client.post("/v1/images/generations", headers={"X-HF-Token": "hf_TESTTOKEN"},
                json={"model": "z-image-turbo", "prompt": "cat", "size": "1024x1024"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(fake.call_args.args[2], "hf_TESTTOKEN")


if __name__ == "__main__":
    unittest.main()
