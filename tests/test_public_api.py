"""Regression tests for public model listings and caller-owned HF tokens."""
import os
import tempfile
from gateway import history
from pathlib import Path
from urllib.parse import urlsplit
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from gateway.main import app, request_hf_token, upstream_error_message


class PublicApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        (root / ".history_bucket_ready").write_text("test")
        bucket_patch = patch.object(history, "ROOT", root)
        bucket_patch.start()
        self.addCleanup(bucket_patch.stop)
        self.client = TestClient(app)

    def test_hf_status_anonymous(self):
        response = self.client.get("/v1/hf/status")
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["status"], "anonymous")
        self.assertIsNone(data["remaining_seconds"])
        self.assertIn("checked_at", data)

    def test_hf_status_token_scoped(self):
        fake = {"authenticated": True, "status": "valid", "username": "demo",
                "remaining_seconds": 100, "base_seconds": 200, "used_seconds": 100,
                "resets_at": None, "checked_at": "2026-11-23T00:00:00Z", "message": "ok"}
        with patch("gateway.main.quota_status", return_value=fake) as quota:
            response = self.client.get("/v1/hf/status", headers={"X-HF-Token": "hf_CALLER123"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), fake)
        quota.assert_called_once_with("hf_CALLER123")

    def test_hf_status_invalid_token_is_401(self):
        with patch("gateway.main.quota_status", side_effect=PermissionError("internal hf_SECRETTOKEN")):
            response = self.client.get("/v1/hf/status", headers={"X-HF-Token": "hf_CALLER123"})
        self.assertEqual(response.status_code, 401)
        self.assertNotIn("hf_SECRETTOKEN", response.text)

    def test_request_duration_headers(self):
        response = self.client.get("/v1/models")
        self.assertIn("X-Request-ID", response.headers)
        self.assertGreaterEqual(float(response.headers["X-Request-Duration-Ms"]), 0)

    def test_models_public_without_secrets(self):
        with patch.dict(os.environ, {}, clear=True):
            response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        ids = {x["id"] for x in response.json()["data"]}
        self.assertIn("qwen-image-2.1", ids)
        self.assertIn("z-image-turbo", ids)
        self.assertEqual(set(response.json()), {"data"})
        expected = {"id", "object", "created", "owned_by", "type", "display_name"}
        for item in response.json()["data"]:
            self.assertEqual(set(item), expected)
            self.assertEqual(item["object"], "model")
            self.assertEqual(item["type"], "model")
            self.assertIsInstance(item["created"], int)
        z = next(item for item in response.json()["data"] if item["id"] == "z-image-turbo")
        self.assertEqual(z["display_name"], "Z-Image-Turbo")
        self.assertEqual(z["owned_by"], "mrfakename")

    def test_models_detail_preserves_previous_format(self):
        with patch.dict(os.environ, {}, clear=True):
            response = self.client.get("/v1/models/detail")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["object"], "list")
        model = next(item for item in response.json()["data"] if item["id"] == "qwen-image-2.1")
        self.assertEqual(model["owned_by"], "duan0320/Qwen-Image-2.1")
        self.assertIn("sizes", model)
        self.assertIn("modes", model)
        self.assertIn("quality", model["extra_parameters"])

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

    def test_default_url_response_and_file_download(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("gateway.temp_images.IMAGE_DIR", Path(folder)):
                with patch("gateway.main.invoke", return_value=b"fake-png-bytes"):
                    response = self.client.post("/v1/images/generations", json={
                        "model": "z-image-turbo", "prompt": "cat", "size": "1024x1024"
                    })
                self.assertEqual(response.status_code, 200, response.text)
                self.assertIn("url", response.json()["data"][0])
                self.assertNotIn("b64_json", response.json()["data"][0])
                image_path = urlsplit(response.json()["data"][0]["url"]).path
                downloaded = self.client.get(image_path)
                self.assertEqual(downloaded.status_code, 200, downloaded.text)
                self.assertEqual(downloaded.content, b"fake-png-bytes")
                self.assertTrue(downloaded.headers["content-type"].startswith("image/png"))

    def test_expired_image_returns_404(self):
        from gateway import temp_images
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(temp_images, "IMAGE_DIR", Path(folder)):
                image_id = temp_images.save_temp_image(b"fake-image")
                old = __import__("time").time() - temp_images.IMAGE_TTL_SECONDS - 10
                os.utime(Path(folder) / image_id, (old, old))
                result = self.client.get("/v1/files/" + image_id)
                self.assertEqual(result.status_code, 404)
                self.assertEqual(result.json()["error"]["code"], "image_not_found")

    def test_explicit_base64_still_supported(self):
        with patch("gateway.main.invoke", return_value=b"abc"):
            response = self.client.post("/v1/images/generations", json={
                "model": "z-image-turbo", "prompt": "cat",
                "response_format": "b64_json"
            })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["data"][0]["b64_json"], "YWJj")

    def test_generation_can_use_per_request_token(self):
        with patch("gateway.main.invoke", return_value=b"abc") as fake:
            response = self.client.post("/v1/images/generations", headers={"X-HF-Token": "hf_TESTTOKEN"},
                json={"model": "z-image-turbo", "prompt": "cat", "size": "1024x1024"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(fake.call_args.args[2], "hf_TESTTOKEN")


if __name__ == "__main__":
    unittest.main()
