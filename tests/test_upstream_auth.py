"""Regression tests for HF identity propagation in cross-Space ZeroGPU calls."""
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from gateway.main import app
from gateway import history
from gateway.upstream_auth import client_kwargs, forward_headers
from gateway.models import load_profiles
from gateway import z_image, qwen_image
import tempfile
from pathlib import Path


class IdentityForwardingTests(unittest.TestCase):
    def test_auth_kwargs_use_both_credentials(self):
        result = client_kwargs("hf_PROUSER123", "signed-edge-token")
        self.assertEqual(result["token"], "hf_PROUSER123")
        self.assertNotIn("headers", result)
        self.assertEqual(client_kwargs(None, "signed-edge-token")["headers"], {"x-ip-token": "signed-edge-token"})
        self.assertEqual(client_kwargs(None, None), {"token": False, "verbose": False})
        self.assertEqual(forward_headers("bad\nheader"), {})
        self.assertEqual(forward_headers(""), {})

    def test_z_image_client_receives_edge_and_hf_tokens(self):
        profile = load_profiles()["z-image-turbo"]
        payload = {"model": "z-image-turbo", "prompt": "cat", "size": "1024x1024"}
        with patch("gateway.z_image.Client") as client:
            with patch("gateway.z_image.predict_logged", return_value=["some/file.png", 42]):
                with patch("gateway.z_image.result_bytes", return_value=b"png"):
                    result = z_image.run(profile, payload, "hf_PROUSER123", ip_token="signed-edge-token")
        self.assertEqual(result, b"png")
        kwargs = client.call_args.kwargs
        self.assertEqual(kwargs["token"], "hf_PROUSER123")
        self.assertNotIn("headers", kwargs)

    def test_qwen_client_receives_edge_and_hf_tokens(self):
        profile = load_profiles()["qwen-image-2.1"]
        payload = {"model": "qwen-image-2.1", "prompt": "cat", "size": "1024x1024"}
        with patch("gateway.qwen_image.Client") as client:
            with patch("gateway.qwen_image.predict_logged", side_effect=[["unused", 42, "cat"], ["file.png"]]):
                with patch("gateway.adapter.result_bytes", return_value=b"png"):
                    result = qwen_image.run(profile, payload, "hf_PROUSER123", ip_token="edge-token")
        self.assertEqual(result, b"png")
        self.assertEqual(client.call_args.kwargs["token"], "hf_PROUSER123")
        self.assertNotIn("headers", client.call_args.kwargs)

    def test_anonymous_client_forwards_edge_identity(self):
        profile = load_profiles()["z-image-turbo"]
        payload = {"model": "z-image-turbo", "prompt": "cat", "size": "1024x1024"}
        with patch("gateway.z_image.Client") as client:
            with patch("gateway.z_image.predict_logged", return_value=["some/file.png", 42]):
                with patch("gateway.z_image.result_bytes", return_value=b"png"):
                    z_image.run(profile, payload, None, ip_token="browser-edge-token")
        self.assertEqual(client.call_args.kwargs["token"], False)
        self.assertEqual(client.call_args.kwargs["headers"], {"x-ip-token": "browser-edge-token"})

    def test_pro_browser_ignores_anonymous_identity(self):
        self.assertEqual(
            client_kwargs("hf_USERPRO123", "anonymous-browser-edge"),
            client_kwargs("hf_USERPRO123", None),
        )
        self.assertEqual(client_kwargs("hf_USERPRO123", "anonymous-browser-edge")["token"],
                         "hf_USERPRO123")

    def test_api_propagates_edge_identity_not_persisted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / ".history_bucket_ready").write_text("test")
            with patch.object(history, "ROOT", root):
                with patch("gateway.main.invoke", return_value=b"fake-png") as invoke:
                    response = TestClient(app).post(
                        "/v1/images/generations",
                        headers={"X-HF-Token": "hf_PROUSER123", "X-IP-Token": "signed-edge-token"},
                        json={"model": "z-image-turbo", "prompt": "cat"},
                    )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(invoke.call_args.kwargs["ip_token"], "signed-edge-token")
                self.assertEqual(invoke.call_args.args[2], "hf_PROUSER123")
                saved = history.get_record(history.list_records()["items"][0]["id"])
                self.assertNotIn("signed-edge-token", str(saved))
                self.assertNotIn("hf_PROUSER123", str(saved))

    def test_api_edits_propagates_edge_identity(self):
        from PIL import Image
        from io import BytesIO
        buffer = BytesIO()
        Image.new("RGB", (4, 4), "red").save(buffer, "PNG")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / ".history_bucket_ready").write_text("test")
            with patch.object(history, "ROOT", root):
                with patch("gateway.main.download_public_image", return_value=buffer.getvalue()):
                    with patch("gateway.main.invoke", return_value=b"fake-png") as invoke:
                        response = TestClient(app).post(
                            "/v1/images/edit",
                            headers={"X-HF-Token": "hf_PROUSER123", "X-IP-Token": "signed-edge-token"},
                            json={"model": "qwen-image-2.1", "prompt": "portrait",
                                  "images": [{"image_url": "https://example.org/photo.png"}]},
                        )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(invoke.call_args.kwargs["ip_token"], "signed-edge-token")
                self.assertEqual(invoke.call_args.args[2], "hf_PROUSER123")


if __name__ == "__main__":
    unittest.main()
