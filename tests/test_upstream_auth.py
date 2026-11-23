"""Regression tests for HF identity propagation in cross-Space ZeroGPU calls."""
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from gateway.main import app
from gateway import history
from gateway.upstream_auth import client_kwargs, forward_headers, token_tail
from gateway.models import load_profiles
from gateway import z_image, qwen_image
import tempfile
from pathlib import Path


class IdentityForwardingTests(unittest.TestCase):
    def test_log_only_masked_suffix_never_full_token(self):
        secret = "hf_NOT_A_REAL_TOKEN_SECRET9876"
        self.assertEqual(token_tail(secret), "****9876")
        self.assertEqual(token_tail(None), "none")
        with self.assertLogs("uvicorn.error", level="INFO") as captured:
            result = client_kwargs(secret, "edge-token")
        text = "\\n".join(captured.output)
        self.assertIn("****9876", text)
        self.assertIn("auth_mode=hf_token", text)
        self.assertNotIn(secret, text)
        self.assertNotIn("edge-token", text)
        self.assertNotIn("headers", result)

    def test_studio_log_shows_only_token_suffix(self):
        from gateway import ui
        from types import SimpleNamespace
        from io import BytesIO
        from PIL import Image
        buffer = BytesIO()
        Image.new("RGB", (4, 4), "red").save(buffer, "PNG")
        secret = "hf_NOT_A_REAL_TOKEN_SECRET9876"
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / ".history_bucket_ready").write_text("test")
            with patch.object(history, "ROOT", root):
                with patch("gateway.ui.invoke", return_value=buffer.getvalue()):
                    with self.assertLogs("uvicorn.error", level="INFO") as captured:
                        ui.generate("文生图", "z-image-turbo", "cat", [],
                                    "1024x1024", 1, -1, secret, "{}", "",
                                    request=SimpleNamespace(headers={}))
        text = "\\n".join(captured.output)
        self.assertIn("source=studio", text)
        self.assertIn("****9876", text)
        self.assertNotIn(secret, text)

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

    def test_studio_passes_explicit_pro_token_to_generation(self):
        """Exercise actual Gradio callback args, not just an adapter mock."""
        from gateway import ui
        from types import SimpleNamespace
        from io import BytesIO
        from PIL import Image
        buffer = BytesIO()
        Image.new("RGB", (4, 4), "red").save(buffer, "PNG")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / ".history_bucket_ready").write_text("test")
            # UI must use the token input value even if request has an edge
            # identity from an unauthenticated private browser.
            browser_request = SimpleNamespace(headers={"x-ip-token": "anonymous-browser-edge"})
            with patch.object(history, "ROOT", root):
                with patch("gateway.ui.invoke", return_value=buffer.getvalue()) as invoke:
                    images, message = ui.generate(
                        "文生图", "z-image-turbo", "cat", [], "1024x1024",
                        1, -1, "hf_USERPRO123", "{}", "",
                        request=browser_request,
                    )
        self.assertEqual(len(images), 1)
        self.assertEqual(invoke.call_args.args[2], "hf_USERPRO123")
        self.assertEqual(invoke.call_args.kwargs["ip_token"], "anonymous-browser-edge")
        self.assertEqual(
            client_kwargs(invoke.call_args.args[2], invoke.call_args.kwargs["ip_token"]),
            {"token": "hf_USERPRO123", "verbose": False},
        )

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
