"""OpenAI image edits: JSON URLs, multipart files, mixed sources, SSRF and bounds."""
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image
from gateway.main import app
from gateway import history
from gateway.image_inputs import _validate_url, ImageInputError, validate_png_bytes


def tiny_png():
    buf = io.BytesIO()
    Image.new("RGB", (5, 5), "red").save(buf, "PNG")
    return buf.getvalue()


class EditUrlTests(unittest.TestCase):
    def setUp(self):
        self.bucket = tempfile.TemporaryDirectory()
        self.addCleanup(self.bucket.cleanup)
        root = Path(self.bucket.name)
        (root / ".history_bucket_ready").write_text("test")
        patched = patch.object(history, "ROOT", root)
        patched.start()
        self.addCleanup(patched.stop)
        self.client = TestClient(app)

    def payload(self, **kwargs):
        return {
            "model": "qwen-image-2.1", "prompt": "change background",
            "size": "1024x1024", "response_format": "b64_json", **kwargs
        }

    def test_single_public_url(self):
        with patch("gateway.main.download_public_image", return_value=tiny_png()) as get:
            with patch("gateway.main.invoke", return_value=tiny_png()) as invoke:
                res = self.client.post("/v1/images/edits", json=self.payload(
                    image="https://images.example.org/a.png"))
        self.assertEqual(res.status_code, 200, res.text)
        get.assert_called_once_with("https://images.example.org/a.png")
        self.assertEqual(len(invoke.call_args.kwargs["image_paths"]), 1)
        self.assertEqual(history.list_records()["total"], 1)
        record = history.get_record(history.list_records()["items"][0]["id"])
        self.assertEqual(len(record["input_images"]), 1)
        self.assertEqual(len(record["images"]), 1)

    def test_multiple_public_urls(self):
        urls = ["https://images.example.org/one.png", "https://images.example.org/two.png"]
        with patch("gateway.main.download_public_image", return_value=tiny_png()) as get:
            with patch("gateway.main.invoke", return_value=tiny_png()) as invoke:
                res = self.client.post("/v1/images/edits", json=self.payload(
                    image_urls=urls, extra_body={"quality": "speed"}))
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(get.call_count, 2)
        self.assertEqual(len(invoke.call_args.kwargs["image_paths"]), 2)

    def test_multi_file_upload(self):
        data = [("image", ("a.png", tiny_png(), "image/png")),
                ("image", ("b.png", tiny_png(), "image/png"))]
        with patch("gateway.main.invoke", return_value=tiny_png()) as invoke:
            res = self.client.post("/v1/images/edits", data={
                "model": "qwen-image-2.1", "prompt": "combine", "response_format": "b64_json"
            }, files=data)
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(len(invoke.call_args.kwargs["image_paths"]), 2)

    def test_mixed_upload_and_url(self):
        with patch("gateway.main.download_public_image", return_value=tiny_png()):
            with patch("gateway.main.invoke", return_value=tiny_png()) as invoke:
                res = self.client.post("/v1/images/edits", data={
                    "model": "qwen-image-2.1", "prompt": "combine",
                    "image_url": "https://images.example.org/a.png"
                }, files={"image": ("local.png", tiny_png(), "image/png")})
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(len(invoke.call_args.kwargs["image_paths"]), 2)

    def test_reject_five_urls(self):
        res = self.client.post("/v1/images/edits", json=self.payload(
            image_urls=["https://example.org/image.png"] * 5))
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["error"]["code"], "invalid_image_count")

    def test_reject_z_image_edits(self):
        res = self.client.post("/v1/images/edits", json=self.payload(
            model="z-image-turbo", image="https://example.org/image.png"))
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["error"]["code"], "unsupported_mode")

    def test_reject_no_images(self):
        res = self.client.post("/v1/images/edits", json=self.payload())
        self.assertEqual(res.status_code, 400)

    def test_bad_remote_url(self):
        with patch("gateway.main.download_public_image", side_effect=ImageInputError("blocked address")):
            res = self.client.post("/v1/images/edits", json=self.payload(
                image="https://example.org/image.png"))
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["error"]["code"], "invalid_image")

    def test_reject_non_image_upload(self):
        res = self.client.post("/v1/images/edits", data={"model":"qwen-image-2.1","prompt":"cat"},
                               files={"image":("note.txt",b"text","text/plain")})
        self.assertEqual(res.status_code, 400)

    def test_legacy_extra_json_string(self):
        with patch("gateway.main.invoke", return_value=tiny_png()):
            res=self.client.post("/v1/images/edits", data={
                "model":"qwen-image-2.1","prompt":"cat",
                "extra_body":json.dumps({"quality":"quality"})},
                files={"image":("a.png",tiny_png(),"image/png")})
        self.assertEqual(res.status_code, 200, res.text)


class RemoteValidationTests(unittest.TestCase):
    def test_block_local_and_non_https(self):
        for url in ("http://example.com/a.png", "https://localhost/a.png",
                    "https://127.0.0.1/a.png", "https://169.254.169.254/latest/meta-data",
                    "https://user:pass@example.com/a.png",
                    "https://example.com:8443/a.png"):
            with self.subTest(url=url):
                with self.assertRaises(ImageInputError):
                    _validate_url(url)

    def test_public_dns_accepted(self):
        with patch("gateway.image_inputs.socket.getaddrinfo", return_value=[
            (2,1,6,"",("93.184.215.14",443))]):
            host, ip, path = _validate_url("https://example.org/a.png?x=1")
        self.assertEqual(host,"example.org")
        self.assertEqual(ip,"93.184.215.14")
        self.assertEqual(path,"/a.png?x=1")

    def test_dns_mixed_private_rejected(self):
        with patch("gateway.image_inputs.socket.getaddrinfo", return_value=[
            (2,1,6,"",("93.184.215.14",443)),(2,1,6,"",("10.0.0.4",443))]):
            with self.assertRaises(ImageInputError):
                _validate_url("https://example.org/a.png")

    def test_png_validation(self):
        self.assertTrue(validate_png_bytes(tiny_png()).startswith(b"\\x89PNG"))


if __name__ == "__main__":
    unittest.main()
