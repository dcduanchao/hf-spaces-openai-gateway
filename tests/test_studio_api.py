"""Studio HTTP API transport and callback end-to-end regressions (no real GPU)."""
import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import httpx
from PIL import Image

from gateway import studio_api, ui, history


def png():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "blue").save(buf, format="PNG")
    return buf.getvalue()


class StudioApiTransportTests(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "model": "z-image-turbo", "prompt": "cat", "size": "1024x1024",
            "n": 2, "seed": -1, "extra_body": {},
        }

    def mocked_http_client(self, callback):
        # Custom transport exercises real httpx encoding, including multipart
        # and authorization headers, without network or GPU.
        real_client = httpx.Client
        def new_client(*args, **kwargs):
            kwargs.pop("trust_env", None)
            kwargs["transport"] = httpx.MockTransport(callback)
            return real_client(*args, **kwargs)
        return patch.object(studio_api.httpx, "Client", side_effect=new_client)

    def test_generation_post_json_token_and_b64(self):
        sample = base64.b64encode(png()).decode()
        def handler(req):
            self.assertEqual(req.url.path, "/v1/images/generations")
            self.assertEqual(req.headers["x-hf-token"], "hf_PRO123")
            self.assertNotIn("x-ip-token", req.headers)
            body = json.loads(req.content)
            self.assertEqual(body["n"], 2)
            self.assertEqual(body["response_format"], "b64_json")
            self.assertNotIn("hf_PRO123", req.content.decode())
            return httpx.Response(200, json={"data": [
                {"b64_json": sample}, {"b64_json": sample},
            ]})
        with self.mocked_http_client(handler):
            images = studio_api.api_generate(self.payload, "hf_PRO123")
        self.assertEqual(images, [png(), png()])

    def test_edits_multipart_preserves_file_order(self):
        with tempfile.TemporaryDirectory() as folder:
            one, two = Path(folder)/"first.png", Path(folder)/"second.png"
            one.write_bytes(png())
            two.write_bytes(png())
            def handler(req):
                self.assertEqual(req.url.path, "/v1/images/edits")
                self.assertEqual(req.headers["x-hf-token"], "hf_PRO123")
                data = req.content
                self.assertLess(data.index(b'filename="first.png"'), data.index(b'filename="second.png"'))
                self.assertIn(b'name="extra_body"', data)
                self.assertIn(b'name="response_format"', data)
                self.assertNotIn(b"hf_PRO123", data)
                return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png()).decode()}]})
            with self.mocked_http_client(handler):
                output = studio_api.api_generate(self.payload, "hf_PRO123", image_paths=[str(one), str(two)])
            self.assertEqual(output, [png()])

    def test_quota_is_a_safe_structured_error(self):
        def handler(req):
            return httpx.Response(429, json={"error": {
                "code": "upstream_quota_exceeded", "message": "HF token hf_SECRET123"
            }})
        with self.mocked_http_client(handler):
            with self.assertRaises(studio_api.StudioAPIError) as raised:
                studio_api.api_generate(self.payload, "hf_PRO123")
        self.assertEqual(raised.exception.code, "upstream_quota_exceeded")
        self.assertNotIn("hf_SECRET123", str(raised.exception))

    def test_anonymous_does_not_add_hf_token(self):
        def handler(req):
            self.assertNotIn("x-hf-token", req.headers)
            return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png()).decode()}]})
        with self.mocked_http_client(handler):
            self.assertEqual(len(studio_api.api_generate(self.payload, None)), 1)

    def test_ui_generates_only_via_api_without_duplicate_history(self):
        with patch.object(ui, "api_generate", return_value=[png()]) as transport:
            with patch.object(ui.history, "new_record", side_effect=AssertionError("duplicate history")):
                images, text = ui.generate("文生图", "z-image-turbo", "cat", [],
                    "1024x1024", 1, -1, "hf_PRO123", "{}", "")
        self.assertEqual(len(images), 1)
        self.assertIn("OpenAI API", text)
        self.assertEqual(transport.call_args.args[1], "hf_PRO123")


if __name__ == "__main__":
    unittest.main()
