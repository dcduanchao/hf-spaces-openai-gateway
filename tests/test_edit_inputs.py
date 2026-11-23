"""OpenAI image edits: JSON URLs, multipart files, mixed sources, SSRF and bounds."""
import io
import base64
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
from gateway.image_inputs import _validate_url, ImageInputError, validate_png_bytes, decode_image_data_url, MAX_IMAGE_BYTES


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

    def test_mixed_base64_data_url_and_public_https_gallery_references(self):
        inline = "data:image/png;base64," + base64.b64encode(tiny_png()).decode("ascii")
        urls = [
            {"image_url": inline},
            {"image_url": "https://ga.du-ai.top/img/costume/SUM-0007.webp"},
            {"image_url": "https://ga.du-ai.top/img/upper80v2_P034_front.png"},
        ]
        captured_inputs = []
        def capture_invoke(profile, payload, token, **kwargs):
            captured_inputs.extend(Path(p).read_bytes() for p in kwargs["image_paths"])
            return tiny_png()
        with patch("gateway.main.download_public_image", return_value=tiny_png()) as downloaded:
            with patch("gateway.main.invoke", side_effect=capture_invoke) as invoke:
                response = self.client.post("/v1/images/edit", json=self.payload(
                    images=urls, size="1024x1536", quality="high",
                    input_fidelity="high", output_format="png",
                ))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(downloaded.call_count, 2)
        self.assertEqual([c.args[0] for c in downloaded.call_args_list],
                         [urls[1]["image_url"], urls[2]["image_url"]])
        self.assertEqual(invoke.call_args.args[1]["extra_body"]["quality"], "quality")
        self.assertEqual(len(invoke.call_args.kwargs["image_paths"]), 3)
        self.assertEqual(captured_inputs, [tiny_png(), tiny_png(), tiny_png()])
        record = history.get_record(history.list_records()["items"][0]["id"])
        self.assertEqual(len(record["input_images"]), 3)
        self.assertNotIn(inline, str(record))

    def test_base64_data_url_in_both_image_aliases(self):
        inline = "data:image/png;base64," + base64.b64encode(tiny_png()).decode()
        for route in ("/v1/images/edit", "/v1/images/edits"):
            for key in ("image", "images", "image_urls"):
                with self.subTest(route=route, key=key):
                    with patch("gateway.main.invoke", return_value=tiny_png()):
                        response = self.client.post(route, json=self.payload(**{
                            key: [{"image_url": inline}] if key == "images" else [inline]
                        }))
                    self.assertEqual(response.status_code, 200, response.text)

    def test_base64_input_validation_errors_redact_input(self):
        bad_cases = [
            "data:image/png;base64,@@@@SECRET",
            "data:image/gif;base64,AAAA",
            "data:image/png;base64," + "A" * (14 * 1024 * 1024),
            "data:image/png;base64," + base64.b64encode(b"not a picture").decode(),
        ]
        for inline in bad_cases:
            with self.subTest(size=len(inline)):
                response = self.client.post("/v1/images/edit", json=self.payload(images=[
                    {"image_url": inline}
                ]))
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(response.json()["error"]["code"], "invalid_image")
                self.assertNotIn("SECRET", response.text)
                self.assertNotIn("AAAA", response.text)

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

    def test_pose_gallery_object_images_and_singular_alias(self):
        from gateway.adapter import parse_size
        from gateway.models import load_profiles
        payload = self.payload(
            images=[
                {"image_url": "https://ga.du-ai.top/img/portrait/f009cba9893.webp"},
                {"image_url": "https://ga.du-ai.top/img/costume/SUM-0007.webp"},
                {"image_url": "https://ga.du-ai.top/img/full_single_P009_low.png"},
            ],
            size="1024x1536",
            quality="high",
            input_fidelity="high",
            output_format="png",
        )
        with patch("gateway.main.download_public_image", return_value=tiny_png()) as get:
            with patch("gateway.main.invoke", return_value=tiny_png()) as invoke:
                response = self.client.post("/v1/images/edit", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(get.call_count, 3)
        sent = invoke.call_args.args[1]
        self.assertEqual(sent["size"], "1024x1536")
        self.assertEqual(sent["extra_body"]["quality"], "quality")
        self.assertNotIn("input_fidelity", sent["extra_body"])
        self.assertNotIn("output_format", sent["extra_body"])
        self.assertEqual(len(invoke.call_args.kwargs["image_paths"]), 3)
        self.assertEqual(parse_size("1024x1536", load_profiles()["qwen-image-2.1"]), (1024, 1536))

    def test_singular_url_object(self):
        with patch("gateway.main.download_public_image", return_value=tiny_png()):
            with patch("gateway.main.invoke", return_value=tiny_png()):
                response = self.client.post("/v1/images/edit", json=self.payload(
                    image={"image_url": "https://example.org/public/reference.png"}))
        self.assertEqual(response.status_code, 200, response.text)

    def test_unsafe_or_unsupported_options(self):
        source = {"images": [{"image_url": "https://example.org/one.png"}]}
        for setting in ({"output_format": "jpeg"}, {"input_fidelity": "ultra"}):
            with self.subTest(setting=setting):
                response = self.client.post("/v1/images/edit", json=self.payload(**source, **setting))
                self.assertEqual(response.status_code, 400, response.text)
        response = self.client.post("/v1/images/edit", json=self.payload(
            images=[{"image_url": "https://example.org/a.png", "secret": "bad"}]))
        self.assertEqual(response.status_code, 400, response.text)

    def test_custom_size_validation(self):
        from gateway.adapter import parse_size
        from gateway.models import load_profiles
        profile = load_profiles()["qwen-image-2.1"]
        self.assertEqual(parse_size("1024x1536", profile), (1024, 1536))
        self.assertEqual(parse_size("1200x1600", profile), (1200, 1600))
        for size in ("10x10", "8192x8192", "hello", "0x1024", "1024x10000"):
            with self.subTest(size=size), self.assertRaises(ValueError):
                parse_size(size, profile)

    def test_private_window_token_forwarded_for_both_edit_routes(self):
        for route in ("/v1/images/edit", "/v1/images/edits"):
            for headers in ({"X-HF-Token": "hf_USERPRO123"},
                            {"Authorization": "Bearer hf_USERPRO123"}):
                with self.subTest(route=route, header=list(headers)):
                    with patch("gateway.main.download_public_image", return_value=tiny_png()):
                        with patch("gateway.main.invoke", return_value=tiny_png()) as invoke:
                            response = self.client.post(route, headers=headers, json=self.payload(
                                images=[{"image_url": "https://example.org/reference.png"}]))
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(invoke.call_args.args[2], "hf_USERPRO123")

    def test_multipart_token_forwarding(self):
        with patch("gateway.main.invoke", return_value=tiny_png()) as invoke:
            response = self.client.post(
                "/v1/images/edits", headers={"X-HF-Token": "hf_USERPRO123"},
                data={"model": "qwen-image-2.1", "prompt": "make a portrait",
                      "response_format": "b64_json"},
                files={"image": ("ref.png", tiny_png(), "image/png")})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(invoke.call_args.args[2], "hf_USERPRO123")

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

    def test_base64_decoder_rejects_invalid_and_oversized(self):
        good = "data:image/png;base64," + base64.b64encode(tiny_png()).decode()
        self.assertEqual(decode_image_data_url(good), tiny_png())
        for bad in ("data:text/plain;base64,SGVsbG8=", "data:image/png;base64,!!!!",
                    "data:image/png;base64," + "A" * (14 * 1024 * 1024)):
            with self.subTest(length=len(bad)), self.assertRaises(ImageInputError):
                decode_image_data_url(bad)

    def test_png_validation(self):
        self.assertTrue(validate_png_bytes(tiny_png()).startswith(bytes([137, 80, 78, 71])))


if __name__ == "__main__":
    unittest.main()
