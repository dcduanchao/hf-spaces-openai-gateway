"""Viggle Turbo regression tests: existing models untouched, adapter contract pinned."""
import base64
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image

from gateway import history
from gateway.main import app
from gateway.models import load_profiles
from gateway.viggle_turbo import build_args, run, MAX_REFERENCES, SQUARE_PRESET, CUSTOM_PRESET, MAX_SEED


def png():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buf, "PNG")
    return buf.getvalue()


class ViggleContractTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_profiles()["qwen-image-2.1-viggle-turbo"]

    def test_catalog_preserves_existing_models(self):
        profiles = load_profiles()
        self.assertIn("qwen-image-2.1", profiles)
        self.assertIn("z-image-turbo", profiles)
        self.assertIn("generations", self.profile.modes)
        self.assertIn("edits", self.profile.modes)
        self.assertEqual(self.profile.api_name, "/generate")
        self.assertEqual(self.profile.extras["steps"].default, 6)
        self.assertEqual(MAX_REFERENCES, 4)

    def test_generate_arguments_for_square(self):
        args = build_args(self.profile, {"prompt":"cat", "seed":123, "size":"1024x1024"})
        self.assertEqual(len(args), 10)
        self.assertEqual(args[0], "cat")
        self.assertEqual(args[1], [])
        self.assertEqual(args[2], SQUARE_PRESET)
        self.assertEqual(args[3:7], [123, False, "Off", 6])
        self.assertEqual(args[7:10], [1024, 1024, "v0.3"])

    def test_custom_size_and_four_ordered_references(self):
        with patch("gateway.viggle_turbo.handle_file", side_effect=lambda p: {"path":p, "meta":{"_type":"gradio.FileData"}}):
            args = build_args(self.profile, {
                "prompt": "hello", "size": "1024x1536", "seed": -1,
                "extra_body": {"steps": 7},
            }, ["one.png","two.png","three.png","four.png"])
        self.assertEqual(args[2], CUSTOM_PRESET)
        self.assertEqual([x["image"]["path"] for x in args[1]],
                         ["one.png","two.png","three.png","four.png"])
        self.assertEqual(args[5:10], ["Off",7,1024,1536,"v0.3"])
        self.assertFalse(args[4])

    def test_random_seed_is_within_upstream_slider_range(self):
        # Before fix, randbelow(2**32) exceeded the Viggle UI's 31-bit slider
        # maximum about half of the time and Gradio rejected the request.
        with patch("gateway.viggle_turbo.secrets.randbelow", return_value=MAX_SEED) as rnd:
            args = build_args(self.profile, {"prompt": "cat", "seed": -1})
        rnd.assert_called_once_with(MAX_SEED + 1)
        self.assertEqual(args[3], MAX_SEED)
        self.assertFalse(args[4])

    def test_seed_over_upstream_limit_rejected_before_client_call(self):
        with self.assertRaisesRegex(ValueError, "2147483647"):
            build_args(self.profile, {"prompt": "cat", "seed": MAX_SEED + 1})

    def test_custom_image_size_matches_upstream_dropdown(self):
        self.assertEqual(
            build_args(self.profile, {"prompt": "cat", "size": "1024x1536"})[2],
            "Custom · set width and height below",
        )
        with self.assertRaisesRegex(ValueError, "32"):
            build_args(self.profile, {"prompt": "cat", "size": "1000x1500"})

    def test_invalid_options_and_reference_limit(self):
        for extras in ({"model_version":"v0.2.1"}, {"enhance_prompt":"On"}, {"steps":10}):
            with self.subTest(extras=extras), self.assertRaises(ValueError):
                build_args(self.profile, {"prompt":"cat", "extra_body":extras})
        with self.assertRaises(ValueError):
            build_args(self.profile, {"prompt":"cat"}, ["a"]*5)

    def test_upstream_receives_token_and_positional_generate(self):
        with patch("gateway.viggle_turbo.Client") as client_class, patch(
            "gateway.viggle_turbo.result_bytes", return_value=png()
        ):
            client_class.return_value.predict.return_value = [{"path":"/tmp/generated.png"}, "status", "prompt"]
            result = run(self.profile, {"prompt":"cat"}, "hf_TEST123", ip_token="browser-edge")
        self.assertEqual(result, png())
        self.assertEqual(client_class.call_args.kwargs["token"], "hf_TEST123")
        self.assertNotIn("headers", client_class.call_args.kwargs)
        args, kw = client_class.return_value.predict.call_args
        self.assertEqual(kw, {"api_name":"/generate"})
        self.assertEqual(len(args), 10)
        self.assertEqual(args[9], "v0.3")
        client_class.return_value.close.assert_called_once()

    def test_gradio_client_valueerror_is_upstream_not_input_validation(self):
        with patch("gateway.viggle_turbo.Client") as client_class:
            client_class.return_value.predict.side_effect = ValueError("possible hf_SECRET or prompt")
            with self.assertRaisesRegex(RuntimeError, "upstream client rejected"):
                run(self.profile, {"prompt": "cat"}, "hf_TEST123")
        client_class.return_value.close.assert_called_once()

    def test_gradio_client_init_valueerror_is_upstream(self):
        with patch("gateway.viggle_turbo.Client", side_effect=ValueError("client schema changed")):
            with self.assertRaisesRegex(RuntimeError, "initialization rejected"):
                run(self.profile, {"prompt": "cat"}, "hf_TEST123")

    def test_real_image_edit_api_uses_viggle_and_bucket(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / ".history_bucket_ready").write_text("test")
            with patch.object(history, "ROOT", root):
                with patch("gateway.main.invoke", return_value=png()) as invoke:
                    res = TestClient(app).post("/v1/images/edits",
                        headers={"X-HF-Token":"hf_TEST123"},
                        data={"model":"qwen-image-2.1-viggle-turbo", "prompt":"portrait", "size":"1024x1536", "response_format":"b64_json"},
                        files={"image":("ref.png",png(),"image/png")})
                self.assertEqual(res.status_code,200,res.text)
                self.assertEqual(base64.b64decode(res.json()["data"][0]["b64_json"]),png())
                self.assertEqual(invoke.call_args.args[0].adapter,"viggle_turbo")
                self.assertEqual(invoke.call_args.args[2],"hf_TEST123")
                self.assertEqual(len(invoke.call_args.kwargs["image_paths"]),1)
                self.assertEqual(history.list_records()["total"],1)


if __name__ == "__main__":
    unittest.main()
