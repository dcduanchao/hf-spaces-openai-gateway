"""Offline contract tests for hugging-apps/moss-image-4step-lora."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from gateway import history
from gateway.main import app
from gateway.models import load_profiles
from gateway.moss_image_4 import RESOLUTIONS, build_kwargs, run
from gateway.ui import model_size_choices, model_space_link


class MossFourStepTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_profiles()["mosss-image-4"]
        self.request = {
            "model": "mosss-image-4", "prompt": "a cat",
            "size": "1024x1024", "seed": 33,
        }

    def test_catalog_and_ui(self):
        self.assertEqual(self.profile.space_id, "hugging-apps/moss-image-4step-lora")
        self.assertEqual(self.profile.api_name, "/generate")
        self.assertEqual(self.profile.modes, ["generations"])
        self.assertEqual(self.profile.allowed_sizes, list(RESOLUTIONS))
        self.assertEqual(len(model_size_choices(self.profile)), 12)
        self.assertIn(self.profile.space_id, model_space_link(self.profile))

    def test_exact_resolution_mapping_for_every_choice(self):
        for size in RESOLUTIONS:
            with self.subTest(size=size):
                result = build_kwargs(self.profile, {**self.request, "size": size})
                width, height = size.split("x")
                self.assertEqual(result, {
                    "prompt": "a cat",
                    "resolution": f"{width} × {height}",
                    "seed": 33,
                })

    def test_random_seed_and_fixed_seed(self):
        with patch("gateway.moss_image_4.secrets.randbelow", return_value=54321) as rnd:
            random_args = build_kwargs(self.profile, {**self.request, "seed": -1})
        self.assertEqual(random_args["seed"], 54321)
        rnd.assert_called_once_with(2**32)
        self.assertEqual(build_kwargs(self.profile, self.request)["seed"], 33)

    def test_reject_invalid_sizes_and_extras(self):
        for size in ("1024x512", "800x800", "2048x2048"):
            with self.subTest(size=size), self.assertRaises(ValueError):
                build_kwargs(self.profile, {**self.request, "size": size})
        for extra in ({"steps": 4}, {"lora_strength": 0.7}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                build_kwargs(self.profile, {**self.request, "extra_body": extra})
        with self.assertRaises(ValueError):
            build_kwargs(self.profile, {**self.request, "seed": -2})
        with self.assertRaises(ValueError):
            build_kwargs(self.profile, self.request, ["example.png"])

    def test_adapter_calls_generate_and_extracts_only_image(self):
        with (
            patch("gateway.moss_image_4.Client") as Client,
            patch("gateway.moss_image_4.predict_logged",
                  return_value=("/tmp/image.png", "seed used")) as predict,
            patch("gateway.moss_image_4.result_bytes", return_value=b"png") as image_bytes,
        ):
            result = run(self.profile, self.request, "hf_TEST")
        self.assertEqual(result, b"png")
        self.assertEqual(Client.call_args.kwargs["token"], "hf_TEST")
        self.assertEqual(predict.call_args.kwargs["api_name"], "/generate")
        self.assertEqual(predict.call_args.kwargs["kwargs"]["resolution"], "1024 × 1024")
        image_bytes.assert_called_once_with("/tmp/image.png", self.profile)
        Client.return_value.close.assert_called_once()

    def test_invalid_result_rejected(self):
        with (
            patch("gateway.moss_image_4.Client"),
            patch("gateway.moss_image_4.predict_logged", return_value=["only_image"]),
        ):
            with self.assertRaises(ValueError):
                run(self.profile, self.request, None)

    def test_public_models_and_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".history_bucket_ready").write_text("ok")
            with patch.object(history, "ROOT", root):
                client = TestClient(app)
                models = client.get("/v1/models").json()["data"]
                self.assertIn("mosss-image-4", {model["id"] for model in models})
                with patch("gateway.main.invoke", return_value=b"png") as invoker:
                    response = client.post("/v1/images/generations", json={
                        **self.request, "response_format": "b64_json",
                    })
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(invoker.call_args.args[0].adapter, "moss_image_4")


if __name__ == "__main__":
    unittest.main()
