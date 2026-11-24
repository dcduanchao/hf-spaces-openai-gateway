"""Offline API contract tests for jmendelblatt/moss-image-nsfw-lora."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from gateway import history
from gateway.main import app
from gateway.models import load_profiles
from gateway.moss_image_nsfw import build_kwargs, run
from gateway.ui import model_size_choices, model_space_link, resolve_seed


class MossNsfwTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_profiles()["moss-image-nsfw"]
        self.request = {"model": "moss-image-nsfw", "prompt": "portrait",
                        "size": "1024x1024", "seed": 3}

    def test_studio_random_seed_default_and_manual_seed(self):
        self.assertEqual(resolve_seed(-1, True), -1)
        self.assertEqual(resolve_seed(123, True), -1)
        self.assertEqual(resolve_seed(123, False), 123)
        self.assertEqual(resolve_seed(0, False), 0)
        for invalid in (-1, -2, 1.5, 2**32, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                resolve_seed(invalid, False)

    def test_openai_api_seed_optional_defaults_to_random(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".history_bucket_ready").write_text("ok")
            with patch.object(history, "ROOT", root):
                with patch("gateway.main.invoke", return_value=b"png") as invoker:
                    response = TestClient(app).post("/v1/images/generations", json={
                        "model": "moss-image-nsfw", "prompt": "portrait",
                        "size": "1024x1024", "response_format": "b64_json",
                    })
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(invoker.call_args.args[1]["seed"], -1)

    def test_profile_and_ui_link(self):
        self.assertEqual(self.profile.api_name, "/generate")
        self.assertEqual(self.profile.space_id, "jmendelblatt/moss-image-nsfw-lora")
        self.assertEqual(self.profile.modes, ["generations"])
        self.assertIn("1024x1024", self.profile.allowed_sizes)
        self.assertIn(self.profile.space_id, model_space_link(self.profile))
        self.assertTrue(model_size_choices(self.profile))

    def test_all_ten_upstream_params_and_types(self):
        kwargs = build_kwargs(self.profile, self.request)
        self.assertEqual(set(kwargs), {
            "prompt", "negative_prompt", "width", "height", "steps",
            "guidance_scale", "lora_scale", "realistic_scale", "add_trigger", "seed",
        })
        self.assertEqual(kwargs["prompt"], "portrait")
        self.assertIn("blurry", kwargs["negative_prompt"])
        for name, value in (("width", 1024.0), ("height", 1024.0),
                            ("steps", 30.0), ("guidance_scale", 3.0),
                            ("lora_scale", 0.7), ("realistic_scale", 1.0)):
            self.assertEqual(kwargs[name], value)
            self.assertIs(type(kwargs[name]), float)
        self.assertIs(kwargs["add_trigger"], True)
        self.assertIs(type(kwargs["seed"]), int)
        self.assertEqual(kwargs["seed"], 3)

    def test_overrides_and_random_seed(self):
        request = {**self.request, "size": "768x512", "seed": -1,
                   "extra_body": {"negative_prompt": "", "steps": 24,
                                  "guidance_scale": 4.5, "lora_scale": 0.5,
                                  "realistic_scale": 0.8, "add_trigger": False}}
        with patch("gateway.moss_image_nsfw.secrets.randbelow", return_value=789) as rnd:
            kw = build_kwargs(self.profile, request)
        self.assertEqual((kw["width"], kw["height"]), (768.0, 512.0))
        self.assertEqual(kw["seed"], 789)
        rnd.assert_called_once_with(2**32)
        self.assertEqual(kw["negative_prompt"], "")
        self.assertEqual(kw["steps"], 24.0)
        self.assertEqual(kw["guidance_scale"], 4.5)
        self.assertEqual(kw["lora_scale"], 0.5)
        self.assertEqual(kw["realistic_scale"], 0.8)
        self.assertIs(kw["add_trigger"], False)

    def test_reject_invalid_values(self):
        for extra in ({"unknown": 1}, {"steps": "30"}, {"steps": True},
                      {"lora_scale": float("nan")}, {"add_trigger": 1},
                      {"negative_prompt": None}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                build_kwargs(self.profile, {**self.request, "extra_body": extra})
        with self.assertRaises(ValueError):
            build_kwargs(self.profile, {**self.request, "seed": -2})
        with self.assertRaises(ValueError):
            build_kwargs(self.profile, self.request, ["reference.png"])

    def test_result_is_a_single_image(self):
        with (
            patch("gateway.moss_image_nsfw.Client") as Client,
            patch("gateway.moss_image_nsfw.predict_logged",
                  return_value=["/tmp/generated.png"]) as predict,
            patch("gateway.moss_image_nsfw.result_bytes", return_value=b"png") as result_bytes,
        ):
            image = run(self.profile, self.request, "hf_TEST")
        self.assertEqual(image, b"png")
        self.assertEqual(Client.call_args.kwargs["token"], "hf_TEST")
        self.assertEqual(predict.call_args.kwargs["api_name"], "/generate")
        self.assertEqual(set(predict.call_args.kwargs["kwargs"]),
                         set(build_kwargs(self.profile, self.request)))
        result_bytes.assert_called_once_with("/tmp/generated.png", self.profile)
        Client.return_value.close.assert_called_once()

    def test_unwrapped_single_filepath_from_gradio_client(self):
        """Real Gradio Client returns a plain filepath for one output."""
        with (
            patch("gateway.moss_image_nsfw.Client") as client,
            patch("gateway.moss_image_nsfw.predict_logged",
                  return_value="/tmp/image.png") as predict,
            patch("gateway.moss_image_nsfw.result_bytes", return_value=b"png") as convert,
        ):
            output = run(self.profile, self.request, None)
        self.assertEqual(output, b"png")
        convert.assert_called_once_with("/tmp/image.png", self.profile)
        self.assertEqual(predict.call_args.kwargs["api_name"], "/generate")
        client.return_value.close.assert_called_once()

    def test_gradio_file_data_response(self):
        """Some Gradio clients provide a FileData-like path dictionary."""
        with (
            patch("gateway.moss_image_nsfw.Client"),
            patch("gateway.moss_image_nsfw.predict_logged",
                  return_value={"path": "/tmp/image.png", "orig_name": "image.png"}),
            patch("gateway.moss_image_nsfw.result_bytes", return_value=b"png") as convert,
        ):
            output = run(self.profile, self.request, None)
        self.assertEqual(output, b"png")
        convert.assert_called_once_with(
            {"path": "/tmp/image.png", "orig_name": "image.png"}, self.profile)

    def test_invalid_result_contract(self):
        with (
            patch("gateway.moss_image_nsfw.Client"),
            patch("gateway.moss_image_nsfw.predict_logged", return_value=[]),
        ):
            with self.assertRaises(ValueError):
                run(self.profile, self.request, None)

    def test_public_model_and_image_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".history_bucket_ready").write_text("ok")
            with patch.object(history, "ROOT", root):
                client = TestClient(app)
                models = client.get("/v1/models").json()["data"]
                self.assertIn("moss-image-nsfw", {item["id"] for item in models})
                with patch("gateway.main.invoke", return_value=b"png") as invoke:
                    result = client.post("/v1/images/generations", json={
                        **self.request, "response_format": "b64_json",
                    })
                self.assertEqual(result.status_code, 200, result.text)
                self.assertEqual(invoke.call_args.args[0].adapter, "moss_image_nsfw")


if __name__ == "__main__":
    unittest.main()
