"""Offline Gradio /generate contract and API integration for MiniMax H3 MysticXXX V5."""
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
from gateway.minimax_h3 import ASPECTS, build_kwargs, run
from gateway.models import load_profiles
from gateway.ui import model_size_choices, model_space_link


def tiny_png():
    buffer = io.BytesIO()
    Image.new("RGB", (6, 6), "red").save(buffer, "PNG")
    return buffer.getvalue()


class MiniMaxH3Tests(unittest.TestCase):
    def setUp(self):
        self.profile = load_profiles()["minimax-h3-mysticxxx-v5"]
        self.payload = {"model": self.profile.id, "prompt": "a cat", "size": "auto", "seed": -1}

    def test_catalog_and_studio(self):
        self.assertEqual(self.profile.space_id, "baka999/MiniMax-H3-Image-MysticXXX-V5")
        self.assertEqual(self.profile.api_name, "/generate")
        self.assertEqual(self.profile.modes, ["generations", "edits"])
        self.assertEqual(model_size_choices(self.profile)[0][1], "auto")
        self.assertIn(self.profile.space_id, model_space_link(self.profile))

    def test_all_twelve_upstream_parameters(self):
        kw = build_kwargs(self.profile, self.payload)
        self.assertEqual(set(kw), {
            "prompt", "images", "aspect", "resolution", "seed", "randomize",
            "enhance", "reasoning_effort", "lora_scale", "lora_attn_scale",
            "lora_mlp_scale", "lora_early_scale",
        })
        self.assertEqual(kw["images"], [])
        self.assertEqual(kw["aspect"], "auto")
        self.assertEqual(kw["resolution"], "4 MP")
        self.assertEqual(kw["seed"], 0.0)
        self.assertIs(kw["randomize"], True)
        self.assertIs(kw["enhance"], False)
        self.assertEqual(kw["reasoning_effort"], "low")
        for k, expected in (("lora_scale", 0.6), ("lora_attn_scale", 0.5),
                            ("lora_mlp_scale", 0.9), ("lora_early_scale", 0.4)):
            self.assertEqual(kw[k], expected)
            self.assertIs(type(kw[k]), float)

    def test_size_ratios_and_fixed_seed(self):
        for size, aspect in (
            ("1024x1024", "1:1"), ("1280x1024", "5:4"),
            ("1024x768", "4:3"), ("1152x768", "3:2"),
            ("1280x720", "16:9"), ("1344x576", "21:9"),
            ("1024x1280", "4:5"), ("768x1024", "3:4"),
            ("768x1152", "2:3"), ("720x1280", "9:16"),
        ):
            with self.subTest(size=size):
                args = build_kwargs(self.profile, {**self.payload, "size": size, "seed": 17})
                self.assertEqual(args["aspect"], aspect)
                self.assertFalse(args["randomize"])
                self.assertEqual(args["seed"], 17.0)
        self.assertEqual(len(ASPECTS), 11)

    def test_eight_references_preserve_order(self):
        paths = [f"/tmp/ref_{i}.png" for i in range(8)]
        with patch("gateway.minimax_h3.handle_file", side_effect=lambda p: {"path": p}):
            kwargs = build_kwargs(self.profile, self.payload, paths)
        self.assertEqual(kwargs["images"], [
            {"image": {"path": path}, "caption": None} for path in paths
        ])
        with self.assertRaises(ValueError):
            build_kwargs(self.profile, self.payload, paths + ["/tmp/ninth.png"])

    def test_override_and_validation(self):
        kwargs = build_kwargs(self.profile, {**self.payload, "extra_body": {
            "aspect": "4:3", "resolution": "2 MP", "enhance": True,
            "reasoning_effort": "high", "lora_scale": 0.4,
            "lora_attn_scale": 0.3, "lora_mlp_scale": 0.8, "lora_early_scale": 0.2,
        }})
        self.assertEqual(kwargs["aspect"], "4:3")
        self.assertEqual(kwargs["resolution"], "2 MP")
        self.assertEqual(kwargs["reasoning_effort"], "high")
        self.assertTrue(kwargs["enhance"])
        for extra in ({"aspect": "17:10"}, {"enhance": "true"},
                      {"lora_scale": float("nan")}, {"unknown": 1}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                build_kwargs(self.profile, {**self.payload, "extra_body": extra})

    def test_run_returns_first_image_from_three_outputs(self):
        with (
            patch("gateway.minimax_h3.Client") as client,
            patch("gateway.minimax_h3.predict_logged",
                  return_value=("/tmp/result.png", 42.0, "used prompt")) as predict,
            patch("gateway.minimax_h3.result_bytes", return_value=tiny_png()) as convert,
        ):
            png = run(self.profile, self.payload, "hf_TEST")
        self.assertEqual(png, tiny_png())
        self.assertEqual(client.call_args.kwargs["token"], "hf_TEST")
        self.assertEqual(predict.call_args.kwargs["api_name"], "/generate")
        convert.assert_called_once_with("/tmp/result.png", self.profile)
        client.return_value.close.assert_called_once()

    def test_generation_and_eight_image_edits_api(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / ".history_bucket_ready").write_text("ok")
            with patch.object(history, "ROOT", root):
                client = TestClient(app)
                models = client.get("/v1/models").json()["data"]
                self.assertIn(self.profile.id, {model["id"] for model in models})
                with patch("gateway.main.invoke", return_value=tiny_png()) as invoker:
                    generated = client.post("/v1/images/generations", json={
                        "model": self.profile.id, "prompt": "cat",
                        "size": "auto", "response_format": "b64_json",
                    })
                self.assertEqual(generated.status_code, 200, generated.text)
                self.assertEqual(invoker.call_args.args[1]["size"], "auto")
                with patch("gateway.main.invoke", return_value=tiny_png()) as invoker:
                    no_size = client.post("/v1/images/generations", json={
                        "model": self.profile.id, "prompt": "cat",
                        "response_format": "b64_json",
                    })
                self.assertEqual(no_size.status_code, 200, no_size.text)
                self.assertEqual(invoker.call_args.args[0].default_size, "auto")
                image_files = [("image", (f"ref{i}.png", tiny_png(), "image/png"))
                               for i in range(8)]
                with patch("gateway.main.invoke", return_value=tiny_png()) as invoker:
                    edited = client.post("/v1/images/edits", data={
                        "model": self.profile.id, "prompt": "edit",
                        "size": "auto", "response_format": "b64_json",
                    }, files=image_files)
                self.assertEqual(edited.status_code, 200, edited.text)
                self.assertEqual(len(invoker.call_args.kwargs["image_paths"]), 8)
                self.assertEqual(base64.b64decode(edited.json()["data"][0]["b64_json"]), tiny_png())

    def test_other_models_still_reject_more_than_four_refs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / ".history_bucket_ready").write_text("ok")
            with patch.object(history, "ROOT", root):
                response = TestClient(app).post("/v1/images/edits", data={
                    "model": "qwen-image-2.1", "prompt": "edit", "size": "1024x1024",
                }, files=[("image", (f"ref{i}.png", tiny_png(), "image/png")) for i in range(5)])
            self.assertEqual(response.status_code, 400, response.text)
            self.assertEqual(response.json()["error"]["code"], "invalid_image_count")


if __name__ == "__main__":
    unittest.main()
