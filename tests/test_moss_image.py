"""Offline contract tests for hugging-apps/moss-image-8-step-dmd /generate."""
import io
import unittest
from unittest.mock import patch

from PIL import Image
from fastapi.testclient import TestClient

from gateway.models import load_profiles
from gateway.moss_image import build_kwargs, run
from gateway.ui import model_size_choices, model_space_link
from gateway import history
from gateway.main import app


class MossImageTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_profiles()["moss-image-8"]
        self.request = {"model": "moss-image-8", "prompt": "cat", "size": "1024x768", "seed": -1}

    def test_catalog(self):
        self.assertEqual(self.profile.space_id, "hugging-apps/moss-image-8-step-dmd")
        self.assertEqual(self.profile.api_name, "/generate")
        self.assertEqual(self.profile.modes, ["generations"])
        self.assertIn("1024x1024", self.profile.allowed_sizes)
        self.assertIn(self.profile.space_id, model_space_link(self.profile))
        self.assertTrue(model_size_choices(self.profile))

    def test_all_seven_arguments_types_and_defaults(self):
        kwargs = build_kwargs(self.profile, self.request)
        self.assertEqual(set(kwargs), {
            "prompt", "height", "width", "num_steps", "seed", "randomize_seed", "lora_strength"
        })
        self.assertEqual(kwargs, {
            "prompt": "cat", "height": 768.0, "width": 1024.0,
            "num_steps": 8.0, "seed": 0, "randomize_seed": True, "lora_strength": 0.7,
        })
        self.assertIs(type(kwargs["height"]), float)
        self.assertIs(type(kwargs["width"]), float)
        self.assertIs(type(kwargs["num_steps"]), float)
        self.assertIs(type(kwargs["seed"]), int)
        self.assertIs(type(kwargs["randomize_seed"]), bool)

    def test_fixed_seed_and_extra_parameters(self):
        request = {**self.request, "seed": 42,
                   "extra_body": {"steps": 10, "lora_strength": 0.8}}
        kwargs = build_kwargs(self.profile, request)
        self.assertEqual(kwargs["seed"], 42)
        self.assertFalse(kwargs["randomize_seed"])
        self.assertEqual(kwargs["num_steps"], 10.0)
        self.assertEqual(kwargs["lora_strength"], 0.8)

    def test_validation(self):
        for extra in ({"unknown": 1}, {"steps": True}, {"steps": 0},
                      {"lora_strength": "bad"}, {"lora_strength": 2}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                build_kwargs(self.profile, {**self.request, "extra_body": extra})
        with self.assertRaises(ValueError):
            build_kwargs(self.profile, {**self.request, "seed": -2})
        with self.assertRaises(ValueError):
            build_kwargs(self.profile, self.request, ["input.png"])

    def test_upstream_predict_image_first_result_and_token(self):
        image_bytes = b"fake-png"
        with patch("gateway.moss_image.Client") as client, \\
             patch("gateway.moss_image.predict_logged", return_value=("/tmp/result.png", 91)) as predict, \\
             patch("gateway.moss_image.result_bytes", return_value=image_bytes) as result_bytes:
            output = run(self.profile, self.request, "hf_USER_TOKEN")
        self.assertEqual(output, image_bytes)
        self.assertEqual(client.call_args.kwargs["token"], "hf_USER_TOKEN")
        self.assertEqual(predict.call_args.kwargs["api_name"], "/generate")
        self.assertEqual(predict.call_args.kwargs["kwargs"]["randomize_seed"], True)
        result_bytes.assert_called_once_with("/tmp/result.png", self.profile)
        client.return_value.close.assert_called_once()

    def test_result_contract_rejects_mismatch(self):
        with patch("gateway.moss_image.Client"), \\
             patch("gateway.moss_image.predict_logged", return_value=("/tmp/result.png",)):
            with self.assertRaises(ValueError):
                run(self.profile, self.request, None)


if __name__ == "__main__":
    unittest.main()
