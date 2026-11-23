"""Offline contract tests for the public Z-Image-Turbo Gradio API."""
import unittest
from gateway.models import load_profiles
from gateway.z_image import build_kwargs


class ZImageContractTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_profiles()["z-image-turbo"]
        self.request = {"model": "z-image-turbo", "prompt": "a cat", "size": "1024x1024", "seed": -1}

    def test_six_arguments(self):
        args = build_kwargs(self.profile, self.request)
        self.assertEqual(set(args), {"prompt", "height", "width", "num_inference_steps", "seed", "randomize_seed"})
        self.assertEqual(args["num_inference_steps"], 8)
        self.assertEqual(args["height"], 1024)
        self.assertGreaterEqual(args["seed"], 0)
        self.assertFalse(args["randomize_seed"])

    def test_seed_preserved(self):
        self.assertEqual(build_kwargs(self.profile, dict(self.request, seed=42))["seed"], 42)

    def test_steps_override(self):
        args = build_kwargs(self.profile, dict(self.request, extra_body={"steps": 12}))
        self.assertEqual(args["num_inference_steps"], 12)

    def test_unknown_parameters_rejected(self):
        with self.assertRaises(ValueError):
            build_kwargs(self.profile, dict(self.request, extra_body={"unsupported": True}))

    def test_edit_not_supported(self):
        with self.assertRaises(ValueError):
            build_kwargs(self.profile, self.request, ["/tmp/test.png"])


if __name__ == "__main__":
    unittest.main()
