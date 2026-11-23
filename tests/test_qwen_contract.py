"""Offline regression tests based on Qwen-Image-2.1 published API signatures."""
import unittest
from unittest.mock import patch
from gateway.models import load_profiles
from gateway.qwen_image import build_requests

class QwenContractTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_profiles()["qwen-image-2.1"]
        self.payload = {"model": "qwen-image-2.1", "prompt": "a cat", "size": "1024x1024", "seed": -1, "extra_body": {}}

    def test_documented_request_keys(self):
        prep, gen = build_requests(self.profile, self.payload)
        self.assertEqual(set(prep), {"input_images", "original_prompt", "enable_extend", "custom_size", "quality", "seed", "randomize_seed"})
        self.assertEqual(set(gen), {"original_prompt", "enable_extend", "custom_size", "log_dir", "seed", "height", "width", "negative_prompt"})
        self.assertEqual(prep["input_images"], [])
        self.assertGreaterEqual(prep["seed"], 0)
        self.assertFalse(prep["randomize_seed"])
        self.assertFalse(prep["enable_extend"])
        self.assertFalse(gen["enable_extend"])

    def test_seed_preserved(self):
        req = dict(self.payload, seed=42)
        prep, gen = build_requests(self.profile, req)
        self.assertEqual(prep["seed"], 42)
        self.assertEqual(gen["seed"], 42)

    def test_negative_seed_rejected(self):
        with self.assertRaises(ValueError):
            build_requests(self.profile, dict(self.payload, seed=-2))

    def test_extra_fields(self):
        req = dict(self.payload, extra_body={"quality": "quality", "negative_prompt": "blurry"})
        prep, gen = build_requests(self.profile, req)
        self.assertEqual(prep["quality"], "quality")
        self.assertFalse(prep["enable_extend"])
        self.assertEqual(gen["negative_prompt"], "blurry")

    def test_prompt_extension_cannot_be_enabled(self):
        req = dict(self.payload, extra_body={"enable_extend": True})
        with self.assertRaises(ValueError):
            build_requests(self.profile, req)

    def test_gallery_structure(self):
        with patch("gateway.qwen_image.handle_file", side_effect=lambda p: {"path": p}):
            prep, _ = build_requests(self.profile, self.payload, ["/tmp/reference.png"])
        self.assertEqual(prep["input_images"], [{"image": {"path": "/tmp/reference.png"}, "caption": None}])

if __name__ == "__main__":
    unittest.main()
