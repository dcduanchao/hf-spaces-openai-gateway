import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from gateway.wan2_2_i2v_v3 import build_kwargs, run, unpack_result, SPACE_ID, API_NAME

class WanVideoTests(unittest.TestCase):
    def test_one_and_two_images(self):
        with patch("gateway.wan2_2_i2v_v3.handle_file", side_effect=lambda p: p):
            one = build_kwargs({"prompt": "Wind in trees"}, ["a.png"])
            two = build_kwargs({"prompt": "Wind in trees", "seed": 7, "seconds": "4"}, ["a.png", "b.png"])
        self.assertEqual(len(one), 18)
        self.assertIsNone(one["last_image"])
        self.assertIs(type(one["frame_multiplier"]), int)
        self.assertTrue(one["randomize_seed"])
        self.assertEqual(two["last_image"], "b.png")
        self.assertEqual(two["seed"], 7)
        self.assertFalse(two["randomize_seed"])
        self.assertEqual(two["duration_seconds"], 4.0)

    def test_invalid_parameters(self):
        for payload in (
            {"seconds": 21}, {"seed": True}, {"extra_body": {"steps": 0}},
            {"extra_body": {"frame_multiplier": "16"}}, {"extra_body": {"unknown": 1}},
            {"extra_body": {"guidance_scale": float("nan")}},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                build_kwargs({"prompt": "Wind", **payload}, ["a.png"])
        with self.assertRaises(ValueError):
            build_kwargs({"prompt": "Wind"}, [])

    def test_submit_and_download(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "result.mp4"
            raw = b"\x00\x00\x00\x18ftypisom"
            path.write_bytes(raw)
            with patch("gateway.wan2_2_i2v_v3.Client") as client, patch(
                "gateway.wan2_2_i2v_v3.handle_file", side_effect=lambda p: p
            ):
                client.return_value.submit.return_value.result.return_value = (None, str(path), 42)
                self.assertEqual(run({"prompt": "Wind"}, "hf_TEST", ["a.png"]), (raw, 42))
                self.assertEqual(client.call_args.args[0], SPACE_ID)
                self.assertEqual(client.return_value.submit.call_args.kwargs["api_name"], API_NAME)
                client.return_value.close.assert_called_once()

    def test_invalid_result(self):
        for result in (None, (None, None, 42), (None, "missing.mp4", 42)):
            with self.assertRaises(RuntimeError):
                unpack_result(result)
