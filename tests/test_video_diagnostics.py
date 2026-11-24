import unittest
from unittest.mock import patch
from gateway.wan2_2_i2v_v3 import VideoUpstreamError, run

class VideoDiagnosticsTests(unittest.TestCase):
    def test_quota_and_secret_redaction(self):
        error = VideoUpstreamError("wait", RuntimeError("quota exceeded hf_SECRET"))
        self.assertEqual(error.code, "upstream_quota_exceeded")
        self.assertNotIn("hf_SECRET", str(error))

    def test_fps_error(self):
        error = VideoUpstreamError("wait", ValueError("frame_multiplier not in choices"))
        self.assertEqual(error.code, "upstream_fps_invalid")

    def test_init_failure(self):
        with patch("gateway.wan2_2_i2v_v3.handle_file", side_effect=lambda p: p), patch("gateway.wan2_2_i2v_v3.Client", side_effect=RuntimeError("bad schema hf_SECRET")):
            with self.assertRaises(VideoUpstreamError) as ctx:
                run({"prompt": "wind"}, None, ["a.png"])
        self.assertEqual(ctx.exception.phase, "client_init")
        self.assertNotIn("hf_SECRET", str(ctx.exception))
