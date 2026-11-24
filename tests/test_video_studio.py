import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from gateway.video_studio import submit_video, refresh_video

class VideoStudioTests(unittest.TestCase):
    def test_submit_uses_loopback(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "input.png"
            image.write_bytes(b"test")
            with patch("gateway.video_studio.httpx.Client") as client:
                response = client.return_value.__enter__.return_value.post.return_value
                response.status_code = 200
                response.json.return_value = {"id": "video_" + "a"*32, "status": "queued"}
                result = submit_video(str(image), None, "wind", 4, 6, 16, -1, "hf_TEST", "{}")
                self.assertEqual(result[0], "video_" + "a"*32)
                post = client.return_value.__enter__.return_value.post
                self.assertEqual(post.call_args.args[0], "/v1/videos")
                self.assertEqual(post.call_args.kwargs["headers"]["X-HF-Token"], "hf_TEST")

    def test_pending_status(self):
        with patch("gateway.video_studio.httpx.Client") as client:
            response = client.return_value.__enter__.return_value.get.return_value
            response.status_code = 200
            response.json.return_value = {"status": "in_progress"}
            self.assertEqual(refresh_video("video_" + "a"*32), ({"status": "in_progress"}, None))
