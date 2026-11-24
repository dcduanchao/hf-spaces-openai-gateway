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

    def test_poll_lifecycle(self):
        from gateway.video_studio import poll_video
        task_id = "video_" + "a"*32
        for state in ("queued", "in_progress", "completed", "failed"):
            with self.subTest(state=state), patch("gateway.video_studio.refresh_video",
                    return_value=({"status": state}, "result.mp4" if state == "completed" else None)):
                record, video, timer, button = poll_video(task_id)
                self.assertEqual(record["status"], state)
                terminal = state in ("completed", "failed")
                self.assertEqual(timer["active"], not terminal)
                self.assertEqual(button["interactive"], terminal)
                if state == "completed":
                    self.assertEqual(video, "result.mp4")

    def test_submit_starts_timer(self):
        from gateway.video_studio import start_video_poll
        with patch("gateway.video_studio.submit_video", return_value=("video_" + "a"*32, {"status": "queued"}, None)):
            result = start_video_poll()
            self.assertTrue(result[3]["active"])
            self.assertFalse(result[4]["interactive"])
