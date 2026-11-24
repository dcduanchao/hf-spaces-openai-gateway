import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from fastapi.testclient import TestClient
from gateway.main import app
from gateway import history, video_tasks

class VideoApiTests(unittest.TestCase):
    def test_create_and_validation(self):
        buf = io.BytesIO()
        Image.new("RGB", (16, 16)).save(buf, "PNG")
        with patch("gateway.video_tasks.create", return_value={"id": "video_test", "status": "queued"}) as create:
            client = TestClient(app)
            data = {"model": "wan2.2-i2v-v3", "prompt": "Wind in trees", "seconds": "4"}
            files = {"input_reference": ("input.png", buf.getvalue(), "image/png")}
            response = client.post("/v1/videos", data=data, files=files)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["status"], "queued")
            self.assertEqual(create.call_args.args[0]["seconds"], 4)
            self.assertEqual(client.post("/v1/videos", data={**data, "seconds": "99"}, files=files).status_code, 400)

    def test_persistence_and_download(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(history, "ROOT", Path(tmp)):
            (Path(tmp) / ".history_bucket_ready").touch()
            video_id = "video_" + "a" * 32
            path = video_tasks.folder(video_id)
            path.mkdir(parents=True)
            video_tasks.write(path, {"id": video_id, "status": "completed", "progress": 100})
            (path / "output.mp4").write_bytes(b"test-video")
            client = TestClient(app)
            self.assertEqual(client.get("/v1/videos/" + video_id).json()["status"], "completed")
            self.assertEqual(client.get("/v1/videos/" + video_id + "/content").content, b"test-video")
            self.assertTrue(client.delete("/v1/videos/" + video_id).json()["deleted"])
            self.assertEqual(client.get("/v1/videos/" + video_id).status_code, 404)
