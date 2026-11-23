"""Offline tests: private Bucket history persistence and admin authorization."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from gateway.main import app
from gateway import history


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / ".history_bucket_ready").write_text("test")
        patched = patch.object(history, "ROOT", self.root)
        patched.start()
        self.addCleanup(patched.stop)
        self.client = TestClient(app)

    def test_persistent_record_and_image(self):
        rid = history.new_record("z-image-turbo", "generations", "a test prompt",
                                 {"seed": 2, "token": "hf_should_not_store"})
        filename = history.store_image(rid, b"fake-image")
        history.finish_record(rid, "success", 333)
        data = history.get_record(rid)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["images"], [filename])
        self.assertEqual(data["duration_ms"], 333)
        self.assertNotIn("hf_should_not_store", json.dumps(data))
        self.assertEqual(history.get_image(rid, filename).read_bytes(), b"fake-image")
        self.assertEqual(history.list_records()["total"], 1)

    def test_admin_rejects_without_config(self):
        with patch.dict("os.environ", {}, clear=True):
            res = self.client.get("/admin/api/history")
        self.assertEqual(res.status_code, 503)

    def test_admin_access_and_deletion(self):
        rid = history.new_record("qwen-image-2.1", "generations", "cat", {})
        history.finish_record(rid, "success", 12)
        with patch.dict("os.environ", {"ADMIN_API_KEY": "long-secret"}):
            noauth = self.client.get("/admin/api/history")
            self.assertEqual(noauth.status_code, 401)
            headers = {"X-Admin-Key": "long-secret"}
            records = self.client.get("/admin/api/history", headers=headers)
            self.assertEqual(records.status_code, 200)
            self.assertEqual(records.json()["total"], 1)
            detail = self.client.get("/admin/api/history/" + rid, headers=headers)
            self.assertEqual(detail.json()["prompt"], "cat")
            deleted = self.client.delete("/admin/api/history/" + rid, headers=headers)
            self.assertEqual(deleted.status_code, 200)
            self.assertEqual(self.client.get("/admin/api/history", headers=headers).json()["total"], 0)

    def test_missing_bucket_never_uses_ephemeral_storage(self):
        (self.root / ".history_bucket_ready").unlink()
        with self.assertRaises(RuntimeError):
            history.new_record("z-image-turbo", "generations", "cat", {})

    def test_admin_modal_ui_is_present_without_exposing_records(self):
        response = self.client.get("/admin/")
        self.assertEqual(response.status_code, 200)
        html = response.text
        self.assertIn('id="record-modal"', html)
        self.assertIn('id="lightbox"', html)
        self.assertIn("modal.showModal()", html)
        self.assertIn('id="output-images"', html)
        self.assertIn('id="input-images"', html)
        self.assertNotIn("ADMIN_API_KEY_VALUE", html)
        self.assertIn("no-store", response.headers["cache-control"])

    def test_unauthorized_image_download(self):
        rid = history.new_record("z-image-turbo", "generations", "cat", {})
        name = history.store_image(rid, b"sample")
        with patch.dict("os.environ", {"ADMIN_API_KEY": "secret"}):
            url = f"/admin/api/history/{rid}/images/{name}"
            self.assertEqual(self.client.get(url).status_code, 401)
            self.assertEqual(self.client.get(url, headers={"X-Admin-Key": "secret"}).content, b"sample")


if __name__ == "__main__":
    unittest.main()
