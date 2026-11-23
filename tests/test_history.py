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

    def test_default_page_size_twenty(self):
        for i in range(23):
            history.new_record("qwen-image-2.1", "generations", f"prompt {i}", {})
        with patch.dict("os.environ", {"ADMIN_API_KEY": "secret"}):
            first = self.client.get("/admin/api/history", headers={"X-Admin-Key": "secret"})
            self.assertEqual(first.status_code, 200)
            self.assertEqual(first.json()["total"], 23)
            self.assertEqual(len(first.json()["items"]), 20)
            second = self.client.get("/admin/api/history?offset=20",
                                     headers={"X-Admin-Key": "secret"})
            self.assertEqual(len(second.json()["items"]), 3)

    def test_bulk_delete_requires_auth_and_confirmation_and_removes_images(self):
        first = history.new_record("qwen-image-2.1", "edits", "one", {})
        image = history.store_image(first, b"test-image")
        second = history.new_record("qwen-image-2.1", "edits", "two", {})
        unrelated = self.root / "unrelated.txt"
        unrelated.write_text("keep")
        with patch.dict("os.environ", {"ADMIN_API_KEY": "secret"}):
            url = "/admin/api/history"
            self.assertEqual(self.client.delete(url).status_code, 401)
            headers = {"X-Admin-Key": "secret"}
            self.assertEqual(self.client.delete(url, headers=headers).status_code, 400)
            headers["X-Confirm-Delete"] = "DELETE_ALL_HISTORY"
            result = self.client.delete(url, headers=headers)
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(result.json(), {"deleted": 2, "failed": 0})
            self.assertEqual(self.client.get(url, headers=headers).json()["total"], 0)
            self.assertEqual(self.client.get(
                f"/admin/api/history/{first}/images/{image}", headers=headers).status_code, 404)
            self.assertTrue(unrelated.is_file())

    def test_bulk_delete_unmounted_bucket_is_safe(self):
        (self.root / ".history_bucket_ready").unlink()
        with patch.dict("os.environ", {"ADMIN_API_KEY": "secret"}):
            result = self.client.delete("/admin/api/history", headers={
                "X-Admin-Key": "secret", "X-Confirm-Delete": "DELETE_ALL_HISTORY"})
        self.assertEqual(result.status_code, 503)

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
        self.assertIn("PAGE_SIZE=20", html)
        self.assertIn("prompt-preview", html)
        self.assertIn('id="delete-all"', html)
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
