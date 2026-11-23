"""Two-stage image reference UI: preview does not select, confirm adds a real file."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from gateway import reference_confirmation as confirm


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.dir = TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        patched = patch.object(confirm, "CACHE_DIR", Path(self.dir.name))
        patched.start()
        self.addCleanup(patched.stop)
        self.reference = {
            "id": "portrait-1", "kind": "人像", "title": "短发写真",
            "url": "https://raw.githubusercontent.com/anhao/ai-pose-gallery-zh/main/img/portrait/a.webp",
        }

    def test_click_only_previews_no_download(self):
        with patch.object(confirm, "download_public_image") as download:
            candidate, url, hint = confirm.preview_candidate([self.reference], 0)
        self.assertEqual(candidate, self.reference)
        self.assertEqual(url, self.reference["url"])
        self.assertIn("未加入", hint)
        download.assert_not_called()

    def test_confirm_adds_file_to_same_upload_list(self):
        with patch.object(confirm, "load_references", return_value=[self.reference]):
            with patch.object(confirm, "download_public_image", return_value=b"image") as download:
                files, entries, hint = confirm.confirm_reference(self.reference, ["manual.png"], [])
        self.assertEqual(len(files), 2)
        self.assertEqual(files[0], "manual.png")
        self.assertEqual(Path(files[1]).read_bytes(), b"image")
        self.assertEqual(entries[0]["path"], files[1])
        self.assertIn("2/4", hint)
        download.assert_called_once_with(self.reference["url"])

    def test_duplicate_ref_does_not_download_twice(self):
        with patch.object(confirm, "load_references", return_value=[self.reference]):
            with patch.object(confirm, "download_public_image", return_value=b"image") as download:
                files, entries, _ = confirm.confirm_reference(self.reference, [], [])
                again, entries2, hint = confirm.confirm_reference(self.reference, files, entries)
        self.assertEqual(files, again)
        self.assertEqual(entries, entries2)
        self.assertEqual(download.call_count, 1)
        self.assertIn("重复", hint)

    def test_four_file_limit(self):
        with patch.object(confirm, "load_references", return_value=[self.reference]):
            with self.assertRaises(ValueError):
                confirm.confirm_reference(self.reference, ["a", "b", "c", "d"], [])

    def test_clear_only_gallery_files(self):
        with patch.object(confirm, "load_references", return_value=[self.reference]):
            with patch.object(confirm, "download_public_image", return_value=b"image"):
                files, entries, _ = confirm.confirm_reference(self.reference, ["manual.png"], [])
        remaining, confirmed, hint = confirm.clear_gallery_files(files, entries)
        self.assertEqual(remaining, ["manual.png"])
        self.assertEqual(confirmed, [])
        self.assertIn("保留", hint)

    def test_invalid_gallery_reference_never_downloaded(self):
        forged = {**self.reference, "url": "https://untrusted.example/private.jpg"}
        with patch.object(confirm, "load_references", return_value=[self.reference]):
            with patch.object(confirm, "download_public_image") as download:
                with self.assertRaises(ValueError):
                    confirm.confirm_reference(forged, [], [])
        download.assert_not_called()

if __name__ == "__main__":
    unittest.main()
