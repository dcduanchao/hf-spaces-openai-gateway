"""Tests for source-attributed, SFW-only pose selector."""
import unittest
from unittest.mock import patch

from gateway.pose_gallery import (
    safe_pose_path, public_asset_url, load_pose_catalog,
    search_poses, gallery_items, add_selected_pose,
)


class PoseGalleryTests(unittest.TestCase):
    def test_path_restrictions(self):
        for path in ("../secrets", "img/../secret.png", "https://bad.example/file.png",
                     "thumbs/p.png", "img/a.svg"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                safe_pose_path(path)
        self.assertEqual(public_asset_url("img/full_normal_P001_front.png"),
                         "https://raw.githubusercontent.com/anhao/ai-pose-gallery-zh/main/img/full_normal_P001_front.png")

    def test_sfw_filter_and_search(self):
        rows = [
            {"id": "a", "体系": "全身单人・标准系", "分类": "站姿",
             "姿势名称": "普通站姿", "角度": "正面",
             "原图": "img/standard_P001.png", "检索": "普通 站姿"},
            {"id": "b", "体系": "全身单人・性感系", "分类": "站姿",
             "姿势名称": "性感站姿", "角度": "正面",
             "原图": "img/sexy_P001.png"},
            {"id": "c", "体系": "双人缠绵", "分类": "互动",
             "姿势名称": "拥抱", "角度": "正面",
             "原图": "img/pair_P001.png"}
        ]
        from gateway import pose_gallery
        response = type("Response", (), {
            "raise_for_status": lambda self: None,
            "json": lambda self: rows
        })()
        with patch.object(pose_gallery, "_cache", []):
            with patch.object(pose_gallery, "_expires_at", 0):
                with patch("gateway.pose_gallery.httpx.get", return_value=response):
                    found = load_pose_catalog()
                    self.assertEqual(len(found), 1)
                    self.assertEqual(found[0]["title"], "普通站姿")
                    searched = search_poses("站姿", "站姿")
                    self.assertEqual(len(searched), 1)
                    self.assertEqual(len(gallery_items(searched)), 1)

    def test_selection_limit_and_deduplication(self):
        items = [{"url": f"https://example.org/{i}.png"} for i in range(5)]
        chosen = []
        for i in range(4):
            chosen, _ = add_selected_pose(items, chosen, i)
        self.assertEqual(len(chosen), 4)
        again, _ = add_selected_pose(items, chosen, 0)
        self.assertEqual(len(again), 4)
        refused, message = add_selected_pose(items, chosen, 4)
        self.assertEqual(len(refused), 4)
        self.assertIn("4", message)


if __name__ == "__main__":
    unittest.main()
