"""Reference library regression: portrait, clothing and pose integration."""
import unittest
from unittest.mock import patch
from gateway import reference_gallery as gallery


class ReferenceGalleryTests(unittest.TestCase):
    def setUp(self):
        gallery._CACHE.clear()

    def test_portrait_schema_and_search(self):
        records = [
            {"id": "face1", "分区": "正脸参考", "发长": "短发",
             "标题": "短发写真", "原图": "img/portrait/face1.webp", "检索": "短发 人脸"},
        ]
        with patch("gateway.reference_gallery.httpx.get") as get:
            get.return_value.json.return_value = records
            items = gallery.search_references("人像", "短发")
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["kind"], "人像")
            self.assertEqual(items[0]["category"], "正脸参考")
            self.assertEqual(gallery.categories("人像"), ["全部", "正脸参考"])
            self.assertTrue(items[0]["url"].endswith("/img/portrait/face1.webp"))
            get.assert_called_once()

    def test_clothing_safe_category_filter(self):
        records = [
            {"id": "coat1", "分类": "冬季穿搭", "服装名称": "灰色大衣",
             "原图": "img/costume/coat1.webp", "检索": "灰色 大衣"},
            {"id": "ling1", "分类": "内衣・睡衣", "服装名称": "内衣",
             "原图": "img/costume/ling1.webp"},
            {"id": "bad", "分类": "冬季穿搭", "服装名称": "something",
             "原图": "https://internal.example/a.png"},
        ]
        with patch("gateway.reference_gallery.httpx.get") as get:
            get.return_value.json.return_value = records
            items = gallery.search_references("服装")
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["title"], "灰色大衣")

    def test_select_across_three_collections(self):
        rows = [
            {"id": "a", "kind": "人像", "title": "发型", "url": "https://example.com/face.png"},
            {"id": "b", "kind": "服装", "title": "外套", "url": "https://example.com/coat.png"},
            {"id": "c", "kind": "姿势", "title": "站姿", "url": "https://example.com/pose.png"},
        ]
        selected = []
        for i in range(3):
            selected, _ = gallery.choose_reference(rows, selected, i)
        self.assertEqual([row["kind"] for row in selected], ["人像", "服装", "姿势"])
        repeat, _ = gallery.choose_reference(rows, selected, 1)
        self.assertEqual(len(repeat), 3)
        self.assertIn("3", gallery.format_selection(selected))
        self.assertEqual(len(gallery.gallery_items(rows)), 3)

    def test_auto_preview_has_no_search_requirement(self):
        from gateway import ui
        records = [{
            "id": "portrait1", "kind": "人像", "category": "正脸参考",
            "title": "短发", "detail": "短发", "keywords": "短发",
            "url": "https://example.com/portrait.png"
        }]
        with patch.object(ui, "reference_categories", return_value=["全部", "正脸参考"]):
            with patch.object(ui, "search_references", return_value=records) as search:
                category, gallery_items, results, hint = ui.default_reference_preview("人像", [])
        search.assert_called_once_with("人像", "", "全部", limit=24)
        self.assertEqual(len(gallery_items), 1)
        self.assertEqual(results, records)
        self.assertIn("自动预览", hint)

    def test_switch_library_preserves_selection(self):
        from gateway import ui
        selected = [{"kind": "人像", "title": "短发", "url": "https://example.com/a.png"}]
        rows = [{"id": "outfit1", "kind": "服装", "category": "冬季穿搭",
                 "title": "大衣", "detail": "", "keywords": "",
                 "url": "https://example.com/b.png"}]
        with patch.object(ui, "reference_categories", return_value=["全部", "冬季穿搭"]):
            with patch.object(ui, "search_references", return_value=rows):
                _, items, found, hint = ui.default_reference_preview("服装", selected)
        self.assertEqual(found, rows)
        self.assertIn("人像", hint)
        self.assertIn("短发", hint)

    def test_reset_preview_clears_filter(self):
        from gateway import ui
        with patch.object(ui, "default_reference_preview", return_value=(
            {"choices": ["全部"], "value": "全部"}, [], [], "已重置")):
            updated = ui.reset_reference_preview("姿势", [])
        self.assertEqual(len(updated), 5)
        self.assertEqual(updated[-1], "已重置")

    def test_reject_untrusted_selection(self):
        with patch.object(gallery, "load_references", return_value=[
            {"url": "https://raw.githubusercontent.com/anhao/ai-pose-gallery-zh/main/img/portrait/a.webp"}]):
            with self.assertRaises(ValueError):
                gallery.validate_selection([{
                    "kind": "人像", "url": "https://evil.example/private.jpg"}])

    def test_allowed_selection(self):
        url = "https://raw.githubusercontent.com/anhao/ai-pose-gallery-zh/main/img/costume/a.webp"
        selected = [{"kind": "服装", "url": url, "title": "a", "id": "a"}]
        with patch.object(gallery, "load_references", return_value=[{"url": url}]):
            self.assertEqual(gallery.validate_selection(selected), selected)


if __name__ == "__main__":
    unittest.main()
