"""Regression tests for generated prompts and true upload order."""
import unittest
from unittest.mock import patch

from gateway.recipe_prompt import build_edit_prompt, STYLES
from gateway import ui


class RecipePromptTests(unittest.TestCase):
    def setUp(self):
        self.upload = "/tmp/manual-reference.png"
        self.face = "/tmp/face-reference.png"
        self.outfit = "/tmp/outfit-reference.png"
        self.pose = "/tmp/pose-reference.png"
        self.confirmed = [
            {"path": self.face, "kind": "人像", "title": "短发写真", "detail": "短发"},
            {"path": self.outfit, "kind": "服装", "title": "蓝色大衣", "detail": ""},
            {"path": self.pose, "kind": "姿势", "title": "站姿", "detail": "侧面"},
        ]

    def test_three_image_roles_and_style(self):
        text = build_edit_prompt([self.face, self.outfit, self.pose], self.confirmed)
        self.assertIn("提供的 3 张参考图", text)
        self.assertIn("图1中的人像", text)
        self.assertIn("图2展示的整套服装", text)
        self.assertIn("图3中的人体动作", text)
        self.assertIn("穿着图2的服装", text)
        self.assertIn("85mm", text)

    def test_interleaved_uploaded_image_preserves_numbering(self):
        text = build_edit_prompt([self.upload, self.outfit, self.face, self.pose], self.confirmed)
        self.assertIn("图1是用户上传的参考图", text)
        self.assertIn("图2展示的整套服装", text)
        self.assertIn("图3中的人像", text)
        self.assertIn("图4中的人体动作", text)
        self.assertIn("穿着图2的服装", text)

    def test_no_reference(self):
        self.assertEqual(build_edit_prompt([], []), "")

    def test_all_style_presets_supported(self):
        for style in STYLES:
            with self.subTest(style=style):
                prompt = build_edit_prompt([self.face], self.confirmed, style)
                self.assertIn("图1中的人像", prompt)
        neutral = build_edit_prompt([self.face], self.confirmed, "无预设（跟随参考图）")
        self.assertNotIn("• 风格：", neutral)

    def test_no_lost_manual_edit(self):
        result = ui.sync_edit_prompt("图片编辑", [self.face], self.confirmed, "写实人像写真", False)
        self.assertIsNotNone(result)
        # Gradio skip is an update object and must not contain a replacement text.
        self.assertNotIsInstance(result, str)

    def test_auto_fills_prompt_in_edit_mode(self):
        text = ui.sync_edit_prompt("图片编辑", [self.face], self.confirmed, "写实人像写真", True)
        self.assertIn("人物", text)
        skipped = ui.sync_edit_prompt("文生图", [self.face], self.confirmed, "写实人像写真", True)
        self.assertNotIsInstance(skipped, str)

    def test_reset_restores_automatic_mode(self):
        text, enabled = ui.reset_edit_prompt("图片编辑", [self.face], self.confirmed, "写实人像写真")
        self.assertTrue(enabled)
        self.assertIn("人物", text)


if __name__ == "__main__":
    unittest.main()
