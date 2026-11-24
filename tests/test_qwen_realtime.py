"""Qwen 2-step /generate contract, size-tier mapping and UI compatibility."""
import base64
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from gateway import history
from gateway.main import app
from gateway.models import load_profiles
from gateway.qwen_realtime import (
    BASE_SIZES, RESOLUTIONS, MAX_REFERENCES, build_kwargs,
    choose_parameters, preset_sizes, run,
)
from gateway.ui import SIZE_LABEL, size_note, model_size_choices, default_size_value, model_space_link, selected_model


def tiny_png():
    buf = io.BytesIO()
    Image.new("RGB", (6, 6), "red").save(buf, "PNG")
    return buf.getvalue()


class QwenRealtimeTests(unittest.TestCase):
    def setUp(self):
        self.model = load_profiles()["qwen-image-2.1-2"]

    def test_catalog_and_legacy_sizes_unchanged(self):
        models = load_profiles()
        self.assertEqual(len(models), 4)
        self.assertEqual(models["qwen-image-2.1"].allowed_sizes, [
            "1024x1024", "1344x768", "768x1344", "1184x864", "864x1184",
            "2048x2048", "2688x1536", "1536x2688", "2368x1728", "1728x2368",
        ])
        self.assertEqual(len(models["qwen-image-2.1-6"].allowed_sizes), 9)
        self.assertEqual(len(models["z-image-turbo"].allowed_sizes), 7)
        self.assertEqual(self.model.api_name, "/generate")
        self.assertEqual(self.model.space_id, "linoyts/qwen-image-2-1-realtime")
        self.assertEqual(self.model.default_size, "1024x1024")
        self.assertEqual(self.model.modes, ["generations", "edits"])

    def test_all_three_bands_and_seven_ratios(self):
        self.assertEqual(len(BASE_SIZES), 7)
        self.assertEqual(RESOLUTIONS, (512, 1024, 1536))
        self.assertEqual(len(preset_sizes()), 21)
        self.assertTrue(all(type(res) is int for res in RESOLUTIONS))
        for size, band_and_ratio in preset_sizes().items():
            self.assertIn(size, self.model.allowed_sizes)
            w, h = map(int, size.split("x"))
            self.assertEqual(choose_parameters(w, h), band_and_ratio)

    def test_custom_size_maps_to_nearest_ratio_and_area(self):
        self.assertEqual(choose_parameters(1024,1536), (1024, "2:3"))
        self.assertEqual(choose_parameters(1024,1024), (1024, "1:1"))

    def test_generate_kwargs_without_enhancement_or_realtime_session(self):
        kw = build_kwargs(self.model, {"prompt": "cat", "size": "1024x1024", "seed": 6})
        self.assertEqual(kw, {"prompt": "cat", "input_images": [], "seed": 6,
                              "resolution": 1024, "aspect_ratio": "1:1"})
        self.assertNotIn("enhance", kw)
        self.assertNotIn("steps", kw)

    def test_four_ordered_references_and_invalid_fifth(self):
        paths = ["a.png","b.png","c.png","d.png"]
        with patch("gateway.qwen_realtime.handle_file", side_effect=lambda p: {"path":p}):
            kw = build_kwargs(self.model, {"prompt":"portrait", "size":"896x1200", "seed":7}, paths)
        self.assertEqual([item["image"]["path"] for item in kw["input_images"]], paths)
        self.assertEqual(kw["resolution"], 1024)
        self.assertEqual(kw["aspect_ratio"], "3:4")
        with self.assertRaises(ValueError):
            build_kwargs(self.model, {"prompt":"cat"}, paths + ["e.png"])

    def test_unknown_extra_rejected_and_seed_randomization(self):
        with self.assertRaises(ValueError):
            build_kwargs(self.model, {"prompt":"cat", "extra_body":{"steps":2}})
        with patch("gateway.qwen_realtime.secrets.randbelow", return_value=123) as rnd:
            kw = build_kwargs(self.model, {"prompt":"cat", "seed":-1})
        self.assertEqual(kw["seed"],123)
        rnd.assert_called_once()

    def test_run_calls_single_generate_and_preserves_token(self):
        with patch("gateway.qwen_realtime.Client") as Client, patch(
            "gateway.qwen_realtime.predict_logged", return_value="fake-result"
        ) as predict, patch("gateway.qwen_realtime.result_bytes", return_value=tiny_png()):
            result = run(self.model, {"prompt":"cat"}, "hf_TEST_TOKEN")
        self.assertEqual(result, tiny_png())
        self.assertEqual(Client.call_args.kwargs["token"], "hf_TEST_TOKEN")
        self.assertEqual(predict.call_args.kwargs["api_name"], "/generate")
        self.assertEqual(predict.call_args.kwargs["kwargs"]["resolution"], 1024)
        Client.return_value.close.assert_called_once()

    def test_ui_standard_display_ratio_and_model_specific_choices(self):
        self.assertEqual(SIZE_LABEL("1024x1024"), "1024×1024（1:1）")
        self.assertEqual(SIZE_LABEL("1344x768"), "1344×768（16:9）")
        self.assertEqual(SIZE_LABEL("1184x864"), "1184×864（4:3）")
        choices = model_size_choices(self.model)
        self.assertEqual(choices[0], ("Auto", "auto"))
        self.assertEqual(len(choices), 8)
        self.assertEqual(choices[1:], [
            (f"{w}×{h}（{ratio}）", f"{w}x{h}")
            for ratio, (w, h) in BASE_SIZES.items()
        ])
        self.assertEqual(default_size_value(self.model), "auto")
        self.assertEqual(default_size_value(load_profiles()["qwen-image-2.1"]), "1024x1024")
        self.assertEqual(len(model_size_choices(load_profiles()["qwen-image-2.1"])), 10)
        self.assertIn("7 种固定宽高比", size_note(self.model))
        self.assertIn("10 种预设", size_note(load_profiles()["qwen-image-2.1"]))

    def test_selected_model_space_links_open_new_tab(self):
        for profile in load_profiles().values():
            html = model_space_link(profile)
            self.assertIn(f"https://huggingface.co/spaces/{profile.space_id}", html)
            self.assertIn('target="_blank"', html)
            self.assertIn('rel="noopener noreferrer"', html)
            updates = selected_model(profile.id)
            self.assertEqual(len(updates), 7)
            self.assertIn(profile.space_id, updates[2]["value"])
        self.assertEqual(model_space_link(None), "")
        self.assertEqual(selected_model("not-a-model")[2]["value"], "")

    def test_auto_and_fixed_reference_dimensions_map_to_upstream(self):
        auto = build_kwargs(self.model, {"prompt": "cat", "size": "auto", "seed": 9})
        self.assertEqual((auto["resolution"], auto["aspect_ratio"]), (1536, "Auto"))
        self.assertIs(type(auto["resolution"]), int)
        for ratio, (w, h) in BASE_SIZES.items():
            kw = build_kwargs(self.model, {"prompt": "cat", "size": f"{w}x{h}", "seed": 9})
            self.assertEqual((kw["resolution"], kw["aspect_ratio"]), (1536, ratio))
            self.assertIs(type(kw["resolution"]), int)

    def test_auto_size_accepted_by_public_api(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / ".history_bucket_ready").write_text("test")
            with patch.object(history, "ROOT", root):
                with patch("gateway.main.invoke", return_value=tiny_png()) as invoke:
                    response = TestClient(app).post("/v1/images/generations", json={
                        "model": "qwen-image-2.1-2", "prompt": "cat",
                        "size": "auto", "response_format": "b64_json",
                    })
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(invoke.call_args.args[1]["size"], "auto")

    def test_public_model_and_edit_route_keep_history(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/".history_bucket_ready").write_text("test")
            with patch.object(history, "ROOT", root):
                client = TestClient(app)
                models = client.get("/v1/models").json()["data"]
                self.assertIn("qwen-image-2.1-2", {x["id"] for x in models})
                with patch("gateway.main.invoke", return_value=tiny_png()) as invoke:
                    res = client.post("/v1/images/edits", headers={"X-HF-Token":"hf_TEST_TOKEN"},
                        data={"model":"qwen-image-2.1-2", "prompt":"edit",
                              "size":"1024x1024", "response_format":"b64_json"},
                        files={"image":("ref.png",tiny_png(),"image/png")})
                self.assertEqual(res.status_code, 200, res.text)
                self.assertEqual(base64.b64decode(res.json()["data"][0]["b64_json"]), tiny_png())
                self.assertEqual(invoke.call_args.args[0].adapter, "qwen_realtime")
                self.assertEqual(invoke.call_args.args[2], "hf_TEST_TOKEN")
                self.assertEqual(len(invoke.call_args.kwargs["image_paths"]), 1)
                self.assertEqual(history.list_records()["total"],1)


if __name__ == "__main__":
    unittest.main()
