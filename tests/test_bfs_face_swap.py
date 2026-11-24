"""BFS head-swap adapter + two-image API/Studio contract tests."""
import base64
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from gateway import history
from gateway.bfs_face_swap import (
    BASE, TURBO, DEFAULT_PROMPT, build_kwargs, image_after, image_pair, run,
)
from gateway.main import app
from gateway.models import load_profiles
from gateway.ui import generate_face_swap, model_space_link


def tiny_png():
    stream = io.BytesIO()
    Image.new("RGB", (6, 6), "red").save(stream, "PNG")
    return stream.getvalue()


class BfsFaceSwapTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_profiles()["bfs-face-swap"]
        self.payload = {"model": "bfs-face-swap", "prompt": DEFAULT_PROMPT,
                        "size": "1024x1024", "seed": -1, "extra_body": {}}
        self.paths = ["/tmp/reference.png", "/tmp/target.png"]

    def test_catalog_only_edits_and_space_link(self):
        self.assertEqual(self.profile.modes, ["edits"])
        self.assertEqual(self.profile.api_name, "/face_swap")
        self.assertEqual(self.profile.space_id, "hugging-apps/bfs-best-face-swap")
        self.assertIn(self.profile.space_id, model_space_link(self.profile))

    def test_two_images_and_nine_exact_named_parameters(self):
        with patch("gateway.bfs_face_swap.handle_file", side_effect=lambda p: {"path": p}):
            data = build_kwargs(self.profile, self.payload, self.paths, 6.0)
        self.assertEqual(set(data), {
            "reference_face", "target_image", "base_model", "resolution",
            "num_inference_steps", "lora_scale", "seed", "randomize_seed", "prompt",
        })
        self.assertEqual(data["reference_face"], {"path": self.paths[0]})
        self.assertEqual(data["target_image"], {"path": self.paths[1]})
        self.assertEqual(data["base_model"], TURBO)
        self.assertEqual(data["resolution"], 1536)
        self.assertIs(type(data["resolution"]), int)
        self.assertEqual(data["num_inference_steps"], 6.0)
        self.assertEqual(data["lora_scale"], 1.0)
        self.assertEqual(data["seed"], 0.0)
        self.assertIs(data["randomize_seed"], True)
        self.assertEqual(data["prompt"], DEFAULT_PROMPT)

    def test_two_model_options_fixed_seed_and_dynamic_steps(self):
        request = {**self.payload, "seed": 42, "extra_body": {"base_model": BASE}}
        with patch("gateway.bfs_face_swap.handle_file", side_effect=lambda p: p):
            data = build_kwargs(self.profile, request, self.paths, 30.0)
        self.assertEqual(data["base_model"], BASE)
        self.assertEqual(data["num_inference_steps"], 30.0)
        self.assertEqual(data["seed"], 42.0)
        self.assertIs(data["randomize_seed"], False)

    def test_reject_missing_images_and_bad_options(self):
        for paths in ([], [self.paths[0]], self.paths + ["/tmp/third.png"]):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                build_kwargs(self.profile, self.payload, paths, 6)
        for extra in ({"base_model": "invalid"}, {"steps": 25}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                build_kwargs(self.profile, {**self.payload, "extra_body": extra}, self.paths, 6)

    def test_after_image_is_selected(self):
        self.assertEqual(image_pair((("/tmp/before.png", "/tmp/after.png"), 25.0)),
                         ["/tmp/before.png", "/tmp/after.png"])
        self.assertEqual(image_after((("/tmp/before.png", "/tmp/after.png"), 25.0)),
                         "/tmp/after.png")
        for result in ((None, 2.0), (("/tmp/before.png", None), 2.0), ("bad", 2.0)):
            with self.subTest(result=result), self.assertRaises(RuntimeError):
                image_after(result)

    def test_calls_mode_change_and_face_swap(self):
        with (
            patch("gateway.bfs_face_swap.Client") as client,
            patch("gateway.bfs_face_swap.handle_file", side_effect=lambda p: p),
            patch("gateway.bfs_face_swap.predict_logged",
                  return_value=(("/tmp/before.png", "/tmp/after.png"), 10.0)) as predict,
            patch("gateway.bfs_face_swap.result_bytes", return_value=b"image") as output,
        ):
            client.return_value.predict.return_value = 6.0
            self.assertEqual(run(self.profile, self.payload, "hf_USER", self.paths), [b"image", b"image"])
        client.return_value.predict.assert_not_called()
        self.assertEqual(predict.call_args.kwargs["api_name"], "/face_swap")
        self.assertEqual(predict.call_args.kwargs["kwargs"]["num_inference_steps"], 6.0)
        self.assertEqual(output.call_count, 2)
        self.assertEqual([call.args[0] for call in output.call_args_list],
                         ["/tmp/before.png", "/tmp/after.png"])
        client.return_value.close.assert_called_once()

    def test_base_model_queries_upstream_mode_change(self):
        request = {**self.payload, "extra_body": {"base_model": BASE}}
        with (
            patch("gateway.bfs_face_swap.Client") as client,
            patch("gateway.bfs_face_swap.handle_file", side_effect=lambda p: p),
            patch("gateway.bfs_face_swap.predict_logged",
                  return_value=(("/tmp/before.png", "/tmp/after.png"), 10.0)) as predict,
            patch("gateway.bfs_face_swap.result_bytes", return_value=b"image"),
        ):
            client.return_value.predict.return_value = 28.0
            self.assertEqual(run(self.profile, request, None, self.paths), [b"image", b"image"])
        client.return_value.predict.assert_called_once_with(
            base_model=BASE, api_name="/on_mode_change")
        self.assertEqual(predict.call_args.kwargs["kwargs"]["num_inference_steps"], 28.0)

    def test_upstream_gradio_valueerror_is_not_bad_client_parameters(self):
        with patch("gateway.bfs_face_swap.Client") as client:
            client.return_value.predict.side_effect = ValueError("remote gradio validation")
            request = {**self.payload, "extra_body": {"base_model": BASE}}
            with self.assertRaisesRegex(RuntimeError, "mode_change"):
                run(self.profile, request, None, self.paths)
            client.return_value.close.assert_called_once()
        with (
            patch("gateway.bfs_face_swap.Client"),
            patch("gateway.bfs_face_swap.handle_file", side_effect=lambda p: p),
            patch("gateway.bfs_face_swap.predict_logged",
                  side_effect=ValueError("remote gradio failure")),
        ):
            with self.assertRaisesRegex(RuntimeError, "face_swap"):
                run(self.profile, self.payload, None, self.paths)

    def test_api_requires_exactly_two_images(self):
        with tempfile.TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            (root / ".history_bucket_ready").write_text("ok")
            with patch.object(history, "ROOT", root):
                client = TestClient(app)
                for count in (1, 3):
                    response = client.post("/v1/images/edits",
                        data={"model": self.profile.id, "prompt": DEFAULT_PROMPT},
                        files=[("image", (f"image{i}.png", tiny_png(), "image/png"))
                               for i in range(count)])
                    self.assertEqual(response.status_code, 400, response.text)
                    self.assertEqual(response.json()["error"]["code"], "invalid_image_count")
                with patch("gateway.main.invoke", return_value=[tiny_png(), tiny_png()]) as invoker:
                    result = client.post("/v1/images/edits", data={
                        "model": self.profile.id, "prompt": DEFAULT_PROMPT,
                        "response_format": "b64_json", "extra_body": '{"base_model":"Base · Qwen-Image-2.1"}',
                    }, files=[("image", ("face.png", tiny_png(), "image/png")),
                              ("image", ("target.png", tiny_png(), "image/png"))])
                self.assertEqual(result.status_code, 200, result.text)
                self.assertEqual(len(invoker.call_args.kwargs["image_paths"]), 2)
                self.assertEqual(invoker.call_args.args[0].adapter, "bfs_face_swap")
                self.assertEqual(len(result.json()["data"]), 2)
                self.assertEqual([base64.b64decode(item["b64_json"])
                                  for item in result.json()["data"]], [tiny_png(), tiny_png()])
                record = history.list_records()["items"][0]
                self.assertEqual(len(history.get_record(record["id"])["images"]), 2)

    def test_studio_simple_path_and_order(self):
        with patch("gateway.ui.api_generate", return_value=[tiny_png(), tiny_png()]) as api:
            gallery, info = generate_face_swap(self.paths[0], self.paths[1], TURBO, "hf_TEST")
        self.assertEqual(len(gallery), 2)
        self.assertEqual([label for _, label in gallery],
                         ["Before · 原图", "After · 换脸结果"])
        self.assertIn("面部交换完成", info)
        self.assertEqual(api.call_args.kwargs["image_paths"], self.paths)
        self.assertEqual(api.call_args.args[0]["extra_body"], {"base_model": TURBO})
        with self.assertRaises(Exception):
            generate_face_swap(None, self.paths[1], TURBO, None)


if __name__ == "__main__":
    unittest.main()
