import unittest
from unittest.mock import patch
from gateway import minimax_h3_workflow as h3
from gateway.video_adapter import validate

class H3WorkflowTests(unittest.TestCase):
    def test_args_without_image(self):
        args = validate({"model": h3.MODEL_ID, "prompt": "A forest in rain", "seconds": 5, "seed": 123}, [])
        self.assertEqual(args, ("A forest in rain", None, h3.CANVASES[0], 5.0, 6, 123, False, "off"))

    def test_image_and_canvas(self):
        with patch.object(h3, "handle_file", return_value="file"):
            args = h3.build_args({"prompt": "Waves", "extra_body": {"canvas": h3.CANVASES[5], "steps": 4}}, ["image.png"])
        self.assertEqual(args[1:5], ("file", h3.CANVASES[5], 5.0, 4))

    def test_validation(self):
        for extra in ({"steps": 5}, {"canvas": "bad"}, {"lora": "unknown"}):
            with self.assertRaises(ValueError):
                h3.build_args({"prompt": "Rain", "extra_body": extra}, [])
        with self.assertRaises(ValueError):
            h3.build_args({"prompt": "Rain"}, ["a", "b"])

    def test_seed(self):
        with patch.object(h3.secrets, "randbelow", return_value=77):
            self.assertEqual(h3.resolve_seed({"seed": -1})["seed"], 77)
        self.assertEqual(h3.resolve_seed({"seed": 123})["seed"], 123)
