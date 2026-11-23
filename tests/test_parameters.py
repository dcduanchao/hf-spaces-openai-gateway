import unittest
from gateway.models import Profile
from gateway.adapter import make_kwargs, parse_size

class ParametersTest(unittest.TestCase):
    def setUp(self):
        self.profile = Profile.model_validate({
            "id": "t", "space_id": "owner/space", "enabled": True, "api_name": "/predict",
            "allowed_sizes": ["1024x1024"],
            "bindings": {"prompt": "text", "width": "w", "height": "h", "seed": "seed"},
            "extras": {"steps": {"target": "num_steps", "kind": "int", "default": 20, "min": 1, "max": 50}},
        })
    def test_mapping(self):
        self.assertEqual(make_kwargs(self.profile, {"prompt": "cat", "size": "1024x1024", "steps": 25}), {"text": "cat", "w": 1024, "h": 1024, "seed": -1, "num_steps": 25})
    def test_reject_unknown(self):
        with self.assertRaises(ValueError):
            make_kwargs(self.profile, {"prompt": "cat", "hack": "anything"})
    def test_reject_size(self):
        with self.assertRaises(ValueError):
            parse_size("4096x4096", self.profile)
    def test_reject_extra_type(self):
        with self.assertRaises(ValueError):
            make_kwargs(self.profile, {"prompt": "cat", "steps": True})

if __name__ == "__main__":
    unittest.main()
