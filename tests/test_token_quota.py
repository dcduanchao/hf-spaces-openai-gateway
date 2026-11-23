"""Token quota unit tests; no real HF credentials or network calls."""
import unittest
from unittest.mock import patch, MagicMock
from gateway.token_quota import format_seconds, format_date, parse_quota, token_quota_info


class TokenQuotaTests(unittest.TestCase):
    def test_time_format(self):
        self.assertEqual(format_seconds(3661), "1 小时 1 分 1 秒")

    def test_reset_timezone(self):
        self.assertIn("2026-11-24 09:00:00 JST", format_date("2026-11-24T00:00:00Z"))
        self.assertIn("未提供", format_date(None))

    def test_invalid_quota(self):
        with self.assertRaises(ValueError):
            parse_quota({"current": -5, "base": 30})

    def test_empty_token(self):
        self.assertIn("匿名模式", token_quota_info(""))

    @patch("gateway.token_quota.httpx.get")
    @patch("gateway.token_quota.HfApi")
    def test_success(self, api, get):
        api.return_value.whoami.return_value = {"name": "demo_account"}
        response = MagicMock()
        response.json.return_value = {"current": 1200, "base": 3600, "resetsAt": "2026-11-24T00:00:00Z"}
        get.return_value = response
        value = token_quota_info("hf_TestToken123")
        self.assertIn("20 分 0 秒", value)
        self.assertIn("40 分 0 秒", value)
        self.assertIn("2026-11-24 09:00:00 JST", value)
        self.assertNotIn("hf_TestToken123", value)

    @patch("gateway.token_quota.httpx.get")
    @patch("gateway.token_quota.HfApi")
    def test_missing_reset(self, api, get):
        api.return_value.whoami.return_value = {"name": "person"}
        response = MagicMock()
        response.json.return_value = {"current": 100, "base": 100}
        get.return_value = response
        self.assertIn("未提供", token_quota_info("hf_TestToken123"))


if __name__ == "__main__":
    unittest.main()
