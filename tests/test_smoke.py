import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_qwen import smoke


class LiveSmokeSuiteTests(unittest.TestCase):
    def test_three_model_checks_must_pass(self):
        stdout = io.StringIO()
        with patch.object(smoke, "check_health", return_value=True), \
             patch.object(smoke, "api_key", return_value="key"), \
             patch.object(smoke, "chat_completion", side_effect=["2", "READY", "琥珀"] ) as completion, \
             contextlib.redirect_stdout(stdout):
            result = smoke.main(["l2"])
        self.assertEqual(result, 0)
        self.assertEqual(completion.call_count, 3)
        self.assertIn("3 组推理检查通过", stdout.getvalue())

    def test_unexpected_answer_fails_suite(self):
        stderr = io.StringIO()
        with patch.object(smoke, "check_health", return_value=True), \
             patch.object(smoke, "api_key", return_value="key"), \
             patch.object(smoke, "chat_completion", return_value="wrong"), \
             contextlib.redirect_stderr(stderr):
            result = smoke.main(["l1"])
        self.assertEqual(result, 1)
        self.assertIn("未通过", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
