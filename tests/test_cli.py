import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_qwen import cli
from local_qwen.store import ConversationStore


class ChatCLITests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "chat.sqlite3"
        self.env = patch.dict(os.environ, {"LLM_MEMORY_DB": str(self.db_path)})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp_dir.cleanup()

    def test_one_shot_sends_explicit_memories_and_saves_turn(self):
        store = ConversationStore(self.db_path)
        store.remember("style", "简洁")
        captured = {}

        def fake_completion(_url, _key, _model, messages, **_kwargs):
            captured["messages"] = messages
            return "你好。"

        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(cli, "check_health", return_value=True), \
             patch.object(cli, "api_key", return_value="test-key"), \
             patch.object(cli, "chat_completion", side_effect=fake_completion), \
             contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = cli.main(["l2", "--prompt", "你好"])
        self.assertEqual(result, 0)
        self.assertEqual(stdout.getvalue().strip(), "你好。")
        self.assertIn("style: 简洁", captured["messages"][0]["content"])
        session = store.list_sessions()[0]
        self.assertEqual(session["tier"], "l2")
        self.assertEqual(session["message_count"], 2)

    def test_session_cannot_be_reused_with_another_tier(self):
        store = ConversationStore(self.db_path)
        session_id = store.create_session("l1")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = cli.main(["l2", "--session", session_id, "--prompt", "x"])
        self.assertEqual(result, 2)
        self.assertIn("不能用 l2 继续", stderr.getvalue())

    def test_interactive_memory_commands_are_manual_and_persistent(self):
        stdout = io.StringIO()
        with patch.object(cli, "check_health", return_value=True), \
             patch.object(cli, "api_key", return_value="test-key"), \
             patch("builtins.input", side_effect=["/remember style: concise", "/exit"]), \
             contextlib.redirect_stdout(stdout):
            result = cli.main(["l1"])
        self.assertEqual(result, 0)
        self.assertEqual(ConversationStore(self.db_path).memories(), [{"key": "style", "value": "concise"}])


if __name__ == "__main__":
    unittest.main()
