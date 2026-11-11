import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_qwen.store import ConversationStore


class ConversationStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "chat.sqlite3"
        self.store = ConversationStore(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sessions_keep_tier_and_messages_isolated(self):
        first = self.store.create_session("l1")
        second = self.store.create_session("l2")
        self.store.append_turn(first, "你好", "你好！")
        self.assertEqual(self.store.session(first)["tier"], "l1")
        self.assertEqual(
            self.store.history(first),
            [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}],
        )
        self.assertEqual(self.store.history(second), [])
        counts = {item["id"]: item["message_count"] for item in self.store.list_sessions()}
        self.assertEqual(counts[first], 2)
        self.assertEqual(counts[second], 0)

    def test_history_limit_returns_most_recent_messages_in_order(self):
        session = self.store.create_session("l2")
        self.store.append_turn(session, "问题一", "回答一")
        self.store.append_turn(session, "问题二", "回答二")
        self.assertEqual(
            self.store.history(session, limit=2),
            [{"role": "user", "content": "问题二"}, {"role": "assistant", "content": "回答二"}],
        )

    def test_clear_removes_history_but_keeps_session(self):
        session = self.store.create_session("l1")
        self.store.append_turn(session, "记住我", "好的")
        self.store.clear_history(session)
        self.assertEqual(self.store.history(session), [])
        self.assertEqual(self.store.session(session)["tier"], "l1")
        self.assertEqual(self.store.session(session)["title"], "")

    def test_memories_are_explicit_upserts_and_can_be_deleted(self):
        self.store.remember("language", "中文")
        self.store.remember("language", "简体中文")
        self.assertEqual(self.store.memories(), [{"key": "language", "value": "简体中文"}])
        self.assertTrue(self.store.forget("language"))
        self.assertFalse(self.store.forget("language"))
        with self.assertRaises(ValueError):
            self.store.remember("", "value")

    def test_database_file_is_private(self):
        self.assertEqual(self.db_path.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
