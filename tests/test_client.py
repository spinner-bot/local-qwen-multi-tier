import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_qwen.client import chat_completion, check_health


class _Handler(BaseHTTPRequestHandler):
    payload = None
    authorization = None

    def log_message(self, *_args):
        pass

    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def do_POST(self):
        _Handler.authorization = self.headers.get("Authorization")
        _Handler.payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        body = json.dumps({"choices": [{"message": {"content": "2"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class LocalAPIClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_health_check_targets_server_root(self):
        self.assertTrue(check_health(self.base_url))

    def test_chat_uses_openai_compatible_schema_and_disables_thinking(self):
        answer = chat_completion(
            self.base_url,
            "test-secret",
            "l2",
            [{"role": "user", "content": "1+1?"}],
        )
        self.assertEqual(answer, "2")
        self.assertEqual(_Handler.authorization, "Bearer test-secret")
        self.assertEqual(_Handler.payload["model"], "l2")
        self.assertEqual(_Handler.payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(_Handler.payload["messages"][0]["content"], "1+1?")

    def test_empty_api_key_omits_authorization_header(self):
        chat_completion(
            self.base_url,
            "",
            "l1",
            [{"role": "user", "content": "hello"}],
        )
        self.assertIsNone(_Handler.authorization)


if __name__ == "__main__":
    unittest.main()
