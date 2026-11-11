import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_qwen.artifacts import verify_model_dir


class ModelArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.temp_dir.name)
        (self.model_dir / ".download_complete").touch()
        (self.model_dir / "config.json").write_text("{}", encoding="utf-8")
        (self.model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        index = {"weight_map": {"layer.weight": "model-00001.safetensors"}}
        (self.model_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
        (self.model_dir / "model-00001.safetensors").write_bytes(b"weights")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_complete_model_directory_passes(self):
        self.assertEqual(verify_model_dir(self.model_dir), [])

    def test_missing_marker_fails(self):
        (self.model_dir / ".download_complete").unlink()
        self.assertIn("缺少 .download_complete 标记", verify_model_dir(self.model_dir))

    def test_missing_indexed_shard_fails(self):
        (self.model_dir / "model-00001.safetensors").unlink()
        errors = verify_model_dir(self.model_dir)
        self.assertTrue(any("权重分片缺失" in item for item in errors))

    def test_missing_tokenizer_config_fails(self):
        (self.model_dir / "tokenizer_config.json").unlink()
        errors = verify_model_dir(self.model_dir)
        self.assertIn("缺少文件：tokenizer_config.json", errors)


if __name__ == "__main__":
    unittest.main()
