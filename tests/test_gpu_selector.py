import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SELECTOR = REPO / "scripts" / "llm-select-gpus"


class GpuSelectorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.bin_dir = Path(self.temp_dir.name)
        fake = self.bin_dir / "nvidia-smi"
        fake.write_text(
            "#!/usr/bin/env bash\n"
            "if [[ \"$*\" == *'--query-gpu=index'* ]]; then printf '0\\n1\\n2\\n3\\n'; exit 0; fi\n"
            "gpu=\"${*#*--id=}\"; gpu=\"${gpu%% *}\"\n"
            "if [[ \"$*\" == *'--query-gpu=memory.used'* ]]; then\n"
            "  case \"$gpu\" in 0) echo 4000;; 1) echo 1000;; 2) echo 800;; 3) echo 1200;; esac; exit 0\n"
            "fi\n"
            "case \"$gpu\" in 1) echo '123, other.py, 900 MiB';; *) :;; esac\n",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        self.env = {**os.environ, "PATH": f"{self.bin_dir}:{os.environ['PATH']}"}
        self.env.pop("LLM_GPU_POOL", None)

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_selector(self, count, pool=None):
        env = dict(self.env)
        if pool is not None:
            env["LLM_GPU_POOL"] = pool
        return subprocess.run(
            [str(SELECTOR), str(count)], env=env, text=True, capture_output=True
        )

    def test_selects_any_free_cards_and_skips_busy_devices(self):
        result = self.run_selector(2)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "2,3")

    def test_restricts_selection_to_requested_pool(self):
        result = self.run_selector(1, "3")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "3")

    def test_fails_when_pool_has_too_few_free_cards(self):
        result = self.run_selector(1, "1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("仅找到 0 张", result.stderr)


if __name__ == "__main__":
    unittest.main()
