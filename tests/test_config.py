import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_qwen.config import get_tier, load_tiers


class TierConfigTests(unittest.TestCase):
    def test_registry_has_all_tiers_and_expected_ports(self):
        tiers = load_tiers()
        self.assertEqual(set(tiers), {"l1", "l2", "l3", "l3q", "l4", "l5"})
        self.assertEqual(tiers["l1"].port, 8101)
        self.assertEqual(tiers["l3q"].port, 8103)
        self.assertEqual(tiers["l5"].port, 8110)

    def test_lookup_is_case_insensitive(self):
        self.assertEqual(get_tier("L2").model, "l2")

    def test_unknown_tier_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "unknown model tier"):
            get_tier("unknown")


if __name__ == "__main__":
    unittest.main()
