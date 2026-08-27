import ast
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "audit_judge_review_blindness.py"
SPEC = importlib.util.spec_from_file_location("audit_judge_review_blindness", MODULE_PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class ReviewBlindnessAuditTest(unittest.TestCase):
    def test_review_task_accepts_judge_json_subscript(self):
        tree = ast.parse(
            """
def review_task():
    review_input = {
        "task_spec": {},
        "observation": {},
        "first_pass_judgment": first_result["judge_json"],
        "first_pass_provenance": {},
    }
"""
        )
        findings = []

        AUDIT.audit_review_task_shape(tree, findings)

        self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
