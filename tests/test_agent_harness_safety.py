from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import test_agent_interactive as harness  # noqa: E402


class TestAgentHarnessSafety(unittest.TestCase):
    def test_invoice_sandbox_requires_safe_sql_and_correct_parameter_order(self) -> None:
        path = "src/billing/invoices.py"
        original = harness.MOCK_FILES[path]
        try:
            self.assertEqual(harness.run_sandbox_tests("tests/security/invoices")["status"], "failed")
            patched = original.replace("'{invoice_id}'", "? AND tenant_id = ?")
            patched = patched.replace(
                "cur.execute(query)",
                "cur.execute(query, (invoice_id, caller_tenant_id))",
            )
            harness.MOCK_FILES[path] = patched
            self.assertEqual(
                harness.run_sandbox_tests("tests/security/invoices"),
                {"status": "passed", "checks": "static_ast", "passed": 9, "failed": 0},
            )
            harness.MOCK_FILES[path] = patched.replace(
                "(invoice_id, caller_tenant_id)", "(caller_tenant_id, invoice_id)"
            )
            failed = harness.run_sandbox_tests("tests/security/invoices")
            self.assertEqual(failed["status"], "failed")
            self.assertIn("placeholder order", failed["failure"])
        finally:
            harness.MOCK_FILES[path] = original

    def test_unknown_mock_test_target_is_not_reported_as_passed(self) -> None:
        result = harness.run_sandbox_tests("tests/does-not-exist")
        self.assertEqual(result["status"], "error")

    def test_mock_tests_do_not_execute_model_written_python(self) -> None:
        path = "src/billing/invoices.py"
        original = harness.MOCK_FILES[path]
        with tempfile.TemporaryDirectory() as temp_dir:
            marker = Path(temp_dir) / "must-not-exist"
            try:
                harness.MOCK_FILES[path] = (
                    "def get_invoice_record(cur, caller_tenant_id, invoice_id):\n"
                    f"    open({str(marker)!r}, 'w').write('executed')\n"
                )
                self.assertEqual(harness.run_sandbox_tests("tests/security/invoices")["status"], "failed")
                self.assertFalse(marker.exists())
            finally:
                harness.MOCK_FILES[path] = original

    def test_real_workspace_symlink_search_cannot_read_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            outside = Path(temp_dir) / "outside.txt"
            root.mkdir()
            outside.write_text("unique_secret_marker", encoding="utf-8")
            (root / "linked.txt").symlink_to(outside)
            result = harness.execute_real_workspace_tool(
                root.resolve(), "search_code", {"query": "unique_secret_marker"}
            )
            self.assertEqual(result["matches"], [])

    def test_mock_edit_returns_hash_receipt_for_verified_write(self) -> None:
        path = "src/billing/invoices.py"
        original = harness.MOCK_FILES[path]
        try:
            result = harness.execute_mock_tool(
                "edit_file",
                {"path": path, "old_text": "cur.execute(query)", "new_text": "cur.execute(query, ())"},
            )
            self.assertEqual(result["status"], "edited")
            self.assertTrue(result["verified"])
            self.assertEqual(len(result["before_sha256"]), 64)
            self.assertEqual(len(result["after_sha256"]), 64)
            self.assertNotEqual(result["before_sha256"], result["after_sha256"])
            self.assertIn("cur.execute(query, ())", harness.MOCK_FILES[path])
        finally:
            harness.MOCK_FILES[path] = original

    def test_real_workspace_edit_is_read_back_and_hash_receipted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            target = root / "module.py"
            target.write_text("value = 1\\n", encoding="utf-8")
            result = harness.execute_real_workspace_tool(
                root, "edit_file", {"path": "module.py", "old_text": "value = 1", "new_text": "value = 2"}
            )
            self.assertEqual(result["status"], "edited")
            self.assertTrue(result["verified"])
            self.assertEqual(target.read_text(encoding="utf-8"), "value = 2\\n")
            self.assertEqual(len(result["before_sha256"]), 64)
            self.assertEqual(len(result["after_sha256"]), 64)

    def test_real_workspace_test_execution_requires_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = harness.execute_real_workspace_tool(
                Path(temp_dir).resolve(), "run_tests", {"target": "."}
            )
            self.assertEqual(result["status"], "error")
            self.assertIn("--allow-test-execution", result["error"])


if __name__ == "__main__":
    unittest.main()
