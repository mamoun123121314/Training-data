from __future__ import annotations

import sys
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from test_agent_interactive import (  # noqa: E402
    MOCK_FILES,
    execute_mock_tool,
    parse_tool_call_payload,
    run_sandbox_tests,
)


class TestToolCallJsonRecovery(unittest.TestCase):
    def test_recovers_unescaped_code_quotes_in_edit_file_payload(self) -> None:
        malformed = r'''{"name":"edit_file","arguments":{"path":"src/billing/invoices.py","old_text":"query = f"SELECT invoice_id, tenant_id, total_cents FROM invoices WHERE invoice_id = '{invoice_id}'"\n    cur.execute(query)\n    return cur.fetchone()","new_text":"query = "SELECT invoice_id, tenant_id, total_cents FROM invoices WHERE invoice_id = ? AND tenant_id = ?"\n    cur.execute(query, (invoice_id, caller_tenant_id))\n    return cur.fetchone()"}}'''

        parsed = parse_tool_call_payload(malformed)
        args = parsed["arguments"]
        self.assertEqual(parsed["name"], "edit_file")
        self.assertEqual(args["path"], "src/billing/invoices.py")
        self.assertIn('f"SELECT', args["old_text"])
        self.assertIn("\n    cur.execute(query)", args["old_text"])
        self.assertIn("AND tenant_id = ?", args["new_text"])
        self.assertIn("(invoice_id, caller_tenant_id)", args["new_text"])

        path = "src/billing/invoices.py"
        original = MOCK_FILES[path]
        try:
            edited = execute_mock_tool("edit_file", args)
            self.assertEqual(edited["status"], "edited", edited)
            self.assertEqual(run_sandbox_tests("tests/security/invoices"), {"status": "passed", "checks": "static_ast", "passed": 9, "failed": 0})
        finally:
            MOCK_FILES[path] = original


if __name__ == "__main__":
    unittest.main()
