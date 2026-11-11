"""
Agent Fine-Tuning Generators Part 4 (Categories 51–59):
SWE Multi-File Bugfix, Git Conflict Resolution, TDD Loop, Stacktrace Debugger,
Typecheck/Linter Loop, Model Context Protocol (MCP), Dynamic Replanning,
Episodic Memory & User Preferences, and Sub-Agent Orchestration.
"""

from __future__ import annotations

import random
from typing import Any

from training_data.schema import (
    build_agent_system,
    format_parallel_tool_calls,
    format_tool_call,
    format_tool_response,
    make_record,
)
from training_data.vocab import (
    ENVIRONMENTS,
    SERVICES,
    TENANT_SLUGS,
    USERNAMES,
    gen_short_sha,
)


def _tool_spec(name: str, desc: str, props: dict[str, Any], req: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "description": desc,
        "parameters": {"type": "object", "properties": props, "required": req},
    }


# ---------------------------------------------------------------------------
# 51. swe_multi_file_bugfix_trajectory
# ---------------------------------------------------------------------------

def gen_swe_multi_file_bugfix_trajectory(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    fee_bps = rng.randint(15, 475)
    sample_amt = rng.randint(2000, 90000)
    expected_fee = (sample_amt * fee_bps) // 10_000
    buggy_fee = (sample_amt * fee_bps) // 100
    caller_path = f"src/{domain_ctx}/checkout.py"
    callee_path = f"src/{domain_ctx}/fees.py"
    test_path = f"tests/test_{domain_ctx}_fees.py"

    tools = [
        _tool_spec("grep_symbol", "Search symbol occurrences across repository.", {"symbol": {"type": "string"}}, ["symbol"]),
        _tool_spec("read_file", "Read file text.", {"path": {"type": "string"}}, ["path"]),
        _tool_spec(
            "edit_file",
            "Replace exact substring in file.",
            {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
            ["path", "old_text", "new_text"],
        ),
        _tool_spec("run_pytest", "Execute pytest target.", {"target": {"type": "string"}}, ["target"]),
    ]

    u_msg = (
        f"Ticket `SWE-{idx + 1000}` (`{svc_name}`, `{env}`): `compute_settlement_fee({sample_amt}, {fee_bps})` "
        f"returns `{buggy_fee}` cents instead of `{expected_fee}` cents (`{fee_bps}` basis points). "
        "Trace across files, fix the contract mismatch, and verify tests."
    )
    c1 = format_tool_call(
        "grep_symbol",
        {"symbol": "compute_settlement_fee"},
        thought="<thought>\nLocate both definition and call sites of `compute_settlement_fee` before reading files.\n</thought>",
    )
    r1 = format_tool_response({"matches": [f"{caller_path}:18", f"{callee_path}:6", f"{test_path}:12"]})
    c2 = format_parallel_tool_calls(
        [("read_file", {"path": caller_path}), ("read_file", {"path": callee_path})],
        thought=f"<thought>\nRead `{caller_path}` and `{callee_path}` in parallel to inspect unit assumptions (percent vs basis points).\n</thought>",
    )
    r2a = format_tool_response(
        {"path": caller_path, "content": f"from .{domain_ctx}_fees import compute_settlement_fee\n# passes fee_bps (1 bp = 1/10000)\nfee = compute_settlement_fee(amount_cents, {fee_bps})\n"}
    )
    r2b = format_tool_response(
        {"path": callee_path, "content": "def compute_settlement_fee(amount_cents: int, rate_bps: int) -> int:\n    return (amount_cents * rate_bps) // 100\n"}
    )
    c3 = format_tool_call(
        "edit_file",
        {
            "path": callee_path,
            "old_text": "return (amount_cents * rate_bps) // 100",
            "new_text": "return (amount_cents * rate_bps) // 10_000",
        },
        thought=f"<thought>\n`{caller_path}` passes basis points (`10,000 bps = 100%`), while `{callee_path}` divides by `100` (treating bps as whole percent). Update divisor to `10_000`.\n</thought>",
    )
    r3 = format_tool_response({"status": "ok", "path": callee_path})
    c4 = format_tool_call("run_pytest", {"target": test_path}, thought=f"<thought>\nRun `{test_path}` to confirm `{sample_amt} * {fee_bps} // 10_000 == {expected_fee}` passes.\n</thought>")
    r4 = format_tool_response({"exit_code": 0, "passed": 9, "failed": 0})
    final = (
        f"Fixed `compute_settlement_fee` in `{callee_path}`: changed divisor from `100` to `10_000` to match basis-point callers in `{caller_path}` "
        f"(`{sample_amt} * {fee_bps} // 10_000 == {expected_fee}`). Verified `{test_path}` (`9 passed`)."
    )

    return make_record(
        record_id=f"td-swemf-{idx:07d}",
        domain="agent_use",
        category="swe_multi_file_bugfix_trajectory",
        difficulty="hard",
        style="agent_trajectory",
        tags=["swe-agent", "multi-file", "bugfix", "pytest"],
        quality_check="tool_schema_verified",
        messages=[
            {"role": "system", "content": build_agent_system(rng, tools)},
            {"role": "user", "content": u_msg},
            {"role": "assistant", "content": c1},
            {"role": "tool", "content": r1},
            {"role": "assistant", "content": c2},
            {"role": "tool", "content": r2a},
            {"role": "tool", "content": r2b},
            {"role": "assistant", "content": c3},
            {"role": "tool", "content": r3},
            {"role": "assistant", "content": c4},
            {"role": "tool", "content": r4},
            {"role": "assistant", "content": final},
        ],
    )


# ---------------------------------------------------------------------------
# 52. swe_git_conflict_and_pr_workflow
# ---------------------------------------------------------------------------

def gen_swe_git_conflict_and_pr_workflow(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    timeout_s = rng.randint(5, 45)
    max_retries = rng.randint(2, 6)
    sha_head = gen_short_sha(rng)
    fpath = f"src/{domain_ctx}/client_cfg.py"

    tools = [
        _tool_spec("git_status", "Show conflicted and modified files.", {}, []),
        _tool_spec("read_file", "Read file with conflict markers.", {"path": {"type": "string"}}, ["path"]),
        _tool_spec("write_file", "Write resolved file content.", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
        _tool_spec("run_ci_suite", "Run unit tests and linter.", {"scope": {"type": "string"}}, ["scope"]),
    ]

    u_msg = (
        f"Rebase `feat/{domain_ctx}-retries` (`{sha_head}`, task `GIT-{idx + 500}`) in `{svc_name}` hit a merge conflict: "
        f"keep BOTH `timeout_sec={timeout_s}` from `HEAD` and `max_retries={max_retries}` from the feature branch, then run CI."
    )
    c1 = format_tool_call("git_status", {}, thought="<thought>\nInspect `git_status` to identify exact conflicted file paths.\n</thought>")
    r1 = format_tool_response({"conflicted": [fpath], "branch": f"feat/{domain_ctx}-retries"})
    c2 = format_tool_call("read_file", {"path": fpath}, thought=f"<thought>\nRead `{fpath}` to inspect both sides of the conflict markers.\n</thought>")
    conflict_txt = (
        "DEFAULT_CLIENT_CFG = {\n"
        "<<<<<<< HEAD\n"
        f"    'timeout_sec': {timeout_s},\n"
        "=======\n"
        f"    'max_retries': {max_retries},\n"
        f">>>>>>> feat/{domain_ctx}-retries\n"
        "}\n"
    )
    resolved_txt = (
        "DEFAULT_CLIENT_CFG = {\n"
        f"    'timeout_sec': {timeout_s},\n"
        f"    'max_retries': {max_retries},\n"
        "}\n"
    )
    r2 = format_tool_response({"path": fpath, "content": conflict_txt})
    c3 = format_tool_call(
        "write_file",
        {"path": fpath, "content": resolved_txt},
        thought=f"<thought>\nCombine `timeout_sec: {timeout_s}` and `max_retries: {max_retries}` cleanly with zero leftover conflict markers.\n</thought>",
    )
    r3 = format_tool_response({"status": "written", "path": fpath})
    c4 = format_tool_call("run_ci_suite", {"scope": domain_ctx}, thought="<thought>\nVerify syntax and unit tests pass after conflict resolution.\n</thought>")
    r4 = format_tool_response({"exit_code": 0, "lint": "clean", "tests_passed": 14})
    final = (
        f"Resolved merge conflict in `{fpath}` preserving both `'timeout_sec': {timeout_s}` and `'max_retries': {max_retries}`, "
        f"and verified `run_ci_suite` (`14` tests passed, lint clean)."
    )

    return make_record(
        record_id=f"td-gitcf-{idx:07d}",
        domain="agent_use",
        category="swe_git_conflict_and_pr_workflow",
        difficulty="medium",
        style="agent_trajectory",
        tags=["git", "merge-conflict", "swe-agent"],
        quality_check="tool_schema_verified",
        messages=[
            {"role": "system", "content": build_agent_system(rng, tools)},
            {"role": "user", "content": u_msg},
            {"role": "assistant", "content": c1},
            {"role": "tool", "content": r1},
            {"role": "assistant", "content": c2},
            {"role": "tool", "content": r2},
            {"role": "assistant", "content": c3},
            {"role": "tool", "content": r3},
            {"role": "assistant", "content": c4},
            {"role": "tool", "content": r4},
            {"role": "assistant", "content": final},
        ],
    )


# ---------------------------------------------------------------------------
# 53. swe_test_driven_feature_agent
# ---------------------------------------------------------------------------

def gen_swe_test_driven_feature_agent(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    cap_n = rng.randint(50, 9500)
    impl_path = f"src/{domain_ctx}/quota.py"
    test_path = f"tests/test_{domain_ctx}_quota.py"

    tools = [
        _tool_spec("write_file", "Write file content.", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
        _tool_spec("run_pytest", "Run pytest on path.", {"target": {"type": "string"}}, ["target"]),
    ]

    u_msg = (
        f"TDD task `TDD-{idx + 100}` in `{svc_name}`: write failing unit tests first in `{test_path}` for "
        f"`clamp_batch_size(requested: int, cap: int = {cap_n}) -> int` (must raise `ValueError` if `requested <= 0` and clamp to `cap`), "
        f"confirm failure, then implement `{impl_path}` and pass tests."
    )
    test_code = (
        "import pytest\n"
        f"from src.{domain_ctx}.quota import clamp_batch_size\n\n"
        "def test_clamp_bounds() -> None:\n"
        f"    assert clamp_batch_size(10) == 10\n"
        f"    assert clamp_batch_size({cap_n + 50}) == {cap_n}\n"
        "    with pytest.raises(ValueError):\n"
        "        clamp_batch_size(0)\n"
    )
    impl_code = (
        f"def clamp_batch_size(requested: int, cap: int = {cap_n}) -> int:\n"
        "    if requested <= 0:\n"
        "        raise ValueError('requested must be positive')\n"
        "    return min(requested, cap)\n"
    )
    c1 = format_tool_call("write_file", {"path": test_path, "content": test_code}, thought=f"<thought>\nRed phase: author boundary and error-case tests in `{test_path}` first.\n</thought>")
    r1 = format_tool_response({"status": "written", "path": test_path})
    c2 = format_tool_call("run_pytest", {"target": test_path}, thought="<thought>\nExecute tests to confirm expected Red-phase failure before writing implementation.\n</thought>")
    r2 = format_tool_response({"exit_code": 2, "error": f"ModuleNotFoundError: No module named 'src.{domain_ctx}.quota'"})
    c3 = format_tool_call("write_file", {"path": impl_path, "content": impl_code}, thought=f"<thought>\nGreen phase: implement `clamp_batch_size` in `{impl_path}` handling `requested <= 0` and `min(requested, cap)`.\n</thought>")
    r3 = format_tool_response({"status": "written", "path": impl_path})
    c4 = format_tool_call("run_pytest", {"target": test_path}, thought="<thought>\nRe-run pytest to verify Green phase.\n</thought>")
    r4 = format_tool_response({"exit_code": 0, "passed": 1, "failed": 0})
    final = f"Completed Red-Green TDD cycle for `clamp_batch_size(cap={cap_n})`: created `{test_path}`, verified initial failure, implemented `{impl_path}`, and confirmed `pytest` passes."

    return make_record(
        record_id=f"td-tdd-{idx:07d}",
        domain="agent_use",
        category="swe_test_driven_feature_agent",
        difficulty="hard",
        style="agent_trajectory",
        tags=["tdd", "swe-agent", "pytest"],
        quality_check="tool_schema_verified",
        messages=[
            {"role": "system", "content": build_agent_system(rng, tools)},
            {"role": "user", "content": u_msg},
            {"role": "assistant", "content": c1},
            {"role": "tool", "content": r1},
            {"role": "assistant", "content": c2},
            {"role": "tool", "content": r2},
            {"role": "assistant", "content": c3},
            {"role": "tool", "content": r3},
            {"role": "assistant", "content": c4},
            {"role": "tool", "content": r4},
            {"role": "assistant", "content": final},
        ],
    )


# ---------------------------------------------------------------------------
# 54. swe_stacktrace_debugger_agent
# ---------------------------------------------------------------------------

def gen_swe_stacktrace_debugger_agent(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    line_no = rng.randint(24, 180)
    fpath = f"src/{domain_ctx}/normalizer.py"

    tools = [
        _tool_spec("read_file_lines", "Read line range from file.", {"path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}}, ["path", "start", "end"]),
        _tool_spec("edit_file", "Edit exact text in file.", {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, ["path", "old_text", "new_text"]),
        _tool_spec("run_repro", "Run reproduction check.", {"case_id": {"type": "string"}}, ["case_id"]),
    ]

    case_id = f"CRASH-{idx + 200}"
    u_msg = (
        f"Investigate `{case_id}` in `{svc_name}`:\n"
        f"`File \"{fpath}\", line {line_no}, in normalize_tag -> AttributeError: 'NoneType' object has no attribute 'strip'`\n"
        "Inspect the crashing frame, patch safely, and run reproduction."
    )
    c1 = format_tool_call(
        "read_file_lines",
        {"path": fpath, "start": line_no - 4, "end": line_no + 4},
        thought=f"<thought>\nRead `{fpath}` around line `{line_no}` to see how `None` reaches `.strip()`.\n</thought>",
    )
    r1 = format_tool_response(
        {"path": fpath, "lines": f"{line_no - 1}: def normalize_tag(raw: str | None) -> str:\n{line_no}:     return raw.strip().lower()\n"}
    )
    c2 = format_tool_call(
        "edit_file",
        {
            "path": fpath,
            "old_text": "return raw.strip().lower()",
            "new_text": "return (raw or '').strip().lower()",
        },
        thought="<thought>\n`raw` is typed `str | None`, so `None` triggers `AttributeError`. Coalesce `(raw or '').strip().lower()`.\n</thought>",
    )
    r2 = format_tool_response({"status": "updated", "path": fpath})
    c3 = format_tool_call("run_repro", {"case_id": case_id}, thought=f"<thought>\nRe-run `{case_id}` reproduction payload (`raw=None`) to verify clean return.\n</thought>")
    r3 = format_tool_response({"case_id": case_id, "exit_code": 0, "output": "''"})
    final = f"Patched `{fpath}:{line_no}` (`return (raw or '').strip().lower()`) so `None` inputs return `''` cleanly; verified `{case_id}` exits `0`."

    return make_record(
        record_id=f"td-trace-{idx:07d}",
        domain="agent_use",
        category="swe_stacktrace_debugger_agent",
        difficulty="medium",
        style="agent_trajectory",
        tags=["debugging", "stacktrace", "swe-agent"],
        quality_check="tool_schema_verified",
        messages=[
            {"role": "system", "content": build_agent_system(rng, tools)},
            {"role": "user", "content": u_msg},
            {"role": "assistant", "content": c1},
            {"role": "tool", "content": r1},
            {"role": "assistant", "content": c2},
            {"role": "tool", "content": r2},
            {"role": "assistant", "content": c3},
            {"role": "tool", "content": r3},
            {"role": "assistant", "content": final},
        ],
    )


# ---------------------------------------------------------------------------
# 55. swe_shell_build_and_linter_loop
# ---------------------------------------------------------------------------

def gen_swe_shell_build_and_linter_loop(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    fpath = f"src/{domain_ctx}/router.py"
    line_n = rng.randint(15, 140)

    tools = [
        _tool_spec("run_typecheck", "Run mypy --strict on target module.", {"path": {"type": "string"}}, ["path"]),
        _tool_spec("edit_file", "Replace exact snippet in file.", {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, ["path", "old_text", "new_text"]),
    ]

    u_msg = f"CI gate `LINT-{idx + 300}` is failing `mypy --strict` on `{fpath}` in `{svc_name}`. Diagnose and fix until strict typechecking passes."
    c1 = format_tool_call("run_typecheck", {"path": fpath}, thought=f"<thought>\nRun `run_typecheck` on `{fpath}` to inspect exact `mypy --strict` diagnostics.\n</thought>")
    r1 = format_tool_response(
        {
            "exit_code": 1,
            "diagnostics": [
                f"{fpath}:{line_n}: error: Incompatible return value type (got 'str | None', expected 'str') [return-value]",
                f"snippet: return headers.get('X-Request-ID')",
            ],
        }
    )
    c2 = format_tool_call(
        "edit_file",
        {
            "path": fpath,
            "old_text": "return headers.get('X-Request-ID')",
            "new_text": "return headers.get('X-Request-ID') or 'req-missing'",
        },
        thought="<thought>\n`dict.get()` returns `str | None` whereas return annotation requires `str`. Provide explicit fallback `'req-missing'`.\n</thought>",
    )
    r2 = format_tool_response({"status": "updated", "path": fpath})
    c3 = format_tool_call("run_typecheck", {"path": fpath}, thought="<thought>\nRe-run `run_typecheck` to confirm zero remaining diagnostics.\n</thought>")
    r3 = format_tool_response({"exit_code": 0, "diagnostics": []})
    final = f"Resolved `[return-value]` error at `{fpath}:{line_n}` by coalescing `headers.get('X-Request-ID') or 'req-missing'`. `mypy --strict` now exits `0`."

    return make_record(
        record_id=f"td-lint-{idx:07d}",
        domain="agent_use",
        category="swe_shell_build_and_linter_loop",
        difficulty="easy",
        style="agent_trajectory",
        tags=["mypy", "typecheck", "swe-agent"],
        quality_check="tool_schema_verified",
        messages=[
            {"role": "system", "content": build_agent_system(rng, tools)},
            {"role": "user", "content": u_msg},
            {"role": "assistant", "content": c1},
            {"role": "tool", "content": r1},
            {"role": "assistant", "content": c2},
            {"role": "tool", "content": r2},
            {"role": "assistant", "content": c3},
            {"role": "tool", "content": r3},
            {"role": "assistant", "content": final},
        ],
    )


# ---------------------------------------------------------------------------
# 56. mcp_server_discovery_and_invocation (Model Context Protocol!)
# ---------------------------------------------------------------------------

def gen_mcp_server_discovery_and_invocation(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    slo_pct = round(rng.uniform(99.50, 99.99), 2)
    burn_rate = round(rng.uniform(0.3, 4.8), 2)
    uri = f"mcp://observability/{svc_name}/{env}/slo-spec"

    tools = [
        _tool_spec("mcp_list_resources", "List resources exposed by connected MCP server.", {"server": {"type": "string"}}, ["server"]),
        _tool_spec("mcp_read_resource", "Read an MCP resource by URI.", {"uri": {"type": "string"}}, ["uri"]),
        _tool_spec("mcp_call_tool", "Invoke a tool on an MCP server.", {"server": {"type": "string"}, "tool": {"type": "string"}, "args": {"type": "object"}}, ["server", "tool", "args"]),
    ]

    u_msg = f"[MCP-{idx + 100}] Using the `observability` MCP server, read the SLO target for `{svc_name}` in `{env}` and check its current 1h burn rate."
    c1 = format_tool_call(
        "mcp_list_resources",
        {"server": "observability"},
        thought="<thought>\nList resources on the `observability` MCP server first rather than guessing resource URIs.\n</thought>",
    )
    r1 = format_tool_response({"resources": [{"uri": uri, "mimeType": "application/json"}], "tools": ["get_burn_rate"]})
    c2 = format_parallel_tool_calls(
        [
            ("mcp_read_resource", {"uri": uri}),
            ("mcp_call_tool", {"server": "observability", "tool": "get_burn_rate", "args": {"service": svc_name, "env": env, "window": "1h"}}),
        ],
        thought=f"<thought>\nRead `{uri}` and invoke `get_burn_rate` in parallel since both identifiers are now verified.\n</thought>",
    )
    r2a = format_tool_response({"service": svc_name, "env": env, "slo_target_pct": slo_pct, "page_threshold_burn": 2.0})
    r2b = format_tool_response({"burn_rate_1h": burn_rate})
    status = "BREACHING page threshold (`> 2.0x`)" if burn_rate > 2.0 else "HEALTHY (`<= 2.0x` threshold)"
    final = f"`{svc_name}` (`{env}`) SLO target is **`{slo_pct}%`**; current 1h error-budget burn rate is **`{burn_rate}x`** — {status}."

    return make_record(
        record_id=f"td-mcp-{idx:07d}",
        domain="agent_use",
        category="mcp_server_discovery_and_invocation",
        difficulty="medium",
        style="agent_trajectory",
        tags=["mcp", "model-context-protocol", "tool-calling"],
        quality_check="tool_schema_verified",
        messages=[
            {"role": "system", "content": build_agent_system(rng, tools)},
            {"role": "user", "content": u_msg},
            {"role": "assistant", "content": c1},
            {"role": "tool", "content": r1},
            {"role": "assistant", "content": c2},
            {"role": "tool", "content": r2a},
            {"role": "tool", "content": r2b},
            {"role": "assistant", "content": final},
        ],
    )


# ---------------------------------------------------------------------------
# 57. agent_long_horizon_plan_and_replan
# ---------------------------------------------------------------------------

def gen_agent_long_horizon_plan_and_replan(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    lag_sec = rng.randint(45, 420)
    rows_migrated = rng.randint(1200, 98000)

    tools = [
        _tool_spec("check_replica_lag", "Check read-replica replication lag in seconds.", {"service": {"type": "string"}, "env": {"type": "string"}}, ["service", "env"]),
        _tool_spec("run_backfill_batch", "Run bounded backfill against specified target (`replica` or `throttled_primary`).", {"table": {"type": "string"}, "mode": {"type": "string"}, "batch_size": {"type": "integer"}}, ["table", "mode", "batch_size"]),
    ]

    u_msg = (
        f"[PLAN-{idx + 100}] Backfill `{table_ctx}` for `{svc_name}` in `{env}`: use `mode='replica'` (`batch_size=5000`) ONLY if replica lag `< 10s`; "
        "otherwise dynamically replan to `mode='throttled_primary'` with `batch_size=500`."
    )
    c1 = format_tool_call(
        "check_replica_lag",
        {"service": svc_name, "env": env},
        thought=f"<thought>\nPlan: (1) Check replica lag for `{svc_name}` in `{env}`; (2) Choose `replica` (`5000`) vs `throttled_primary` (`500`) based on `< 10s` threshold.\n</thought>",
    )
    r1 = format_tool_response({"service": svc_name, "env": env, "lag_seconds": lag_sec})
    c2 = format_tool_call(
        "run_backfill_batch",
        {"table": table_ctx, "mode": "throttled_primary", "batch_size": 500},
        thought=f"<thought>\nReplan: Replica lag is `{lag_sec}s` (`>= 10s`), so `mode='replica'` would read stale state and worsen lag. Switch to `mode='throttled_primary'` with `batch_size=500`.\n</thought>",
    )
    r2 = format_tool_response({"table": table_ctx, "mode": "throttled_primary", "rows_updated": rows_migrated, "status": "completed"})
    final = (
        f"Observed replica lag of **`{lag_sec}s`** (`>= 10s` limit), replanned from `replica` to **`throttled_primary` (`batch_size=500`)**, "
        f"and completed backfilling **`{rows_migrated}`** rows in `{table_ctx}`."
    )

    return make_record(
        record_id=f"td-replan-{idx:07d}",
        domain="agent_use",
        category="agent_long_horizon_plan_and_replan",
        difficulty="hard",
        style="agent_trajectory",
        tags=["planning", "replanning", "state-machine"],
        quality_check="tool_schema_verified",
        messages=[
            {"role": "system", "content": build_agent_system(rng, tools)},
            {"role": "user", "content": u_msg},
            {"role": "assistant", "content": c1},
            {"role": "tool", "content": r1},
            {"role": "assistant", "content": c2},
            {"role": "tool", "content": r2},
            {"role": "assistant", "content": final},
        ],
    )


# ---------------------------------------------------------------------------
# 58. agent_memory_and_preferences_retrieval
# ---------------------------------------------------------------------------

def gen_agent_memory_and_preferences_retrieval(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    user_handle = rng.choice(USERNAMES)
    pkg_mgr = rng.choice(["uv run pytest", "poetry run pytest", " hatch run test"]) .strip()

    tools = [
        _tool_spec("search_workspace_memory", "Retrieve stored user/repo preferences.", {"user": {"type": "string"}, "repo": {"type": "string"}}, ["user", "repo"]),
        _tool_spec("run_command", "Execute workspace shell command.", {"cmd": {"type": "string"}}, ["cmd"]),
    ]

    u_msg = f"[MEM-{idx + 100}] User `{user_handle}` says: 'Run the `{domain_ctx}` unit tests in `{svc_name}` using our repo's preferred runner.'"
    c1 = format_tool_call(
        "search_workspace_memory",
        {"user": user_handle, "repo": svc_name},
        thought=f"<thought>\nLook up stored test-runner conventions for `{svc_name}` / `{user_handle}` before invoking `run_command`.\n</thought>",
    )
    r1 = format_tool_response({"repo": svc_name, "preferred_test_runner": pkg_mgr, "rule": "Never invoke bare pytest directly"})
    cmd_str = f"{pkg_mgr} tests/{domain_ctx}"
    c2 = format_tool_call(
        "run_command",
        {"cmd": cmd_str},
        thought=f"<thought>\nMemory specifies `{pkg_mgr}` (avoiding bare `pytest`). Run `{cmd_str}`.\n</thought>",
    )
    passed_n = rng.randint(8, 64)
    r2 = format_tool_response({"cmd": cmd_str, "exit_code": 0, "summary": f"{passed_n} passed"})
    final = f"Executed `{cmd_str}` per `{svc_name}` workspace preference: **`{passed_n}` passed** (`exit_code=0`)."

    return make_record(
        record_id=f"td-mem-{idx:07d}",
        domain="agent_use",
        category="agent_memory_and_preferences_retrieval",
        difficulty="easy",
        style="agent_trajectory",
        tags=["agent-memory", "personalization", "tool-calling"],
        quality_check="tool_schema_verified",
        messages=[
            {"role": "system", "content": build_agent_system(rng, tools)},
            {"role": "user", "content": u_msg},
            {"role": "assistant", "content": c1},
            {"role": "tool", "content": r1},
            {"role": "assistant", "content": c2},
            {"role": "tool", "content": r2},
            {"role": "assistant", "content": final},
        ],
    )


# ---------------------------------------------------------------------------
# 59. agent_subagent_orchestration_and_merge
# ---------------------------------------------------------------------------

def gen_agent_subagent_orchestration_and_merge(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    qps_cap = rng.randint(500, 12000)
    conn_pool = rng.randint(20, 120)

    tools = [
        _tool_spec(
            "spawn_subagent",
            "Run isolated specialist sub-agent and return structured findings.",
            {"role": {"type": "string"}, "task": {"type": "string"}},
            ["role", "task"],
        )
    ]

    u_msg = (
        f"[ORCH-{idx + 100}] Orchestrate capacity sign-off for `{svc_name}` (`{env}`) targeting `{qps_cap} req/s`: "
        "dispatch `load_estimator` and `db_capacity_auditor` in parallel and synthesize whether DB pool scaling is needed."
    )
    c1 = format_parallel_tool_calls(
        [
            ("spawn_subagent", {"role": "load_estimator", "task": f"Estimate peak DB connections for {svc_name} at {qps_cap} req/s"}),
            ("spawn_subagent", {"role": "db_capacity_auditor", "task": f"Check configured max DB pool size for {svc_name} in {env}"}),
        ],
        thought="<thought>\nFan out both read-only specialist sub-agents concurrently to compute required vs configured DB connections.\n</thought>",
    )
    req_conns = max(25, qps_cap // 80)
    r1a = format_tool_response({"role": "load_estimator", "required_db_connections": req_conns})
    r1b = format_tool_response({"role": "db_capacity_auditor", "configured_max_connections": conn_pool})
    deficit = req_conns - conn_pool
    verdict = (
        f"**Action Required**: scale `{svc_name}` DB pool (or add PgBouncer multiplexing) by at least **`+{deficit}`** connections (`{conn_pool}` -> `>={req_conns}`)."
        if deficit > 0
        else f"**Capacity Sufficient**: configured pool (`{conn_pool}`) covers required peak (`{req_conns}` connections, headroom `{ -deficit }`)."
    )
    final = f"Sub-agent synthesis for `{svc_name}` (`{env}`, `{qps_cap} req/s`):\n- Required connections: **`{req_conns}`** | Configured max: **`{conn_pool}`**\n- {verdict}"

    return make_record(
        record_id=f"td-subag-{idx:07d}",
        domain="agent_use",
        category="agent_subagent_orchestration_and_merge",
        difficulty="medium",
        style="agent_trajectory",
        tags=["multi-agent", "subagents", "orchestration"],
        quality_check="tool_schema_verified",
        messages=[
            {"role": "system", "content": build_agent_system(rng, tools)},
            {"role": "user", "content": u_msg},
            {"role": "assistant", "content": c1},
            {"role": "tool", "content": r1a},
            {"role": "tool", "content": r1b},
            {"role": "assistant", "content": final},
        ],
    )
