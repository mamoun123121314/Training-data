"""Agent Use Generators Part 1 (Categories 24–29): Single Tool, Chained Tools, Error Recovery, SWE Repo Agent, AppSec Agent, Clarification."""

from __future__ import annotations

import random
from typing import Any

from training_data.schema import (
    build_agent_system,
    format_tool_call,
    format_tool_response,
    make_record,
)
from training_data.vocab import (
    DOCUMENTATION_TEST_IPS,
    ENVIRONMENTS,
    FILE_PATHS,
    SAFE_INTERNAL_IPS,
    SERVICES,
    TENANT_SLUGS,
    gen_short_sha,
    gen_uuid_short,
)


def _tool_spec(name: str, desc: str, props: dict[str, Any], req: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "description": desc,
        "parameters": {"type": "object", "properties": props, "required": req},
    }


# ---------------------------------------------------------------------------
# 24. single_tool_precision_calling
# ---------------------------------------------------------------------------

def gen_single_tool_precision_calling(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    tenant = rng.choice(TENANT_SLUGS)
    variant = idx % 3

    if variant == 0:
        difficulty = "easy"
        win_m = rng.randint(5, 180)
        p99 = rng.randint(45, 920)
        err_rate = round(rng.uniform(0.01, 2.45), 2)
        tools = [
            _tool_spec(
                "get_service_slo",
                "Query p99 latency and error rate for a service.",
                {"service": {"type": "string"}, "env": {"type": "string"}, "window_minutes": {"type": "integer"}},
                ["service", "env", "window_minutes"],
            ),
            _tool_spec(
                "restart_worker_pool",
                "Restart worker instances.",
                {"service": {"type": "string"}, "env": {"type": "string"}},
                ["service", "env"],
            ),
        ]
        u_msg = f"Check p99 latency and error rate for `{svc_name}` in `{env}` over the past `{win_m}` minutes."
        a_call = format_tool_call("get_service_slo", {"service": svc_name, "env": env, "window_minutes": win_m})
        t_resp = format_tool_response({"service": svc_name, "env": env, "p99_ms": p99, "error_pct": err_rate})
        a_final = f"`{svc_name}` in `{env}` (last `{win_m}m`): **p99 latency `{p99} ms`**, **error rate `{err_rate}%`**."
    elif variant == 1:
        difficulty = "medium"
        cve_num = f"CVE-2026-{rng.randint(10000, 49999)}"
        pkg = f"lib-{domain_ctx}"
        fixed_v = f"{rng.randint(2, 6)}.{rng.randint(1, 9)}.{rng.randint(1, 9)}"
        tools = [
            _tool_spec(
                "lookup_cve_advisory",
                "Fetch CVE severity and patched package version.",
                {"cve_id": {"type": "string"}, "ecosystem": {"type": "string"}},
                ["cve_id", "ecosystem"],
            ),
            _tool_spec(
                "list_open_prs",
                "List open pull requests.",
                {"repo": {"type": "string"}},
                ["repo"],
            ),
        ]
        u_msg = f"Look up `{cve_num}` in the `pypi` ecosystem for `{svc_name}` dependency `{pkg}`."
        a_call = format_tool_call("lookup_cve_advisory", {"cve_id": cve_num, "ecosystem": "pypi"})
        t_resp = format_tool_response({"cve_id": cve_num, "package": pkg, "cvss": 8.1, "patched_version": fixed_v})
        a_final = f"`{cve_num}` affects `{pkg}` (**CVSS 8.1 High**). Upgrade `{pkg}` to **`>={fixed_v}`**."
    else:
        difficulty = "medium"
        quota_val = rng.randint(500, 25000)
        used_val = rng.randint(100, quota_val)
        acct_code = f"{tenant}_{rng.randint(10, 999)}"
        tools = [
            _tool_spec(
                "get_tenant_quota",
                "Fetch current usage and limit for a tenant.",
                {"tenant_id": {"type": "string"}, "resource": {"type": "string"}},
                ["tenant_id", "resource"],
            ),
            _tool_spec(
                "delete_tenant_workspace",
                "Permanently delete a tenant workspace.",
                {"tenant_id": {"type": "string"}},
                ["tenant_id"],
            ),
        ]
        u_msg = f"How much `{table_ctx}` quota is tenant `{acct_code}` using in `{svc_name}` (`{env}`)?"
        a_call = format_tool_call("get_tenant_quota", {"tenant_id": acct_code, "resource": table_ctx})
        t_resp = format_tool_response({"tenant_id": acct_code, "resource": table_ctx, "used": used_val, "limit": quota_val})
        rem = quota_val - used_val
        a_final = f"Tenant `{acct_code}` is using **`{used_val}` / `{quota_val}`** `{table_ctx}` units (`{rem}` remaining)."

    return make_record(
        record_id=f"td-tool1-{idx:06d}",
        domain="agent_use",
        category="single_tool_precision_calling",
        difficulty=difficulty,
        style="agent_trajectory",
        tags=["tool-calling", "function-calling", "grounded"],
        quality_check="tool_schema_verified",
        messages=[
            {"role": "system", "content": build_agent_system(rng, tools)},
            {"role": "user", "content": u_msg},
            {"role": "assistant", "content": a_call},
            {"role": "tool", "content": t_resp},
            {"role": "assistant", "content": a_final},
        ],
    )


# ---------------------------------------------------------------------------
# 25. parallel_and_chained_tool_use (True Sequential Dependency!)
# ---------------------------------------------------------------------------

def gen_parallel_and_chained_tool_use(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    sha = gen_short_sha(rng)
    deploy_id = gen_uuid_short(rng, "dep")
    author = rng.choice(TENANT_SLUGS)
    changed_files = rng.randint(2, 14)

    tools = [
        _tool_spec(
            "get_active_deployment",
            "Return active deployment ID and git commit SHA for a service/env.",
            {"service": {"type": "string"}, "env": {"type": "string"}},
            ["service", "env"],
        ),
        _tool_spec(
            "get_commit_metadata",
            "Fetch commit author and changed file count by commit SHA.",
            {"service": {"type": "string"}, "commit_sha": {"type": "string"}},
            ["service", "commit_sha"],
        ),
    ]

    inc_ref = f"INC-{idx + 1000}"
    u_msg = f"[{inc_ref}] Find who authored the commit currently deployed in `{env}` for `{svc_name}` and how many files it touched."
    c1 = format_tool_call(
        "get_active_deployment",
        {"service": svc_name, "env": env},
        thought=f"First look up the active deployment in `{env}` to obtain its `commit_sha`.",
    )
    r1 = format_tool_response({"deployment_id": deploy_id, "commit_sha": sha, "status": "healthy"})
    c2 = format_tool_call(
        "get_commit_metadata",
        {"service": svc_name, "commit_sha": sha},
        thought=f"Now query commit metadata for `{sha}`.",
    )
    r2 = format_tool_response({"commit_sha": sha, "author": author, "files_changed": changed_files})
    final = (
        f"`{svc_name}` in `{env}` (deployment `{deploy_id}`) is running commit **`{sha}`**, "
        f"authored by **`{author}`** with **`{changed_files}`** files changed."
    )

    return make_record(
        record_id=f"td-chain-{idx:06d}",
        domain="agent_use",
        category="parallel_and_chained_tool_use",
        difficulty="medium",
        style="agent_trajectory",
        tags=["multi-step-agent", "tool-chaining", "devops"],
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
# 26. agent_error_recovery_and_retry
# ---------------------------------------------------------------------------

def gen_agent_error_recovery_and_retry(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    rec_val = gen_uuid_short(rng, "row")
    variant = idx % 2

    if variant == 0:
        tools = [
            _tool_spec(
                "query_metrics",
                "Query metric series by metric_name and env.",
                {"metric_name": {"type": "string"}, "env": {"type": "string"}},
                ["metric_name", "env"],
            ),
            _tool_spec(
                "list_metric_names",
                "List registered metric names for a service prefix.",
                {"prefix": {"type": "string"}},
                ["prefix"],
            ),
        ]
        real_metric = f"{domain_ctx}_rpc_duration_ms"
        p95 = rng.randint(18, 640)
        alert_code = f"ALR-{idx + 1000}"
        u_msg = f"[{alert_code}] Fetch the `{domain_ctx}` latency metric in `{env}` for `{svc_name}`."
        c1 = format_tool_call("query_metrics", {"metric_name": f"{svc_name}.latency", "env": env})
        r1 = format_tool_response({"error": "METRIC_NOT_FOUND", "hint": f"Use list_metric_names with prefix '{domain_ctx}'"})
        c2 = format_tool_call(
            "list_metric_names",
            {"prefix": domain_ctx},
            thought=f"`{svc_name}.latency` was not found; discover exact registered metric names under `{domain_ctx}`.",
        )
        r2 = format_tool_response({"metrics": [real_metric, f"{domain_ctx}_queue_depth"]})
        c3 = format_tool_call("query_metrics", {"metric_name": real_metric, "env": env})
        r3 = format_tool_response({"metric_name": real_metric, "env": env, "p95_ms": p95})
        final = f"Resolved metric name **`{real_metric}`** in `{env}`: current **p95 is `{p95} ms`**."
        msgs = [
            {"role": "system", "content": build_agent_system(rng, tools)},
            {"role": "user", "content": u_msg},
            {"role": "assistant", "content": c1},
            {"role": "tool", "content": r1},
            {"role": "assistant", "content": c2},
            {"role": "tool", "content": r2},
            {"role": "assistant", "content": c3},
            {"role": "tool", "content": r3},
            {"role": "assistant", "content": final},
        ]
    else:
        tools = [
            _tool_spec(
                "fetch_primary_record",
                "Read record from primary region.",
                {"table": {"type": "string"}, "record_id": {"type": "string"}},
                ["table", "record_id"],
            ),
            _tool_spec(
                "fetch_replica_record",
                "Read record from read-replica when primary is rate-limited or unavailable.",
                {"table": {"type": "string"}, "record_id": {"type": "string"}},
                ["table", "record_id"],
            ),
        ]
        amt = rng.randint(50, 9900)
        u_msg = f"Retrieve `{rec_val}` from `{table_ctx}` in `{svc_name}`."
        c1 = format_tool_call("fetch_primary_record", {"table": table_ctx, "record_id": rec_val})
        r1 = format_tool_response({"error": "HTTP_429_RATE_LIMITED", "retryable": False, "fallback": "fetch_replica_record"})
        c2 = format_tool_call(
            "fetch_replica_record",
            {"table": table_ctx, "record_id": rec_val},
            thought="Primary returned `429`; fall back to `fetch_replica_record`.",
        )
        r2 = format_tool_response({"record_id": rec_val, "amount": amt, "state": "settled"})
        final = f"Retrieved `{rec_val}` from `{table_ctx}` via read-replica: `state='settled'`, `amount={amt}`."
        msgs = [
            {"role": "system", "content": build_agent_system(rng, tools)},
            {"role": "user", "content": u_msg},
            {"role": "assistant", "content": c1},
            {"role": "tool", "content": r1},
            {"role": "assistant", "content": c2},
            {"role": "tool", "content": r2},
            {"role": "assistant", "content": final},
        ]

    return make_record(
        record_id=f"td-errrec-{idx:06d}",
        domain="agent_use",
        category="agent_error_recovery_and_retry",
        difficulty="hard",
        style="agent_trajectory",
        tags=["error-recovery", "resilience", "tool-calling"],
        quality_check="tool_schema_verified",
        messages=msgs,
    )


# ---------------------------------------------------------------------------
# 27. code_repo_navigation_agent
# ---------------------------------------------------------------------------

def gen_code_repo_navigation_agent(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    fpath = f"src/{domain_ctx}/validator.py"
    max_lim = rng.randint(25, 25000)
    tools = [
        _tool_spec("read_file", "Read source file content.", {"path": {"type": "string"}}, ["path"]),
        _tool_spec(
            "edit_file",
            "Replace exact text in a source file.",
            {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
            ["path", "old_text", "new_text"],
        ),
        _tool_spec("run_pytest", "Run pytest target.", {"target": {"type": "string"}}, ["target"]),
    ]

    u_msg = f"Issue `BUG-{idx + 100}` in `{svc_name}` (`{env}`): boundary value `{max_lim}` is wrongly rejected by `check_limit` in `{fpath}` (it should allow `<= {max_lim}`). Fix it and run tests."
    c1 = format_tool_call("read_file", {"path": fpath}, thought=f"Read `{fpath}` before editing.")
    r1 = format_tool_response({"path": fpath, "content": f"def check_limit(n: int) -> bool:\n    return 0 < n < {max_lim}\n"})
    c2 = format_tool_call(
        "edit_file",
        {"path": fpath, "old_text": f"return 0 < n < {max_lim}", "new_text": f"return 0 < n <= {max_lim}"},
        thought=f"Update strict `< {max_lim}` to inclusive `<= {max_lim}`.",
    )
    r2 = format_tool_response({"status": "updated", "path": fpath})
    c3 = format_tool_call("run_pytest", {"target": f"tests/test_{domain_ctx}.py"})
    r3 = format_tool_response({"exit_code": 0, "summary": "6 passed in 0.09s"})
    final = f"Updated `check_limit` in `{fpath}` from `0 < n < {max_lim}` to `0 < n <= {max_lim}` and verified `tests/test_{domain_ctx}.py` (`6 passed`)."

    return make_record(
        record_id=f"td-swe-{idx:06d}",
        domain="agent_use",
        category="code_repo_navigation_agent",
        difficulty="hard",
        style="agent_trajectory",
        tags=["swe-agent", "code-editing", "pytest"],
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
# 28. security_audit_agent_workflow
# ---------------------------------------------------------------------------

def gen_security_audit_agent_workflow(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    fpath = f"src/{domain_ctx}/db_store.py"
    finding_id = f"SAST-{idx + 1000}"

    tools = [
        _tool_spec("run_semgrep_scan", "Run SAST scan on service directory.", {"service": {"type": "string"}}, ["service"]),
        _tool_spec("read_source_snippet", "Read lines around a finding.", {"path": {"type": "string"}, "line": {"type": "integer"}}, ["path", "line"]),
    ]
    line_no = rng.randint(20, 180)
    u_msg = f"Run a SAST scan on `{svc_name}` (`{env}`), inspect any SQLi finding (`{finding_id}`), and confirm whether it is a real vulnerability."
    c1 = format_tool_call("run_semgrep_scan", {"service": svc_name})
    r1 = format_tool_response({"findings": [{"id": finding_id, "rule": "python.sql-injection", "path": fpath, "line": line_no}]})
    c2 = format_tool_call(
        "read_source_snippet",
        {"path": fpath, "line": line_no},
        thought=f"Inspect `{fpath}:{line_no}` to verify whether user input is concatenated.",
    )
    is_tp = idx % 2 == 0
    if is_tp:
        r2 = format_tool_response({"snippet": f"cur.execute(f\"SELECT * FROM {table_ctx} WHERE {id_col}='{{user_key}}'\")"})
        final = (
            f"**Confirmed True Positive (`{finding_id}`, CWE-89)** at `{fpath}:{line_no}`: `user_key` is interpolated via f-string into `SELECT * FROM {table_ctx}`. "
            f"Replace with parameterized `cur.execute('SELECT * FROM {table_ctx} WHERE {id_col} = ?', (user_key,))`."
        )
    else:
        r2 = format_tool_response({"snippet": f"# test fixture constant\ncur.execute('SELECT * FROM {table_ctx} WHERE {id_col} = ?', (user_key,))"})
        final = (
            f"**False Positive (`{finding_id}`)** at `{fpath}:{line_no}`: query already binds `user_key` via parameterized `?` placeholder without string interpolation."
        )

    return make_record(
        record_id=f"td-secagent-{idx:06d}",
        domain="agent_use",
        category="security_audit_agent_workflow",
        difficulty="hard",
        style="agent_trajectory",
        tags=["appsec-agent", "sast-triage", "semgrep"],
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
# 29. missing_params_clarification_agent
# ---------------------------------------------------------------------------

def gen_missing_params_clarification_agent(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env1, env2 = rng.sample(ENVIRONMENTS, 2)
    tenant = rng.choice(TENANT_SLUGS)
    variant = idx % 2

    if variant == 0:
        tools = [
            _tool_spec(
                "purge_partition",
                "Irreversibly delete data partition (requires table, partition_date, env, confirm_ticket).",
                {
                    "table": {"type": "string"},
                    "partition_date": {"type": "string"},
                    "env": {"type": "string"},
                    "confirm_ticket": {"type": "string"},
                },
                ["table", "partition_date", "env", "confirm_ticket"],
            )
        ]
        u_msg = f"Purge the old `{table_ctx}` partition for `{svc_name}` right away (`ref #{idx + 1000}`)."
        a_msg = (
            f"`purge_partition` is irreversible and missing required parameters. Before proceeding, please specify:\n"
            f"1. **Target environment** (e.g. `{env1}` or `{env2}`)\n"
            f"2. **Exact `partition_date`** (`YYYY-MM-DD`) to delete from `{table_ctx}`\n"
            f"3. Confirmation that `ref #{idx + 1000}` is the approved change ticket."
        )
    else:
        tools = [
            _tool_spec(
                "revoke_api_credential",
                "Immediately revoke a live credential.",
                {"tenant_id": {"type": "string"}, "key_id": {"type": "string"}, "env": {"type": "string"}},
                ["tenant_id", "key_id", "env"],
            )
        ]
        u_msg = f"Revoke `{tenant}`'s leaked API key on `{svc_name}` (`incident-{idx + 1000}`)."
        a_msg = (
            f"To avoid revoking the wrong active key for `{tenant}`, please provide:\n"
            f"1. The exact **`key_id`** to revoke\n"
            f"2. The target **`env`** (e.g. `{env1}` vs `{env2}`)."
        )

    return make_record(
        record_id=f"td-clarify-{idx:06d}",
        domain="agent_use",
        category="missing_params_clarification_agent",
        difficulty="easy",
        style="direct",
        tags=["agent-safety", "clarification", "destructive-guard"],
        quality_check="tool_schema_verified",
        messages=[
            {"role": "system", "content": build_agent_system(rng, tools)},
            {"role": "user", "content": u_msg},
            {"role": "assistant", "content": a_msg},
        ],
    )
