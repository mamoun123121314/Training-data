"""
Agent Fine-Tuning Generators Part 5 (Categories 60–67):
No-Tool Restraint (Anti-Overcalling), 8-Tool Disambiguation, REST/OpenAPI Client,
Browser Accessibility DOM Navigation, Smart Large-File Search, Transactional Rollback,
PII/Secret Hygiene, and Anti-Hallucination Abstention.
"""

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
    ENVIRONMENTS,
    SERVICES,
    TENANT_SLUGS,
    USERNAMES,
)


def _tool_spec(name: str, desc: str, props: dict[str, Any], req: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "description": desc,
        "parameters": {"type": "object", "properties": props, "required": req},
    }


# ---------------------------------------------------------------------------
# 60. agent_no_tool_needed_restraint (Prevents compulsive tool over-calling!)
# ---------------------------------------------------------------------------

def gen_agent_no_tool_needed_restraint(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    items = [rng.randint(10, 400) for _ in range(5)]
    threshold = rng.randint(80, 250)
    filtered = [x for x in items if x >= threshold]
    total = sum(filtered)

    tools = [
        _tool_spec("query_production_db", "Run SQL against production cluster.", {"sql": {"type": "string"}}, ["sql"]),
        _tool_spec("restart_service", "Restart production pods.", {"service": {"type": "string"}}, ["service"]),
    ]

    u_msg = (
        f"[SELF-{idx + 100}] Given this inline `{svc_name}` batch `{items}`, what is the sum of elements `>= {threshold}`? "
        "Do not query external systems if inline data suffices."
    )
    ans = (
        f"<thought>\n"
        f"The complete input list `{items}` and threshold `{threshold}` are already provided inline. "
        f"Calling `query_production_db` or `restart_service` is unnecessary and wasteful. "
        f"Elements `>= {threshold}`: `{filtered}`, sum = `{total}`.\n"
        f"</thought>\n\n"
        f"Elements `>= {threshold}` in `{items}`: **`{filtered}`** — sum = **`{total}`**."
    )

    return make_record(
        record_id=f"td-notool-{idx:07d}",
        domain="agent_use",
        category="agent_no_tool_needed_restraint",
        difficulty="easy",
        style="ultra_reasoning",
        tags=["tool-restraint", "anti-overcalling", "agent-alignment"],
        quality_check="tool_schema_verified",
        messages=[
            {"role": "system", "content": build_agent_system(rng, tools)},
            {"role": "user", "content": u_msg},
            {"role": "assistant", "content": ans},
        ],
    )


# ---------------------------------------------------------------------------
# 61. agent_ambiguous_tool_disambiguation
# ---------------------------------------------------------------------------

def gen_agent_ambiguous_tool_disambiguation(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    tenant = rng.choice(TENANT_SLUGS)
    inv_id = f"inv_{tenant}_{idx + 1000}"
    amt_cents = rng.randint(1500, 850000)

    tools = [
        _tool_spec("get_invoice_by_number", "Fetch single invoice by human invoice_number (e.g. INV-2026-01).", {"invoice_number": {"type": "string"}}, ["invoice_number"]),
        _tool_spec("get_invoice_by_id", "Fetch single invoice by internal inv_... primary ID.", {"invoice_id": {"type": "string"}}, ["invoice_id"]),
        _tool_spec("list_tenant_invoices", "List all invoices for a tenant_slug.", {"tenant_slug": {"type": "string"}}, ["tenant_slug"]),
        _tool_spec("void_invoice_by_id", "Permanently void an invoice.", {"invoice_id": {"type": "string"}}, ["invoice_id"]),
        _tool_spec("get_payment_intent", "Fetch payment intent by pi_... ID.", {"payment_intent_id": {"type": "string"}}, ["payment_intent_id"]),
    ]

    u_msg = f"Look up the status and `amount_cents` for internal invoice ID `{inv_id}` in `{svc_name}`."
    c1 = format_tool_call(
        "get_invoice_by_id",
        {"invoice_id": inv_id},
        thought=(
            f"<thought>\n"
            f"`{inv_id}` is an internal `inv_...` primary ID (not a human `INV-2026-...` number or `pi_...` intent). "
            f"Select read-only `get_invoice_by_id` and avoid mutating `void_invoice_by_id`.\n"
            f"</thought>"
        ),
    )
    r1 = format_tool_response({"invoice_id": inv_id, "status": "paid", "amount_cents": amt_cents})
    final = f"Invoice `{inv_id}` has `status='paid'` and `amount_cents={amt_cents}`."

    return make_record(
        record_id=f"td-disamb-{idx:07d}",
        domain="agent_use",
        category="agent_ambiguous_tool_disambiguation",
        difficulty="medium",
        style="agent_trajectory",
        tags=["tool-selection", "disambiguation", "schema-precision"],
        quality_check="tool_schema_verified",
        messages=[
            {"role": "system", "content": build_agent_system(rng, tools)},
            {"role": "user", "content": u_msg},
            {"role": "assistant", "content": c1},
            {"role": "tool", "content": r1},
            {"role": "assistant", "content": final},
        ],
    )


# ---------------------------------------------------------------------------
# 62. agent_api_http_graphql_rest_client
# ---------------------------------------------------------------------------

def gen_agent_api_http_graphql_rest_client(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    tenant = rng.choice(TENANT_SLUGS)
    limit_n = rng.choice([25, 50, 100])
    active_cnt = rng.randint(3, limit_n)

    tools = [
        _tool_spec("get_openapi_operation", "Inspect OpenAPI schema for an operationId.", {"operation_id": {"type": "string"}}, ["operation_id"]),
        _tool_spec("http_json_request", "Send authenticated REST request.", {"method": {"type": "string"}, "path": {"type": "string"}, "query": {"type": "object"}}, ["method", "path", "query"]),
    ]

    op_id = f"list_{table_ctx}_v2"
    u_msg = f"[API-{idx + 100}] Inspect the OpenAPI spec for `{op_id}` in `{svc_name}`, then fetch active `{table_ctx}` for tenant `{tenant}` (`limit={limit_n}`)."
    c1 = format_tool_call("get_openapi_operation", {"operation_id": op_id}, thought=f"<thought>\nFetch `{op_id}` OpenAPI spec first to confirm exact route path and query parameter names.\n</thought>")
    route = f"/v2/tenants/{tenant}/{table_ctx}"
    r1 = format_tool_response({"method": "GET", "path_template": f"/v2/tenants/{{tenant_id}}/{table_ctx}", "query_params": ["state", "page_size"]})
    c2 = format_tool_call(
        "http_json_request",
        {"method": "GET", "path": route, "query": {"state": "active", "page_size": limit_n}},
        thought=f"<thought>\nSpec uses `page_size` (not `limit`) and `state='active'` on `{route}`.\n</thought>",
    )
    r2 = format_tool_response({"status_code": 200, "count": active_cnt, "next_cursor": None})
    final = f"Queried `GET {route}?state=active&page_size={limit_n}` per `{op_id}` OpenAPI contract: **`{active_cnt}`** active `{table_ctx}` returned."

    return make_record(
        record_id=f"td-rest-{idx:07d}",
        domain="agent_use",
        category="agent_api_http_graphql_rest_client",
        difficulty="medium",
        style="agent_trajectory",
        tags=["openapi", "rest-client", "tool-calling"],
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
# 63. agent_browser_dom_navigation_workflow (Web / UI Accessibility Agent!)
# ---------------------------------------------------------------------------

def gen_agent_browser_dom_navigation_workflow(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    order_code = f"ORD-{idx + 10000}"
    eta_date = f"2026-11-{rng.randint(12, 28):02d}"

    tools = [
        _tool_spec("browser_snapshot", "Capture accessibility tree with element refs (@e1, @e2...).", {}, []),
        _tool_spec("browser_fill_and_submit", "Fill input ref and click submit button ref.", {"input_ref": {"type": "string"}, "text": {"type": "string"}, "submit_ref": {"type": "string"}}, ["input_ref", "text", "submit_ref"]),
    ]

    u_msg = f"On the `{svc_name}` admin portal already open in the browser, look up order `{order_code}` and report its delivery ETA."
    c1 = format_tool_call("browser_snapshot", {}, thought="<thought>\nCapture the accessibility tree first to obtain live `@e...` element refs.\n</thought>")
    r1 = format_tool_response(
        {
            "url": f"https://admin.{svc_name}.internal/orders",
            "elements": [
                {"ref": "@e4", "role": "textbox", "label": "Search Order ID"},
                {"ref": "@e5", "role": "button", "label": "Find Order"},
                {"ref": "@e9", "role": "button", "label": "Purge All Orders"},
            ],
        }
    )
    c2 = format_tool_call(
        "browser_fill_and_submit",
        {"input_ref": "@e4", "text": order_code, "submit_ref": "@e5"},
        thought=f"<thought>\nFill `@e4` ('Search Order ID') with `{order_code}` and click `@e5` ('Find Order'); ignore destructive `@e9`.\n</thought>",
    )
    r2 = format_tool_response({"status": "navigated", "result_card": {"order_id": order_code, "state": "in_transit", "eta": eta_date}})
    final = f"Order `{order_code}` is **`in_transit`** with delivery ETA **`{eta_date}`**."

    return make_record(
        record_id=f"td-uiag-{idx:07d}",
        domain="agent_use",
        category="agent_browser_dom_navigation_workflow",
        difficulty="medium",
        style="agent_trajectory",
        tags=["browser-agent", "accessibility-tree", "computer-use"],
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
# 64. agent_streaming_chunk_and_large_file_handling
# ---------------------------------------------------------------------------

def gen_agent_streaming_chunk_and_large_file_handling(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    hit_line = rng.randint(18_000, 92_000)
    err_code = f"E_{domain_ctx.upper()}_{idx + 100}"
    log_path = f"/var/log/{svc_name}/app.log"

    tools = [
        _tool_spec("search_in_file", "Search regex/literal in large file and return matching line numbers.", {"path": {"type": "string"}, "pattern": {"type": "string"}}, ["path", "pattern"]),
        _tool_spec("read_file_range", "Read bounded line window from file.", {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, ["path", "offset", "limit"]),
    ]

    u_msg = f"File `{log_path}` has `110,000` lines. Find error code `{err_code}` and report the root `cause` field from its surrounding context."
    c1 = format_tool_call(
        "search_in_file",
        {"path": log_path, "pattern": err_code},
        thought=f"<thought>\nAvoid reading `110,000` lines sequentially; pinpoint exact line numbers for `{err_code}` via `search_in_file`.\n</thought>",
    )
    r1 = format_tool_response({"matches": [{"line": hit_line, "preview": f"ERROR code={err_code}"}]})
    c2 = format_tool_call(
        "read_file_range",
        {"path": log_path, "offset": hit_line - 2, "limit": 6},
        thought=f"<thought>\nRead a tight 6-line window around line `{hit_line}` to extract `cause` without overflowing context.\n</thought>",
    )
    r2 = format_tool_response(
        {"offset": hit_line - 2, "text": f"line {hit_line}: ERROR code={err_code} cause=\"upstream_tls_handshake_expired\" retryable=false"}
    )
    final = f"`{err_code}` appears at `{log_path}:{hit_line}` with `cause=\"upstream_tls_handshake_expired\"` (`retryable=false`)."

    return make_record(
        record_id=f"td-lgfile-{idx:07d}",
        domain="agent_use",
        category="agent_streaming_chunk_and_large_file_handling",
        difficulty="easy",
        style="agent_trajectory",
        tags=["context-efficiency", "large-files", "search-then-read"],
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
# 65. agent_idempotency_and_rollback_safety (Saga Compensating Transaction!)
# ---------------------------------------------------------------------------

def gen_agent_idempotency_and_rollback_safety(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    tenant = rng.choice(TENANT_SLUGS)
    res_id = f"rsv_{idx + 5000}"
    units = rng.randint(2, 40)

    tools = [
        _tool_spec("reserve_capacity", "Place temporary capacity hold.", {"tenant": {"type": "string"}, "units": {"type": "integer"}}, ["tenant", "units"]),
        _tool_spec("commit_billing_charge", "Charge tenant payment method for reservation.", {"reservation_id": {"type": "string"}}, ["reservation_id"]),
        _tool_spec("release_capacity_hold", "Compensating rollback: release held capacity if billing fails.", {"reservation_id": {"type": "string"}}, ["reservation_id"]),
    ]

    u_msg = f"[SAGA-{idx + 100}] Allocate `{units}` units for tenant `{tenant}` in `{svc_name}` and charge their billing method; leave zero orphaned holds on payment failure."
    c1 = format_tool_call("reserve_capacity", {"tenant": tenant, "units": units}, thought=f"<thought>\nStep 1: Reserve `{units}` units for `{tenant}` before charging.\n</thought>")
    r1 = format_tool_response({"reservation_id": res_id, "status": "held", "units": units})
    c2 = format_tool_call("commit_billing_charge", {"reservation_id": res_id}, thought=f"<thought>\nStep 2: Attempt billing capture for `{res_id}`.\n</thought>")
    r2 = format_tool_response({"status": "failed", "error_code": "PAYMENT_METHOD_EXPIRED"})
    c3 = format_tool_call(
        "release_capacity_hold",
        {"reservation_id": res_id},
        thought=f"<thought>\nBilling failed with `PAYMENT_METHOD_EXPIRED`. Execute compensating rollback `release_capacity_hold` on `{res_id}` immediately to prevent leaked capacity.\n</thought>",
    )
    r3 = format_tool_response({"reservation_id": res_id, "status": "released"})
    final = (
        f"Billing capture for `{tenant}` failed (`PAYMENT_METHOD_EXPIRED`), so reservation `{res_id}` (`{units}` units) "
        "was cleanly rolled back (`status='released'`) with zero orphaned holds."
    )

    return make_record(
        record_id=f"td-saga-{idx:07d}",
        domain="agent_use",
        category="agent_idempotency_and_rollback_safety",
        difficulty="hard",
        style="agent_trajectory",
        tags=["saga-rollback", "transactions", "reliability-agent"],
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
# 66. agent_pii_redaction_and_secret_hygiene
# ---------------------------------------------------------------------------

def gen_agent_pii_redaction_and_secret_hygiene(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    fake_tok = f"sk_test_EXAMPLE_{idx + 10000}_DO_NOT_USE"
    db_host = f"{domain_ctx}-db.{env}.internal"

    tools = [
        _tool_spec("inspect_worker_config", "Fetch runtime worker config snapshot.", {"service": {"type": "string"}, "env": {"type": "string"}}, ["service", "env"]),
    ]

    u_msg = f"[SEC-HYG-{idx + 100}] Check which database host `{svc_name}` (`{env}`) is pointing to, without exposing any secret keys in your reply."
    c1 = format_tool_call("inspect_worker_config", {"service": svc_name, "env": env}, thought=f"<thought>\nRead worker config for `{svc_name}` (`{env}`) and scrub any credential fields before responding.\n</thought>")
    r1 = format_tool_response({"DB_HOST": db_host, "DB_PORT": 5432, "API_SECRET_KEY": fake_tok})
    final = f"`{svc_name}` (`{env}`) points to database host **`{db_host}:5432`** (`API_SECRET_KEY`: `[REDACTED]`)."

    return make_record(
        record_id=f"td-redact-{idx:07d}",
        domain="agent_use",
        category="agent_pii_redaction_and_secret_hygiene",
        difficulty="easy",
        style="agent_trajectory",
        tags=["secret-hygiene", "redaction", "safe-agent"],
        quality_check="tool_schema_verified",
        messages=[
            {"role": "system", "content": build_agent_system(rng, tools)},
            {"role": "user", "content": u_msg},
            {"role": "assistant", "content": c1},
            {"role": "tool", "content": r1},
            {"role": "assistant", "content": final},
        ],
    )


# ---------------------------------------------------------------------------
# 67. agent_hallucination_abstention_on_empty_results
# ---------------------------------------------------------------------------

def gen_agent_hallucination_abstention_on_empty_results(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    rfc_id = f"RFC-{idx + 4000}"

    tools = [
        _tool_spec("search_architecture_docs", "Search internal RFCs by ID or keyword.", {"query": {"type": "string"}}, ["query"]),
    ]

    u_msg = f"Summarize the retention SLA approved in `{rfc_id}` for `{svc_name}`."
    c1 = format_tool_call("search_architecture_docs", {"query": f"{rfc_id} {svc_name}"}, thought=f"<thought>\nQuery `search_architecture_docs` for `{rfc_id}` before making any claim about its SLA.\n</thought>")
    r1 = format_tool_response({"query": f"{rfc_id} {svc_name}", "results": [], "total_hits": 0})
    final = (
        f"No document matching **`{rfc_id}`** for `{svc_name}` was found in `search_architecture_docs` (`0` hits), "
        "so I cannot verify an approved retention SLA without guessing. Please share the RFC text or confirm the document ID."
    )

    return make_record(
        record_id=f"td-abstain-{idx:07d}",
        domain="agent_use",
        category="agent_hallucination_abstention_on_empty_results",
        difficulty="medium",
        style="agent_trajectory",
        tags=["anti-hallucination", "epistemic-humility", "grounding"],
        quality_check="tool_schema_verified",
        messages=[
            {"role": "system", "content": build_agent_system(rng, tools)},
            {"role": "user", "content": u_msg},
            {"role": "assistant", "content": c1},
            {"role": "tool", "content": r1},
            {"role": "assistant", "content": final},
        ],
    )
