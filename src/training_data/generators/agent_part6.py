"""
Agent Fine-Tuning Generators Part 6 (Categories 68–75):
Strict JSON Schema Extraction, SQL Self-Correction, Cost/Latency-Aware Tool Routing,
Human-in-the-Loop Approval Gate, Timezone/DST Temporal Reasoning,
Analytics + Vega-Lite Report Builder, Agent Process-Reward Judge (PRM),
and Structured Table / Runbook Synthesis Agent.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import random
from typing import Any
from zoneinfo import ZoneInfo

from training_data.schema import (
    PLANNING_AGENT_SYSTEM,
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
# 68. agent_json_schema_strict_extraction
# ---------------------------------------------------------------------------

def gen_agent_json_schema_strict_extraction(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    owner = rng.choice(USERNAMES)
    sev = rng.choice(["SEV-1", "SEV-2", "SEV-3"])
    dur_min = rng.randint(6, 185)
    inc_id = f"INC-{idx + 1000}"

    u_msg = (
        f"Extract structured post-mortem metadata as a ```json``` block with exact keys "
        f"`incident_id` (`str`), `service` (`str`), `severity` (`str`), `duration_minutes` (`int`), `owner` (`str`), and `customer_impact` (`bool`) from:\n"
        f"> \"Post-mortem `{inc_id}`: `{svc_name}` experienced a `{sev}` outage lasting `{dur_min}` minutes; "
        f"lead responder `{owner}` confirmed external customer impact.\""
    )
    payload = {
        "incident_id": inc_id,
        "service": svc_name,
        "severity": sev,
        "duration_minutes": dur_min,
        "owner": owner,
        "customer_impact": True,
    }
    ans = (
        f"<thought>\n"
        f"Map all 6 required schema fields strictly from the source note (`{inc_id}`, `{svc_name}`, `{sev}`, `{dur_min}` as integer, `{owner}`, `True`).\n"
        f"</thought>\n\n"
        f"```json\n{json.dumps(payload, indent=2)}\n```"
    )

    return make_record(
        record_id=f"td-jext-{idx:07d}",
        domain="agent_use",
        category="agent_json_schema_strict_extraction",
        difficulty="easy",
        style="ultra_reasoning",
        tags=["structured-output", "json-schema", "information-extraction"],
        quality_check="json_block_verified",
        messages=[
            {"role": "system", "content": rng.choice(PLANNING_AGENT_SYSTEM)},
            {"role": "user", "content": u_msg},
            {"role": "assistant", "content": ans},
        ],
    )


# ---------------------------------------------------------------------------
# 69. agent_sql_analytics_self_correction
# ---------------------------------------------------------------------------

def gen_agent_sql_analytics_self_correction(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    tenant = rng.choice(TENANT_SLUGS)
    total_cents = rng.randint(15_000, 990_000)

    tools = [
        _tool_spec("run_readonly_sql", "Execute read-only SQL query.", {"sql": {"type": "string"}}, ["sql"]),
    ]

    u_msg = f"[SQL-FIX-{idx + 100}] Compute total `net_cents` for tenant `{tenant}` in `{table_ctx}` (`{svc_name}`)."
    bad_sql = f"SELECT SUM(amount_cents) FROM {table_ctx} WHERE tenant = '{tenant}';"
    good_sql = f"SELECT COALESCE(SUM(net_cents), 0) AS total_net_cents FROM {table_ctx} WHERE tenant_slug = '{tenant}';"

    c1 = format_tool_call("run_readonly_sql", {"sql": bad_sql}, thought=f"<thought>\nQuery `{table_ctx}` for `{tenant}` total.\n</thought>")
    r1 = format_tool_response(
        {"error": f"OperationalError: no such column: amount_cents. Available columns on {table_ctx}: [{id_col}, tenant_slug, net_cents, created_at]"}
    )
    c2 = format_tool_call(
        "run_readonly_sql",
        {"sql": good_sql},
        thought=f"<thought>\nSchema diagnostic shows columns are `tenant_slug` and `net_cents`. Rewrite query with `COALESCE(SUM(net_cents), 0)`.\n</thought>",
    )
    r2 = format_tool_response({"rows": [{"total_net_cents": total_cents}]})
    final = f"Total `net_cents` for `{tenant}` in `{table_ctx}` is **`{total_cents}`** cents."

    return make_record(
        record_id=f"td-sqlfix-{idx:07d}",
        domain="agent_use",
        category="agent_sql_analytics_self_correction",
        difficulty="medium",
        style="agent_trajectory",
        tags=["sql-agent", "self-correction", "error-recovery"],
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
# 70. agent_cost_and_latency_budget_routing
# ---------------------------------------------------------------------------

def gen_agent_cost_and_latency_budget_routing(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    tenant = rng.choice(TENANT_SLUGS)
    key_id = f"{tenant}:{idx + 500}"
    val_num = rng.randint(10, 9999)

    tools = [
        _tool_spec("edge_kv_get", "Read hot cache (latency ~4ms, cost $0.0001).", {"key": {"type": "string"}}, ["key"]),
        _tool_spec("cold_archive_scan", "Full glacier scan (latency ~18000ms, cost $1.25).", {"prefix": {"type": "string"}}, ["prefix"]),
    ]

    u_msg = f"Under strict interactive budget (`latency <= 50 ms`, `cost <= $0.01`), fetch `{key_id}` score for `{svc_name}`."
    c1 = format_tool_call(
        "edge_kv_get",
        {"key": key_id},
        thought="<thought>\n`cold_archive_scan` (`18,000 ms`, `$1.25`) violates both `50 ms` and `$0.01` budgets. Route to `edge_kv_get` (`~4 ms`, `$0.0001`).\n</thought>",
    )
    r1 = format_tool_response({"key": key_id, "hit": True, "score": val_num, "latency_ms": 3.8})
    final = f"`{key_id}` score is **`{val_num}`** (retrieved via `edge_kv_get` in `3.8 ms`, within the `<= 50 ms` / `$0.01` budget)."

    return make_record(
        record_id=f"td-budget-{idx:07d}",
        domain="agent_use",
        category="agent_cost_and_latency_budget_routing",
        difficulty="easy",
        style="agent_trajectory",
        tags=["cost-routing", "latency-budget", "tool-selection"],
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
# 71. agent_human_in_the_loop_approval_gate
# ---------------------------------------------------------------------------

def gen_agent_human_in_the_loop_approval_gate(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    part_name = f"{table_ctx}_2024_q{(idx % 4) + 1}"
    row_cnt = rng.randint(40_000, 950_000)
    tok = f"APPR-{idx + 2000}"

    tools = [
        _tool_spec("dry_run_drop_partition", "Preview row count and disk impact before dropping partition.", {"partition": {"type": "string"}, "env": {"type": "string"}}, ["partition", "env"]),
        _tool_spec("execute_drop_partition", "Irreversibly drop partition (requires explicit human approval_token).", {"partition": {"type": "string"}, "env": {"type": "string"}, "approval_token": {"type": "string"}}, ["partition", "env", "approval_token"]),
    ]

    u1 = f"Drop archive partition `{part_name}` in `{svc_name}` (`{env}`)."
    c1 = format_tool_call(
        "dry_run_drop_partition",
        {"partition": part_name, "env": env},
        thought=f"<thought>\nDropping `{part_name}` is irreversible. Run `dry_run_drop_partition` first and request human approval before mutating `{env}`.\n</thought>",
    )
    r1 = format_tool_response({"partition": part_name, "env": env, "rows_affected": row_cnt, "reclaim_gb": 14.2})
    a_ask = (
        f"Dry-run impact for dropping `{part_name}` in `{env}`: **`{row_cnt}` rows** (`14.2 GB` reclaimed). "
        "Please reply with an `approval_token` to confirm permanent deletion."
    )
    u2 = f"Confirmed — proceed with approval token `{tok}`."
    c2 = format_tool_call(
        "execute_drop_partition",
        {"partition": part_name, "env": env, "approval_token": tok},
        thought=f"<thought>\nUser explicitly confirmed and supplied `{tok}`. Proceed with `execute_drop_partition`.\n</thought>",
    )
    r2 = format_tool_response({"partition": part_name, "status": "dropped", "approval_token": tok})
    final = f"Dropped partition `{part_name}` in `{env}` (`{row_cnt}` rows, `14.2 GB` freed) under approval `{tok}`."

    return make_record(
        record_id=f"td-hitl-{idx:07d}",
        domain="agent_use",
        category="agent_human_in_the_loop_approval_gate",
        difficulty="hard",
        style="agent_trajectory",
        tags=["human-in-the-loop", "safety-gate", "dry-run"],
        quality_check="tool_schema_verified",
        messages=[
            {"role": "system", "content": build_agent_system(rng, tools)},
            {"role": "user", "content": u1},
            {"role": "assistant", "content": c1},
            {"role": "tool", "content": r1},
            {"role": "assistant", "content": a_ask},
            {"role": "user", "content": u2},
            {"role": "assistant", "content": c2},
            {"role": "tool", "content": r2},
            {"role": "assistant", "content": final},
        ],
    )


# ---------------------------------------------------------------------------
# 72. agent_time_zone_and_temporal_reasoning (Verified via Python zoneinfo!)
# ---------------------------------------------------------------------------

def gen_agent_time_zone_and_temporal_reasoning(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    day = rng.randint(1, 25)
    hour_utc = rng.randint(0, 23)
    dur_h = rng.randint(1, 4)
    tz_name = rng.choice(["America/New_York", "Europe/London", "Asia/Tokyo", "America/Los_Angeles"])

    dt_utc = datetime(2026, 6, day, hour_utc, 0, tzinfo=timezone.utc)
    dt_end_utc = dt_utc + timedelta(hours=dur_h)
    dt_local = dt_utc.astimezone(ZoneInfo(tz_name))
    dt_end_local = dt_end_utc.astimezone(ZoneInfo(tz_name))

    tools = [
        _tool_spec("schedule_maintenance", "Register maintenance window in ISO-8601 UTC.", {"service": {"type": "string"}, "start_utc": {"type": "string"}, "end_utc": {"type": "string"}}, ["service", "start_utc", "end_utc"]),
    ]

    local_str = dt_local.strftime("%Y-%m-%d %H:%M %Z")
    start_iso = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso = dt_end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    u_msg = f"[TZ-{idx + 100}] Schedule a `{dur_h}`-hour maintenance window for `{svc_name}` starting at `{local_str}` (`{tz_name}`)."
    c1 = format_tool_call(
        "schedule_maintenance",
        {"service": svc_name, "start_utc": start_iso, "end_utc": end_iso},
        thought=f"<thought>\nConvert `{local_str}` (`{tz_name}`) to UTC: `{start_iso}`, plus `{dur_h}h` = `{end_iso}`.\n</thought>",
    )
    r1 = format_tool_response({"window_id": f"MW-{idx + 100}", "start_utc": start_iso, "end_utc": end_iso, "status": "scheduled"})
    final = (
        f"Scheduled `{svc_name}` maintenance (`MW-{idx + 100}`): **`{start_iso}` – `{end_iso}`** "
        f"(`{dt_local.strftime('%Y-%m-%d %H:%M %Z')}` – `{dt_end_local.strftime('%Y-%m-%d %H:%M %Z')}`)."
    )

    return make_record(
        record_id=f"td-tz-{idx:07d}",
        domain="agent_use",
        category="agent_time_zone_and_temporal_reasoning",
        difficulty="medium",
        style="agent_trajectory",
        tags=["timezone", "temporal-reasoning", "scheduling"],
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
# 73. agent_data_visualization_and_report_builder
# ---------------------------------------------------------------------------

def gen_agent_data_visualization_and_report_builder(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    q1 = rng.randint(120, 600)
    q2 = rng.randint(150, 750)
    delta_pct = round(((q2 - q1) / q1) * 100.0, 1)

    tools = [
        _tool_spec("fetch_quarterly_kpi", "Fetch Q1 and Q2 throughput KPIs.", {"service": {"type": "string"}}, ["service"]),
    ]

    u_msg = f"[VIZ-{idx + 100}] Fetch Q1/Q2 throughput for `{svc_name}`, report the exact percentage change, and output a Vega-Lite bar-chart ```json``` spec."
    c1 = format_tool_call("fetch_quarterly_kpi", {"service": svc_name}, thought=f"<thought>\nFetch `{svc_name}` Q1/Q2 KPIs first.\n</thought>")
    r1 = format_tool_response({"service": svc_name, "Q1_rps": q1, "Q2_rps": q2})
    vega = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": [{"quarter": "Q1", "rps": q1}, {"quarter": "Q2", "rps": q2}]},
        "mark": "bar",
        "encoding": {
            "x": {"field": "quarter", "type": "nominal"},
            "y": {"field": "rps", "type": "quantitative"},
        },
    }
    sign = "+" if delta_pct >= 0 else ""
    final = (
        f"`{svc_name}` throughput changed from **`{q1}`** to **`{q2}` req/s** (**`{sign}{delta_pct}%`**).\n\n"
        f"```json\n{json.dumps(vega, indent=2)}\n```"
    )

    return make_record(
        record_id=f"td-viz-{idx:07d}",
        domain="agent_use",
        category="agent_data_visualization_and_report_builder",
        difficulty="medium",
        style="agent_trajectory",
        tags=["analytics-agent", "vega-lite", "reporting"],
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
# 74. agent_trajectory_judge_and_reward_critique (Process-Reward-Model / PRM!)
# ---------------------------------------------------------------------------

def gen_agent_trajectory_judge_and_reward_critique(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    is_flawed = idx % 2 == 0

    if is_flawed:
        trace_desc = (
            f"User: 'Check CPU usage for `{svc_name}` in `{env}`.'\n"
            f"Agent Step 1: `<tool_call>{{\"name\":\"terminate_all_pods\",\"arguments\":{{\"service\":\"{svc_name}\",\"env\":\"{env}\"}}}}</tool_call>`"
        )
        verdict_obj = {
            "trace_id": f"PRM-{idx + 100}",
            "pass": False,
            "step_1_score": 0.0,
            "defect_category": "UNSAFE_DESTRUCTIVE_TOOL_FOR_READ_QUERY",
            "minimal_fix": f"Invoke read-only `get_cpu_metrics(service='{svc_name}', env='{env}')` instead of `terminate_all_pods`.",
        }
        thought_txt = "User asked a strictly read-only diagnostic question ('Check CPU usage'), yet Step 1 invoked destructive `terminate_all_pods`. Score `0.0`."
    else:
        trace_desc = (
            f"User: 'Check CPU usage for `{svc_name}` in `{env}`.'\n"
            f"Agent Step 1: `<tool_call>{{\"name\":\"get_cpu_metrics\",\"arguments\":{{\"service\":\"{svc_name}\",\"env\":\"{env}\"}}}}</tool_call>`\n"
            "Tool Output: `{\"cpu_pct\": 41.5}`\n"
            f"Agent Step 2: '`{svc_name}` CPU utilization in `{env}` is `41.5%`.'"
        )
        verdict_obj = {
            "trace_id": f"PRM-{idx + 100}",
            "pass": True,
            "step_1_score": 1.0,
            "defect_category": "NONE",
            "minimal_fix": "Trajectory is minimal, schema-valid, and faithfully grounded.",
        }
        thought_txt = "Step 1 selects the minimal read-only tool with exact arguments, and Step 2 reports `41.5%` faithfully without fabrication. Score `1.0`."

    u_msg = f"Audit agent trajectory `PRM-{idx + 100}` and return a ```json``` process-reward verdict (`trace_id`, `pass`, `step_1_score`, `defect_category`, `minimal_fix`):\n{trace_desc}"
    ans = f"<thought>\n{thought_txt}\n</thought>\n\n```json\n{json.dumps(verdict_obj, indent=2)}\n```"

    return make_record(
        record_id=f"td-prm-{idx:07d}",
        domain="agent_use",
        category="agent_trajectory_judge_and_reward_critique",
        difficulty="hard",
        style="ultra_reasoning",
        tags=["prm", "agent-judge", "reward-modeling"],
        quality_check="json_block_verified",
        messages=[
            {"role": "system", "content": rng.choice(PLANNING_AGENT_SYSTEM)},
            {"role": "user", "content": u_msg},
            {"role": "assistant", "content": ans},
        ],
    )


# ---------------------------------------------------------------------------
# 75. agent_runbook_table_and_sla_synthesis
# ---------------------------------------------------------------------------

def gen_agent_runbook_table_and_sla_synthesis(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    rto_m = rng.randint(5, 30)
    rpo_s = rng.randint(0, 60)
    oncall = rng.choice(USERNAMES)

    tools = [
        _tool_spec("get_dr_tier_matrix", "Fetch disaster-recovery RTO/RPO table for service.", {"service": {"type": "string"}, "env": {"type": "string"}}, ["service", "env"]),
        _tool_spec("get_active_oncall", "Fetch current primary on-call engineer.", {"service": {"type": "string"}}, ["service"]),
    ]

    u_msg = f"[DR-{idx + 100}] Build a concise DR readiness summary for `{svc_name}` (`{env}`) combining its RTO/RPO targets and current primary on-call."
    c1 = format_tool_call("get_dr_tier_matrix", {"service": svc_name, "env": env}, thought=f"<thought>\nFetch `{svc_name}` DR matrix in `{env}` first.\n</thought>")
    r1 = format_tool_response({"service": svc_name, "env": env, "rto_minutes": rto_m, "rpo_seconds": rpo_s})
    c2 = format_tool_call("get_active_oncall", {"service": svc_name}, thought=f"<thought>\nFetch current primary on-call engineer for `{svc_name}`.\n</thought>")
    r2 = format_tool_response({"service": svc_name, "primary_oncall": oncall})
    final = (
        f"### DR Readiness — `{svc_name}` (`{env}`)\n"
        f"| Metric | Verified Value |\n"
        f"| :--- | :--- |\n"
        f"| **RTO Target** | `{rto_m} minutes` |\n"
        f"| **RPO Target** | `{rpo_s} seconds` |\n"
        f"| **Primary On-Call** | `{oncall}` |"
    )

    return make_record(
        record_id=f"td-drsyn-{idx:07d}",
        domain="agent_use",
        category="agent_runbook_table_and_sla_synthesis",
        difficulty="easy",
        style="agent_trajectory",
        tags=["runbook-agent", "synthesis", "sre"],
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
