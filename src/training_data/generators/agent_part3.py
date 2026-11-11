"""Agent Use Generators Part 3 (Categories 47–50): Parallel Tool Fanout, Multi-Turn User+Agent, Cursor Pagination, Guardrail Policy Gate."""

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
    gen_short_sha,
)


def _tool_spec(name: str, desc: str, props: dict[str, Any], req: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "description": desc,
        "parameters": {"type": "object", "properties": props, "required": req},
    }


# ---------------------------------------------------------------------------
# 47. parallel_multi_tool_fanout
# ---------------------------------------------------------------------------

def gen_parallel_multi_tool_fanout(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env_a, env_b = rng.sample(ENVIRONMENTS, 2)
    lat_a, lat_b = rng.randint(18, 320), rng.randint(25, 490)

    tools = [
        _tool_spec(
            "get_region_latency",
            "Fetch p95 latency (ms) for a service in a region.",
            {"service": {"type": "string"}, "env": {"type": "string"}},
            ["service", "env"],
        )
    ]

    u_msg = f"[CMP-{idx + 100}] Compare p95 latency for `{svc_name}` across `{env_a}` and `{env_b}`."
    a_calls = format_parallel_tool_calls(
        [
            ("get_region_latency", {"service": svc_name, "env": env_a}),
            ("get_region_latency", {"service": svc_name, "env": env_b}),
        ],
        thought=f"Query `{env_a}` and `{env_b}` in parallel since they are independent.",
    )
    r_a = format_tool_response({"env": env_a, "p95_ms": lat_a})
    r_b = format_tool_response({"env": env_b, "p95_ms": lat_b})
    delta = lat_b - lat_a
    faster = env_a if lat_a <= lat_b else env_b
    final = (
        f"`{svc_name}` p95 latency: **`{env_a}` = `{lat_a} ms`**, **`{env_b}` = `{lat_b} ms`** "
        f"(`{faster}` is faster by `{abs(delta)} ms`)."
    )

    return make_record(
        record_id=f"td-fanout-{idx:07d}",
        domain="agent_use",
        category="parallel_multi_tool_fanout",
        difficulty="medium",
        style="agent_trajectory",
        tags=["parallel-tool-calls", "fanout", "sre"],
        quality_check="tool_schema_verified",
        messages=[
            {"role": "system", "content": build_agent_system(rng, tools)},
            {"role": "user", "content": u_msg},
            {"role": "assistant", "content": a_calls},
            {"role": "tool", "content": r_a},
            {"role": "tool", "content": r_b},
            {"role": "assistant", "content": final},
        ],
    )


# ---------------------------------------------------------------------------
# 48. multi_turn_user_agent_collaboration
# ---------------------------------------------------------------------------

def gen_multi_turn_user_agent_collaboration(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    flag_name = f"enable_{domain_ctx}_v2_{idx + 10}"
    pct = rng.choice([10, 25, 50, 100])

    tools = [
        _tool_spec("get_flag_state", "Inspect feature flag rollout %.", {"flag": {"type": "string"}, "env": {"type": "string"}}, ["flag", "env"]),
        _tool_spec("set_flag_rollout", "Update feature flag rollout %.", {"flag": {"type": "string"}, "env": {"type": "string"}, "pct": {"type": "integer"}}, ["flag", "env", "pct"]),
    ]

    u1 = f"What is the current rollout percentage of `{flag_name}` in `{env}` for `{svc_name}`?"
    c1 = format_tool_call("get_flag_state", {"flag": flag_name, "env": env})
    r1 = format_tool_response({"flag": flag_name, "env": env, "pct": 5})
    a1 = f"`{flag_name}` is currently at **`5%`** in `{env}`."
    u2 = f"Bump it to `{pct}%` in `{env}`."
    c2 = format_tool_call("set_flag_rollout", {"flag": flag_name, "env": env, "pct": pct})
    r2 = format_tool_response({"flag": flag_name, "env": env, "previous_pct": 5, "pct": pct, "status": "applied"})
    a2 = f"Updated `{flag_name}` in `{env}` from `5%` to **`{pct}%`**."

    return make_record(
        record_id=f"td-mtag-{idx:07d}",
        domain="agent_use",
        category="multi_turn_user_agent_collaboration",
        difficulty="medium",
        style="agent_trajectory",
        tags=["multi-turn-agent", "feature-flags", "tool-calling"],
        quality_check="tool_schema_verified",
        messages=[
            {"role": "system", "content": build_agent_system(rng, tools)},
            {"role": "user", "content": u1},
            {"role": "assistant", "content": c1},
            {"role": "tool", "content": r1},
            {"role": "assistant", "content": a1},
            {"role": "user", "content": u2},
            {"role": "assistant", "content": c2},
            {"role": "tool", "content": r2},
            {"role": "assistant", "content": a2},
        ],
    )


# ---------------------------------------------------------------------------
# 49. context_budget_and_pagination_agent
# ---------------------------------------------------------------------------

def gen_context_budget_and_pagination_agent(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    tenant = f"{rng.choice(TENANT_SLUGS)}_{idx + 10}"
    v1, v2, v3 = rng.randint(10, 90), rng.randint(10, 90), rng.randint(10, 90)
    cur_tok = f"cur_{gen_short_sha(rng)}"

    tools = [
        _tool_spec(
            "list_failed_jobs",
            "List failed job error counts with cursor pagination.",
            {"tenant": {"type": "string"}, "cursor": {"type": "string"}},
            ["tenant"],
        )
    ]

    u_msg = f"Sum `failures` across **all** pages of failed jobs for tenant `{tenant}` in `{svc_name}`."
    c1 = format_tool_call("list_failed_jobs", {"tenant": tenant})
    r1 = format_tool_response({"items": [{"failures": v1}, {"failures": v2}], "next_cursor": cur_tok, "has_more": True})
    c2 = format_tool_call("list_failed_jobs", {"tenant": tenant, "cursor": cur_tok}, thought="`has_more` is true; fetch the remaining page.")
    r2 = format_tool_response({"items": [{"failures": v3}], "next_cursor": None, "has_more": False})
    final = f"Across both pages (`3` items total), tenant `{tenant}` has **`{v1 + v2 + v3}`** total failures (`{v1} + {v2} + {v3}`)."

    return make_record(
        record_id=f"td-pageag-{idx:07d}",
        domain="agent_use",
        category="context_budget_and_pagination_agent",
        difficulty="medium",
        style="agent_trajectory",
        tags=["pagination", "cursor-traversal", "tool-calling"],
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
# 50. guardrail_policy_check_agent
# ---------------------------------------------------------------------------

def gen_guardrail_policy_check_agent(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    sha = gen_short_sha(rng)
    frozen = idx % 2 == 0

    tools = [
        _tool_spec("check_change_window", "Check if deployments are allowed in env.", {"env": {"type": "string"}}, ["env"]),
        _tool_spec("promote_release", "Promote commit SHA to env (forbidden during freeze).", {"service": {"type": "string"}, "env": {"type": "string"}, "sha": {"type": "string"}}, ["service", "env", "sha"]),
    ]

    u_msg = f"[REL-{idx + 100}] Promote `{svc_name}` commit `{sha}` to `{env}`, respecting change-freeze policy."
    c1 = format_tool_call("check_change_window", {"env": env}, thought=f"Verify `{env}` change-window policy before calling `promote_release`.")
    if frozen:
        r1 = format_tool_response({"env": env, "frozen": True, "reason": "PCI quarterly close freeze"})
        msgs = [
            {"role": "system", "content": build_agent_system(rng, tools)},
            {"role": "user", "content": u_msg},
            {"role": "assistant", "content": c1},
            {"role": "tool", "content": r1},
            {
                "role": "assistant",
                "content": f"**Promotion Halted**: `{env}` is currently under an active change freeze (`PCI quarterly close freeze`). Did not invoke `promote_release` for `{sha}`.",
            },
        ]
    else:
        r1 = format_tool_response({"env": env, "frozen": False})
        c2 = format_tool_call("promote_release", {"service": svc_name, "env": env, "sha": sha})
        r2 = format_tool_response({"status": "promoted", "service": svc_name, "env": env, "sha": sha})
        msgs = [
            {"role": "system", "content": build_agent_system(rng, tools)},
            {"role": "user", "content": u_msg},
            {"role": "assistant", "content": c1},
            {"role": "tool", "content": r1},
            {"role": "assistant", "content": c2},
            {"role": "tool", "content": r2},
            {"role": "assistant", "content": f"Verified `{env}` window is open and promoted `{svc_name}` commit **`{sha}`**."},
        ]

    return make_record(
        record_id=f"td-guard-{idx:07d}",
        domain="agent_use",
        category="guardrail_policy_check_agent",
        difficulty="hard",
        style="agent_trajectory",
        tags=["guardrails", "policy-compliance", "deployment-safety"],
        quality_check="tool_schema_verified",
        messages=msgs,
    )
