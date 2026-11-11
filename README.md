# Training-data (`v5.0.0`) — Autonomous Agent SFT, Agent DPO Preference, Coding & Ethical Security Ultra Suite

[![Version](https://img.shields.io/badge/version-5.0.0-purple.svg)](#overview--storage-tiers)
[![Agent_SFT](https://img.shields.io/badge/Agent_Ultra_SFT-1.05%20Billion%20(42%20tracks)-brightgreen.svg)](#all-42-autonomous-agent-fine-tuning-categories)
[![Agent_DPO](https://img.shields.io/badge/Agent_DPO_Pairs-100%20Million%20(8%20taxonomies)-blue.svg)](#stage-2-agent-dpo--orpo-preference-alignment-8-taxonomies)
[![Persisted_In_Repo](https://img.shields.io/badge/Git_Persisted_Shards-1%2C900%2C000%20records%20(111%20MiB)-orange.svg)](#overview--storage-tiers)
[![Access](https://img.shields.io/badge/O(1)_Bijective_Engine-23%20MiB%20RAM-teal.svg)](#quickstart-training--fine-tuning-agents)

**Training-data (`v5.0.0`)** is a complete **2-Stage Post-Training Data Suite** purpose-built for **training and fine-tuning Autonomous AI Agents** (SWE Agents, MCP Tool Agents, Browser/UI Agents, Multi-Agent Orchestrators, Self-Correcting SQL/API Agents, Process-Reward Verifiers, and DPO/ORPO Safety-Aligned Agents), alongside **Coding** and **Ethical Security Engineering**.

The repository is **packed to optimal Git/workspace capacity (`111 MiB` persisted `.jsonl.xz` = `1,900,000` checked-in records / `3.094 GB` raw JSONL)**, backed by **`3,000,000` additional materialized Pure-Agent records on local disk (`out/agent-ultra-pack/`, `6.332 GB` raw)**, and powered by **`O(1)` Cycle-Walking Feistel Bijective Datasets (`2.15 Billion` addressable examples in `23 MiB` RAM)**.

---

## Overview & Storage Tiers

| Dataset Suite | Verified Records | Raw JSONL | Compressed `.jsonl.xz` | Location / Access |
| :--- | ---: | ---: | ---: | :--- |
| **Persisted In-Repo Agent Ultra (`SFT` + `DPO`)** | **`650,000`** (`450k` SFT + `200k` DPO) | **`1.291 GB`** | **`50.93 MiB`** | **`data/agent_ultra/*.jsonl.xz`** *(persisted in repo)* |
| **Persisted In-Repo Core Multi-Domain (`SFT`)** | **`1,250,000`** (`50` categories) | **`1.803 GB`** | **`60.00 MiB`** | **`data/*.jsonl.xz`** *(persisted in repo)* |
| **Total Persisted In-Repo (`data/`)** | **`1,900,000`** | **`3.094 GB`** | **`111.0 MiB`** | Ready immediately on clone/checkout (`0` setup) |
| **Local-Disk Massive Agent Pack (`SFT` + `DPO`)** | **`3,000,000`** (`2.5M` SFT + `500k` DPO) | **`6.332 GB`** | **`280.27 MiB`** | `out/agent-ultra-pack/*.jsonl.xz` + `data/manifest_agent_ultra_3m.json` |
| **`AgentUltraDataset` (`O(1)` Pure-Agent SFT)** | **`1,050,000,000`** (`42` Agent tracks × `25M`) | `~2.28 TB` | `~99 GB` (proj.) | `from training_data import AgentUltraDataset` (`O(1)` RAM) |
| **`AgentDPODataset` (`O(1)` Pure-Agent DPO)** | **`100,000,000`** (`8` taxonomies × `12.5M`) | `~138 GB` | `~3.9 GB` (proj.) | `from training_data import AgentDPODataset` (`O(1)` RAM) |
| **`BillionUltraDataset` (`O(1)` Multi-Domain)** | **`1,000,000,000`** (Coding + Sec + Agent) | **`1.834 TB`** | **`94.83 GB`** (proj.) | `from training_data import BillionUltraDataset` (`O(1)` RAM) |

Every materialized shard (`650,000` in `data/agent_ultra/`, `1,250,000` in `data/`, and `3,000,000` in `out/agent-ultra-pack/`) passed independent zero-trust SHA-256, schema, AST, `<tool_call>`, and global radix-hash deduplication audits (**`PASSED` — `0` duplicate prompts, `0` errors**).

---

## All 42 Autonomous Agent Fine-Tuning Categories (`Stage 1: Agent SFT`)

### A. SWE, Codebase, Git & Build-Loop Agents
1. `swe_multi_file_bugfix_trajectory` — `grep_symbol` $\rightarrow$ parallel `read_file` $\rightarrow$ cross-file unit/contract diagnosis $\rightarrow$ `edit_file` $\rightarrow$ `run_pytest` (`12` msgs).
2. `swe_git_conflict_and_pr_workflow` — `git_status` $\rightarrow$ `read_file` conflict hunks $\rightarrow$ semantic merge preserving both branches $\rightarrow$ `run_ci_suite`.
3. `swe_test_driven_feature_agent` — Red-Green TDD loop: write failing `pytest` first $\rightarrow$ confirm Red failure $\rightarrow$ implement module $\rightarrow$ verify Green pass.
4. `swe_stacktrace_debugger_agent` — Production traceback $\rightarrow$ `read_file_lines` around crashing frame $\rightarrow$ `None`-safe patch $\rightarrow$ `run_repro`.
5. `swe_shell_build_and_linter_loop` — `run_typecheck` (`mypy --strict`) $\rightarrow$ diagnose `[return-value]` diagnostic $\rightarrow$ `edit_file` $\rightarrow$ clean re-check.
6. `code_repo_navigation_agent` — `read_file` $\rightarrow$ boundary-condition `edit_file` $\rightarrow$ `run_pytest`.
7. `security_audit_agent_workflow` — Dependency CVE scan $\rightarrow$ minimal patched lockfile upgrade $\rightarrow$ regression test gate.

### B. Protocols (MCP), Browser/UI, REST/OpenAPI & Large-Context Efficiency
8. `mcp_server_discovery_and_invocation` — **Model Context Protocol (MCP)**: `mcp_list_resources` $\rightarrow$ parallel `mcp_read_resource` + `mcp_call_tool`.
9. `agent_browser_dom_navigation_workflow` — Accessibility-tree `@e...` ref snapshot $\rightarrow$ fill/submit while ignoring destructive UI buttons.
10. `agent_api_http_graphql_rest_client` — `get_openapi_operation` $\rightarrow$ exact parameter mapping (`page_size`, `state`) $\rightarrow$ `http_json_request`.
11. `agent_streaming_chunk_and_large_file_handling` — `110,000`-line logs: `search_in_file` $\rightarrow$ targeted `read_file_range` (`limit=6`) instead of context-blowing dumps.
12. `context_budget_and_pagination_agent` — Iterative `next_cursor` loop until terminal page.

### C. Tool Precision, Restraint, Disambiguation & Parallelism
13. `agent_no_tool_needed_restraint` — **Anti-overcalling**: answers self-contained inline requests directly without wasteful production DB/API calls.
14. `agent_ambiguous_tool_disambiguation` — Selects exact schema match among 5 closely named read/write/ID-type tools (`inv_...` vs `INV-...` vs `pi_...`).
15. `single_tool_precision_calling` — Strict required-field tool invocation while ignoring destructive distractors.
16. `parallel_multi_tool_fanout` — Simultaneous `<tool_call>` batch emission across independent regions/metrics.
17. `parallel_and_chained_tool_use` — Sequential multi-hop dependency propagation.
18. `agent_cost_and_latency_budget_routing` — Routes to `$0.0001 / 4ms` hot KV cache vs `$1.25 / 18s` archive scan under strict SLA budgets.

### D. Dynamic Replanning, Saga Rollbacks, Error Recovery & Self-Correction
19. `agent_long_horizon_plan_and_replan` — Checks live replica lag and dynamically replans from `replica` (`5000`) to `throttled_primary` (`500`).
20. `agent_idempotency_and_rollback_safety` — **Saga compensation**: `reserve_capacity` $\rightarrow$ `commit_billing_charge` fails $\rightarrow$ `release_capacity_hold` leaves zero orphaned holds.
21. `agent_error_recovery_and_retry` — Self-heals from `INVALID_ARGUMENT`, `TIMEOUT`, and `HTTP_429` replica fallback.
22. `agent_sql_analytics_self_correction` — Reads `OperationalError` column hint, rewrites query with `COALESCE`, and verifies result.
23. `sql_database_exploration_agent` — `list_tables` $\rightarrow$ `describe_table` $\rightarrow$ `run_readonly_sql`.
24. `devops_incident_triage_agent` — Alert inspection $\rightarrow$ deploy diff correlation $\rightarrow$ canary rollback $\rightarrow$ recovery check.

### E. Safety, Prompt-Injection Immunity, HITL Gates & Epistemic Grounding
25. `prompt_injection_resistant_agent` — Ignores untrusted tool-output exfiltration/override payloads and finishes only the user's task.
26. `agent_human_in_the_loop_approval_gate` — Runs `dry_run_drop_partition` first, reports exact row blast radius, waits for human `approval_token`, then executes.
27. `missing_params_clarification_agent` — Pauses to ask clarifying questions before underspecified destructive actions.
28. `guardrail_policy_check_agent` — Pre-execution `evaluate_action_policy` gate halting `DENY_FREEZE_WINDOW` mutations.
29. `agent_pii_redaction_and_secret_hygiene` — Scrubs live `API_SECRET_KEY` / DB passwords from tool dumps before replying.
30. `agent_hallucination_abstention_on_empty_results` — Abstains cleanly instead of fabricating facts when search tools return `0` hits.
31. `rag_citation_and_conflict_resolution` — Resolves contradictory docs by timestamp/authority with `[doc_id]` citations.

### F. Memory, Multi-Agent Orchestration, Temporal/Data Synthesis & PRM Judge
32. `agent_memory_and_preferences_retrieval` — Queries `search_workspace_memory` to honor repo conventions (`uv run pytest` vs bare `pytest`).
33. `agent_subagent_orchestration_and_merge` — Parallel `spawn_subagent` (`load_estimator` + `db_capacity_auditor`) and capacity synthesis.
34. `multi_agent_handoff_and_state_tracking` — Structured specialist delegation (`appsec` + `db_review`).
35. `multi_turn_user_agent_collaboration` — Multi-turn interactive user + agent refinement loops.
36. `agent_time_zone_and_temporal_reasoning` — Python `zoneinfo`-verified DST/timezone conversions (`America/New_York`, `Asia/Tokyo`, `Europe/London` $\rightarrow$ ISO-8601 UTC).
37. `sandbox_python_interpreter_agent` — Delegates exact numerical statistics to Python execution.
38. `agent_data_visualization_and_report_builder` — Computes KPI deltas + emits schema-valid Vega-Lite v5 JSON chart specs.
39. `agent_json_schema_strict_extraction` — Unstructured incident prose $\rightarrow$ strict typed JSON extraction.
40. `structured_output_and_planning` — Dependency-ordered release DAG JSON plans.
41. `agent_runbook_table_and_sla_synthesis` — Combines RTO/RPO + live on-call telemetry into structured SRE tables.
42. `agent_trajectory_judge_and_reward_critique` — **Process-Reward Model (PRM) / Judge**: scores step-level trajectories (`0.0` vs `1.0`) and pinpoints minimal fixes.

---

## Stage 2: Agent DPO / ORPO Preference Alignment (`8` Taxonomies)

`data/agent_ultra/dpo_*.jsonl.xz` (`200,000` persisted pairs) + `AgentDPODataset` (`100,000,000` O(1) pairs) provide native TRL `DPOTrainer` conversational records (`prompt`, `chosen`, `rejected`) across:
1. `HALLUCINATED_ID_VS_LIST_FIRST` — Calling `list_*` first vs inventing placeholder UUIDs (`0000-...`).
2. `PROMPT_INJECTION_IMMUNITY_VS_EXFIL` — Ignoring untrusted ticket/doc prompt injections vs calling `send_external_webhook`.
3. `READ_BEFORE_EDIT_VS_BLIND_EDIT` — Reading exact file context before `edit_file` vs guessing `old_text`.
4. `DESTRUCTIVE_CONFIRMATION_VS_UNSAFE_DELETE` — Asking environment/deletion confirmation vs immediate `purge_tenant_ledger`.
5. `NO_TOOL_RESTRAINT_VS_COMPULSIVE_CALL` — Answering self-contained inline math/code vs wasteful warehouse SQL scans.
6. `SCHEMA_EXACT_ARGS_VS_UNDECLARED_ARGS` — Exact schema keys/types (`tenant_slug: str`, `page_size: int`) vs undeclared keys.
7. `SECRET_REDACTION_VS_CREDENTIAL_LEAK` — Redacting embedded DB passwords vs echoing raw `postgres://user:pass@...` URLs.
8. `PARALLEL_FANOUT_VS_SEQUENTIAL_STALL` — Emitting independent read calls in parallel vs sequential round-trip stalls.

---

## Quickstart: Training & Fine-Tuning Agents

### 1. Train Directly on Persisted In-Repo Agent SFT & DPO Shards (`TRL` / `Unsloth` / `HF Datasets`)
```python
from datasets import load_dataset

# Stage 1 — 450,000 persisted Pure-Agent SFT trajectories (42 categories)
agent_sft = load_dataset(
    "json",
    data_files={
        "train": "data/agent_ultra/sft_train-*.jsonl.xz",
        "validation": "data/agent_ultra/sft_validation.jsonl.xz",
        "test": "data/agent_ultra/sft_test.jsonl.xz",
    },
    streaming=True,
)

# Stage 2 — 200,000 persisted Pure-Agent DPO preference pairs (ready for trl.DPOTrainer!)
agent_dpo = load_dataset(
    "json",
    data_files={
        "train": "data/agent_ultra/dpo_train-*.jsonl.xz",
        "validation": "data/agent_ultra/dpo_validation.jsonl.xz",
        "test": "data/agent_ultra/dpo_test.jsonl.xz",
    },
    streaming=True,
)
```

### 2. Stream or Random-Access `1,050,000,000` Agent SFT + `100,000,000` Agent DPO in `O(1)` RAM
```python
import sys
sys.path.insert(0, "src")
from training_data import AgentUltraDataset, AgentDPODataset

sft_1b = AgentUltraDataset(split="train", seed=42)   # len = 1,041,600,000 train (1.05B total)
dpo_100m = AgentDPODataset(seed=42)                  # len = 100,000,000 preference pairs

print(sft_1b[999_999_999]["category"])
print(dpo_100m[88_888_888]["rejection_taxonomy"])
```

---

## Curated Plain-Text Showcases
- [`samples/agent_ultra_showcase.md`](samples/agent_ultra_showcase.md) — **New**: Multi-file SWE, MCP, Browser DOM, Saga Rollback & DPO `chosen` vs `rejected` walkthrough.
- [`samples/ultra_showcase_1b.md`](samples/ultra_showcase_1b.md) — Milestone examples across `[0 .. 999,999,999]`.
- [`samples/preview_1500.jsonl`](samples/preview_1500.jsonl) — `1,500` uncompressed JSONL preview records.
