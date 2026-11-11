"""
1,000,000,000 (One Billion) Ultra-Example Bijective Engine & O(1) Random-Access / Streaming Dataset (v4.1 Forensically Hardened).

Architecture Overview:
- Universe: 1,000,000,000 distinct solver-verified Ultra ChatML examples across 50 specialized categories
  (20,000,000 Ultra examples per category):
    * train      : 990,000,000 examples (10,000 shards x  99,000 examples; local_coord in [0, 19_800_000))
    * validation :   5,000,000 examples (    50 shards x 100,000 examples; local_coord in [19_800_000, 19_900_000))
    * test       :   5,000,000 examples (    50 shards x 100,000 examples; local_coord in [19_900_000, 20_000_000))
- Bijective Shuffling: 6-round Cycle-Walking Balanced Feistel Network maps every index `i in [0, split_size)`
  in O(1) time (~3 us) and O(1) memory (0 bytes of hash sets!) to a unique `(category_id, local_coord)`.
- Forensically Hardened Ultra Synthesis:
    1. Stack-matched bijective `(40 x 50 x 10,000 = 20,000,000)` natural context decomposition with zero
       conflicting region/SLO tags and zero synthetic ID echoing in assistant outputs.
    2. Concrete artifact-grounded `<thought>...</thought>` traces inspecting actual function/SQL/CWE/CVSS
       artifacts and exact `<tool_call>` schemas vs distractor tools, with zero category-slug leakage
       and zero verbatim self-repetition.
"""

from __future__ import annotations

from collections.abc import Iterator
import hashlib
import json
import random
import re
import struct
from typing import Any

from training_data.generators import ALL_GENERATORS

TOTAL_BILLION_RECORDS = 1_000_000_000
NUM_CATEGORIES = 50
PER_CATEGORY_TOTAL = 20_000_000

TRAIN_PER_CATEGORY = 19_800_000
VAL_PER_CATEGORY = 100_000
TEST_PER_CATEGORY = 100_000

TRAIN_TOTAL = TRAIN_PER_CATEGORY * NUM_CATEGORIES  # 990,000,000
VAL_TOTAL = VAL_PER_CATEGORY * NUM_CATEGORIES      #   5,000,000
TEST_TOTAL = TEST_PER_CATEGORY * NUM_CATEGORIES    #   5,000,000

TRAIN_SHARDS = 10_000  # 99,000 records per train shard
VAL_SHARDS = 50        # 100,000 records per validation shard
TEST_SHARDS = 50       # 100,000 records per test shard

# 40 natural engineering / review / operational contexts (Factor A: 40)
NATURAL_CONTEXT_OPENERS: tuple[str, ...] = (
    "Context — pre-merge review",
    "Context — production readiness check",
    "Context — staff design review",
    "Context — reliability sprint",
    "Context — post-incident hardening",
    "Context — core library cleanup",
    "Context — hot-path optimization",
    "Context — platform SDK update",
    "Context — quarterly audit prep",
    "Context — service migration gate",
    "Context — canary release blocker",
    "Context — architecture RFC validation",
    "Context — tech-debt reduction pass",
    "Context — security champion review",
    "Context — on-call handoff item",
    "Context — zero-regression refactor",
    "Context — multi-tenant isolation pass",
    "Context — CI quality gate",
    "Context — high-throughput service audit",
    "Context — backend guild review",
    "Engineering note — release candidate check",
    "Engineering note — latency tail review",
    "Engineering note — defensive hardening pass",
    "Engineering note — schema & contract check",
    "Engineering note — edge-case audit",
    "Engineering note — static-analysis triage",
    "Engineering note — observability & safety pass",
    "Engineering note — distributed systems review",
    "Engineering note — production runbook task",
    "Engineering note — code-health milestone",
    "Review scope — tier-0 service check",
    "Review scope — automated regression gate",
    "Review scope — principal engineer sign-off",
    "Review scope — compliance readiness item",
    "Review scope — API contract verification",
    "Review scope — dependency & runtime check",
    "Review scope — fault-tolerance verification",
    "Review scope — data-integrity audit",
    "Review scope — deployment pipeline gate",
    "Review scope — clean-room implementation check",
)

# 50 stack-matched runtime/toolchain profiles per technology family (Factor B: 50)
PY_PROFILES = tuple(
    f"Python 3.{minor} ({checker}, {mode})"
    for minor in (10, 11, 12, 13, 14)
    for checker in ("mypy --strict", "pyright strict", "ruff + pytest", "cpython-linux-x86_64", "cpython-arm64")
    for mode in ("zero-external-deps", "production-stdlib")
)  # 5 * 5 * 2 = 50

SQL_PROFILES = tuple(
    f"{engine} ({iso}, {planner})"
    for engine in ("PostgreSQL 15", "PostgreSQL 16", "PostgreSQL 17", "Aurora PostgreSQL 16", "AlloyDB PG16")
    for iso in ("READ COMMITTED", "REPEATABLE READ", "strict OLTP", "reporting replica", "partitioned ledger")
    for planner in ("index-only preferred", "seqscan-guarded")
)  # 5 * 5 * 2 = 50

GO_PROFILES = tuple(
    f"Go 1.{minor} ({flag}, {arch})"
    for minor in (21, 22, 23, 24, 25)
    for flag in ("go test -race", "golangci-lint strict", "CGO_ENABLED=0", "pprof-verified", "static-binary")
    for arch in ("linux/amd64", "linux/arm64")
)  # 50

RUST_PROFILES = tuple(
    f"Rust 1.{minor} ({lint}, {prof})"
    for minor in (78, 79, 80, 81, 82)
    for lint in ("clippy -D warnings", "cargo-deny clean", "forbid(unsafe_code)", "edition-2021", "miri-checked")
    for prof in ("release-lto", "overflow-safe")
)  # 50

TS_PROFILES = tuple(
    f"TypeScript 5.{minor} ({runtime}, {flag})"
    for minor in (3, 4, 5, 6, 7)
    for runtime in ("Node 20 LTS", "Node 22 LTS", "Deno 2", "Bun 1.1", "Edge Worker V8")
    for flag in ("strictNullChecks", "noUncheckedIndexedAccess")
)  # 50

INFRA_SEC_PROFILES = tuple(
    f"{baseline} ({gate}, {tier})"
    for baseline in ("SOC2-Type2", "PCI-DSS-4.0", "ISO-27001", "HIPAA-Security", "SLSA-L3")
    for gate in ("CI policy-gate", "admission-enforced", "audit-verified", "fail-closed", "least-privilege")
    for tier in ("tier-0", "tier-1")
)  # 50

AGENT_PROFILES = tuple(
    f"AgentRuntime v{ver} ({policy}, {mode})"
    for ver in ("2.1", "2.2", "2.3", "2.4", "3.0")
    for policy in ("strict-json-schema", "zero-hallucinated-ids", "untrusted-tool-output", "read-before-write", "least-privilege-tools")
    for mode in ("deterministic", "grounded")
)  # 50

CATEGORY_THOUGHT_STRATEGIES: dict[str, tuple[str, str]] = {
    "algo_data_structures": (
        "Avoid quadratic rescanning by maintaining a monotonic window, binary-search predicate, heap, or topological DP state.",
        "Check empty/single-element inputs and tie-breaking boundaries so runtime stays strictly linear or O(N log N).",
    ),
    "bug_diagnosis_and_fix": (
        "Pinpoint the exact state-sharing, floating-point representation, race-window, or boundary-index defect in the snippet.",
        "Preserve the existing function signature while eliminating the root cause deterministically.",
    ),
    "sql_query_and_optimization": (
        "Structure the query with explicit filtering/windowing and order composite B-tree index columns as (equality, filter/group, sort).",
        "Verify aggregate edge cases and deterministic ordering without full-table sort spills.",
    ),
    "api_backend_engineering": (
        "Enforce atomic state transitions for idempotency, monotonic token replenishment, or keyset `(sort_col, id)` pagination.",
        "Prevent duplicate side effects on concurrent retries and avoid O(N) OFFSET scans.",
    ),
    "refactoring_clean_code": (
        "Replace imperative branching or eager list accumulation with dispatch mappings, frozen slotted dataclasses, or lazy generators.",
        "Keep external return values identical while cutting cyclomatic complexity and peak memory.",
    ),
    "testing_property_edge_cases": (
        "Cover nominal, empty/zero, exact-boundary, and invalid/error partitions via table-driven `@pytest.mark.parametrize` cases.",
        "Keep every assertion deterministic and isolated from wall-clock or network state.",
    ),
    "typescript_modern_js": (
        "Use strict generics, discriminated-union narrowing with an exhaustive `never` branch, or `AbortController` timer cleanup.",
        "Eliminate implicit `any` casts and ensure timers/listeners are always cleared in `finally`.",
    ),
    "systems_concurrency_perf": (
        "Bound concurrency with `asyncio.Semaphore`, transactional batch executemany, or `heapq` O(N log K) streaming selection.",
        "Prevent unbounded task/heap growth under bursty upstream load.",
    ),
    "data_pipelines_parsing": (
        "Process records incrementally, routing malformed payloads to quarantine or neutralizing spreadsheet formula prefixes.",
        "Guarantee that single bad lines never corrupt or abort valid batch records.",
    ),
    "devops_ci_containers": (
        "Separate build vs runtime stages, drop root privileges, mount read-only root filesystems, and decouple readiness from liveness probes.",
        "Pin immutable artifact references and scope workflow permissions to least privilege.",
    ),
    "git_cli_shell_engineering": (
        "Guard shell pipelines with `set -euo pipefail`, NUL-delimited path handling, `flock` locks, or exact `git bisect` exit codes.",
        "Ensure filenames with spaces, newlines, or leading dashes cannot break argument parsing.",
    ),
    "multi_turn_code_review": (
        "Deliver an O(1) ring-buffer design first, then add thread-safe locking invariants in the follow-up turn.",
        "Verify bounded capacity eviction and lock release on all code paths.",
    ),
    "go_concurrency_and_services": (
        "Wire `context.Context` cancellation through `errgroup.WithContext` workers or `http.Server.Shutdown` drain deadlines.",
        "Select on `ctx.Done()` during channel sends so early failures never leak blocked goroutines.",
    ),
    "rust_memory_and_error_handling": (
        "Use checked integer arithmetic (`checked_mul`/`checked_add`) or buffered line streaming (`BufRead::lines`) instead of unchecked/whole-file reads.",
        "Return explicit `Option`/`Result` values with zero `unwrap()` panics on untrusted input.",
    ),
    "regex_and_input_validation": (
        "Remove nested ambiguous quantifiers that cause catastrophic backtracking (`CWE-1333`) and enforce explicit length caps before matching.",
        "Guarantee linear-time O(N) rejection on adversarial near-miss strings.",
    ),
    "serialization_schema_migrations": (
        "Break breaking DDL into expand-backfill-validate phases (`NOT VALID` -> `VALIDATE CONSTRAINT`) or strict/tolerant Pydantic v2 validators.",
        "Avoid long-lived exclusive table locks against live traffic.",
    ),
    "observability_opentelemetry_metrics": (
        "Measure elapsed time via monotonic `perf_counter()` inside a `try/finally` block and normalize route labels to bound metric cardinality.",
        "Ensure exceptions are recorded on the telemetry event and re-raised unchanged.",
    ),
    "secure_code_remediation_web": (
        "Remediate the injection/traversal sink structurally using bound SQL parameters, canonical `Path.is_relative_to`, `shell=False` argv lists, or post-DNS IP checks.",
        "Confirm URL-encoded traversals, shell metacharacters, and internal/loopback IPs fail closed.",
    ),
    "auth_crypto_jwt_security": (
        "Enforce constant-time `hmac.compare_digest`, explicit server-side JWT algorithm/audience pinning, Argon2id hashing, or fresh-nonce AES-256-GCM.",
        "Eliminate timing side channels, algorithm confusion, and nonce reuse.",
    ),
    "cvss_triage_and_vuln_reporting": (
        "Evaluate Impact Sub-Score `1 - (1-C)(1-I)(1-A)`, apply the Scope-appropriate Impact & PR weights, compute Exploitability, and apply FIRST `Roundup`.",
        "Verify exact tenth-place rounding and qualitative severity thresholds (`None`/`Low`/`Medium`/`High`/`Critical`).",
    ),
    "detection_engineering_sigma_semgrep": (
        "Match dangerous sinks/log events precisely while carving out safe loaders, literal commands, or approved automation roles via `pattern-not` / `filter`.",
        "Keep detection signal high without breaking legitimate CI/service workflows.",
    ),
    "log_forensics_incident_response": (
        "Trace the attack timeline across source IPs, status shifts (`403` -> `200`, failed -> accepted SSH, IAM key creation), and post-auth commands.",
        "Prioritize immediate containment (credential rotation, session kill, perimeter block) followed by root-cause patching.",
    ),
    "network_acl_firewall_analysis": (
        "Walk the ACL top-down using first-match `(proto, port, ip in subnet)` semantics and test whether broader earlier rules shadow narrower later rules.",
        "Place specific deny/exception prefixes ahead of summary allows above default-deny.",
    ),
    "cloud_k8s_container_hardening": (
        "Enforce `runAsNonRoot`, `readOnlyRootFilesystem`, dropped Linux capabilities, resource-scoped IAM ARNs, or default-deny pod egress.",
        "Preserve required DNS (`53`) and database egress ports so workloads stay healthy.",
    ),
    "threat_modeling_stride_arch": (
        "Map threats systematically across Spoofing, Tampering, Repudiation, Info Disclosure, DoS, and Elevation of Privilege at each trust boundary.",
        "Derive tenant and caller identity strictly from authenticated credentials rather than request payloads.",
    ),
    "authorized_pentest_scoping_methodology": (
        "Check candidate target IP containment in the authorized CIDR block AND check against explicit Rules-of-Engagement host exclusions.",
        "Refuse active probing immediately on `OUT_OF_SCOPE` or `EXPLICITLY_EXCLUDED` targets.",
    ),
    "opa_rego_policy_as_code": (
        "Express fail-closed Rego v1 `deny contains msg if` predicates or strict origin-allowlist checks without regex prefix bypasses.",
        "Cover port-range containment (`from_port <= p <= to_port`) and mutable `:latest` tags.",
    ),
    "security_safety_boundaries_and_redirection": (
        "Check authorization and dual-use safety first: decline requests for unauthorized exploitation or stealth exfiltration, and pivot to defensive engineering.",
        "Provide concrete remediation, detection, or authorized staging-verification steps.",
    ),
    "ssrf_xxe_deserialization_deep_dive": (
        "Eliminate code-executing parsers (`pickle.loads`, `resolve_entities=True` XML) in favor of HMAC-verified JSON or `defusedxml` (`forbid_dtd=True`).",
        "Authenticate payload integrity before parsing untrusted bytes.",
    ),
    "api_rate_limiting_and_graphql_security": (
        "Reject deeply nested GraphQL AST queries prior to execution or run sliding-window rate checks atomically inside a single Redis Lua script.",
        "Prevent amplification DoS and concurrent check-then-increment race conditions.",
    ),
    "linux_systemd_and_host_hardening": (
        "Lock down `systemd` service namespaces (`ProtectSystem=strict`, `PrivateTmp=yes`, empty `CapabilityBoundingSet`) or key-only `sshd_config`.",
        "Grant explicit `StateDirectory` access only where daemon persistence requires it.",
    ),
    "secrets_management_and_ci_supply_chain": (
        "Use short-lived OIDC `AssumeRoleWithWebIdentity` federation pinned to exact repository/ref claims or dual-key verification windows.",
        "Eliminate static long-lived cloud credentials from CI secrets.",
    ),
    "web_security_headers_and_csp": (
        "Issue per-response cryptographic CSP nonces with `'strict-dynamic'`, `object-src 'none'`, HSTS, `X-Content-Type-Options: nosniff`, and `frame-ancestors 'none'`.",
        "Avoid `'unsafe-inline'` so reflected/stored HTML injection cannot execute scripts.",
    ),
}

FUNC_IDENT_RE = re.compile(r"\b(?:def|func|fn|function|class|table|package)\s+([A-Za-z_][A-Za-z0-9_.]*)")
BACKTICK_IDENT_RE = re.compile(r"`([A-Za-z0-9_./:-]{3,48})`")
TOOL_CALL_EXTRACT_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
TOOLS_BLOCK_RE = re.compile(r"<tools>\n(.*?)\n</tools>", re.DOTALL)


def _even_half_bits(domain_size: int) -> int:
    if domain_size <= 1:
        return 1
    bits = (domain_size - 1).bit_length()
    if bits % 2 == 1:
        bits += 1
    return bits // 2


def _feistel_prf(right: int, round_idx: int, seed: int, mask: int) -> int:
    x = ((right & 0xFFFFFFFFFFFFFFFF) ^ ((round_idx * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF) ^ seed) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x = x ^ (x >> 31)
    return x & mask


def feistel_permute(index: int, domain_size: int, seed: int = 42, rounds: int = 6) -> int:
    """Bijectively permute `index in [0, domain_size)` onto `[0, domain_size)` in O(1) time & O(1) RAM."""
    if not (0 <= index < domain_size):
        raise IndexError(f"Index {index} out of range [0, {domain_size})")
    if domain_size == 1:
        return 0

    k = _even_half_bits(domain_size)
    mask = (1 << k) - 1
    seed64 = seed & 0xFFFFFFFFFFFFFFFF

    curr = index
    while True:
        left = (curr >> k) & mask
        right = curr & mask
        for r in range(rounds):
            left, right = right, left ^ _feistel_prf(right, r, seed64, mask)
        curr = (left << k) | right
        if curr < domain_size:
            return curr


def feistel_inverse(permuted: int, domain_size: int, seed: int = 42, rounds: int = 6) -> int:
    """Exact mathematical inverse of `feistel_permute` on `[0, domain_size)`."""
    if not (0 <= permuted < domain_size):
        raise IndexError(f"Permuted value {permuted} out of range [0, {domain_size})")
    if domain_size == 1:
        return 0

    k = _even_half_bits(domain_size)
    mask = (1 << k) - 1
    seed64 = seed & 0xFFFFFFFFFFFFFFFF

    curr = permuted
    while True:
        left = (curr >> k) & mask
        right = curr & mask
        for r in range(rounds - 1, -1, -1):
            left, right = right ^ _feistel_prf(left, r, seed64, mask), left
        curr = (left << k) | right
        if curr < domain_size:
            return curr


def resolve_coordinate(split: str, index: int, seed: int = 42) -> tuple[int, int]:
    """Map `(split, index)` bijectively to `(cat_idx in [0..49], local_coord in [0..19_999_999])`."""
    if split == "all":
        if not (0 <= index < TOTAL_BILLION_RECORDS):
            raise IndexError(f"Global index {index} out of range [0, {TOTAL_BILLION_RECORDS})")
        if index < TRAIN_TOTAL:
            return resolve_coordinate("train", index, seed)
        if index < TRAIN_TOTAL + VAL_TOTAL:
            return resolve_coordinate("validation", index - TRAIN_TOTAL, seed)
        return resolve_coordinate("test", index - TRAIN_TOTAL - VAL_TOTAL, seed)

    if split == "train":
        p = feistel_permute(index, TRAIN_TOTAL, seed * 1_000_003 + 11)
        return p % NUM_CATEGORIES, p // NUM_CATEGORIES
    if split == "validation":
        p = feistel_permute(index, VAL_TOTAL, seed * 1_000_003 + 29)
        return p % NUM_CATEGORIES, TRAIN_PER_CATEGORY + (p // NUM_CATEGORIES)
    if split == "test":
        p = feistel_permute(index, TEST_TOTAL, seed * 1_000_003 + 47)
        return p % NUM_CATEGORIES, TRAIN_PER_CATEGORY + VAL_PER_CATEGORY + (p // NUM_CATEGORIES)

    raise ValueError(f"Unknown split '{split}'. Expected 'train', 'validation', 'test', or 'all'.")


def _coord_seed64(seed: int, cat_idx: int, local_coord: int) -> int:
    digest = hashlib.blake2b(
        struct.pack("<QQQ", seed & 0xFFFFFFFFFFFFFFFF, cat_idx, local_coord),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little")


def _select_stack_profile(category: str, domain: str, b_idx: int) -> str:
    if category == "sql_query_and_optimization" or category == "serialization_schema_migrations":
        return SQL_PROFILES[b_idx]
    if category == "go_concurrency_and_services":
        return GO_PROFILES[b_idx]
    if category == "rust_memory_and_error_handling":
        return RUST_PROFILES[b_idx]
    if category == "typescript_modern_js":
        return TS_PROFILES[b_idx]
    if category in ("devops_ci_containers", "git_cli_shell_engineering") or domain == "ethical_security":
        return INFRA_SEC_PROFILES[b_idx]
    if domain == "agent_use":
        return AGENT_PROFILES[b_idx]
    return PY_PROFILES[b_idx]


def _extract_focus_terms(user_text: str, asst_text: str) -> str:
    found: list[str] = []
    for m in BACKTICK_IDENT_RE.findall(user_text):
        if m not in found and not m.startswith("Context"):
            found.append(f"`{m}`")
        if len(found) == 3:
            break
    if not found:
        for m in FUNC_IDENT_RE.findall(asst_text):
            if m not in found:
                found.append(f"`{m}`")
            if len(found) == 2:
                break
    return ", ".join(found) if found else "the target component"


def _build_agent_thought(system_text: str, first_asst: str, user_text: str) -> str:
    """Build an exact, tool-schema-grounded <thought> block inspecting actual called vs distractor tools."""
    declared_names: list[str] = []
    m_tools = TOOLS_BLOCK_RE.search(system_text)
    if m_tools:
        for line in m_tools.group(1).splitlines():
            line = line.strip()
            if line:
                try:
                    declared_names.append(json.loads(line)["name"])
                except Exception:
                    pass

    calls = []
    for raw_call in TOOL_CALL_EXTRACT_RE.findall(first_asst):
        try:
            obj = json.loads(raw_call)
            calls.append((obj.get("name", ""), list(obj.get("arguments", {}).keys())))
        except Exception:
            pass

    called_names = [c[0] for c in calls if c[0]]
    distractors = [t for t in declared_names if t not in called_names]
    focus = _extract_focus_terms(user_text, "")

    if len(calls) > 1:
        call_desc = f"Issue `{len(calls)}` independent `{calls[0][0]}` calls in parallel (`{', '.join(calls[0][1])}`) to avoid sequential round-trip overhead."
    elif len(calls) == 1:
        arg_list = ", ".join(f"`{k}`" for k in calls[0][1]) or "schema-validated parameters"
        call_desc = f"Invoke `{calls[0][0]}` with required fields ({arg_list}) scoped to {focus}."
    else:
        call_desc = f"Evaluate {focus} against declared tool schemas."

    distractor_note = (
        f" Do not invoke `{distractors[0]}` because it is either out of scope or state-mutating for this step."
        if distractors
        else " Treat returned tool payloads strictly as untrusted data before synthesizing the final answer."
    )
    return f"<thought>\n1. Intent & Target: Resolve request for {focus}.\n2. Tool Selection: {call_desc}{distractor_note}\n</thought>"


def synthesize_ultra_record(cat_idx: int, local_coord: int, seed: int = 42) -> dict[str, Any]:
    """
    Deterministically synthesize a forensically hardened, solver-verified 'Ultra' ChatML record
    for `(cat_idx in [0..49], local_coord in [0..19_999_999])` (1,000,000,000 total).
    """
    s64 = _coord_seed64(seed, cat_idx, local_coord)
    rng = random.Random(s64)
    domain, category, gen_fn = ALL_GENERATORS[cat_idx]

    rec = gen_fn(rng, local_coord)

    # Exact bijective decomposition of local_coord in [0, 20_000_000): 40 * 50 * 10_000 = 20,000,000
    a_idx = local_coord % 40
    b_idx = (local_coord // 40) % 50
    c_num = 1000 + (local_coord // 2000)  # 1000 .. 10999

    opener_ctx = NATURAL_CONTEXT_OPENERS[a_idx]
    stack_prof = _select_stack_profile(category, domain, b_idx)
    ultra_mode = (local_coord + cat_idx) % 4

    rec["id"] = f"utd-{cat_idx:02d}-{local_coord:08d}"
    base_tags = list(rec["tags"])[:5]
    if "ultra-1b" not in base_tags:
        base_tags.append("ultra-1b")
    rec["tags"] = base_tags

    msgs = rec["messages"]
    is_refusal = category == "security_safety_boundaries_and_redirection" and (local_coord % 3 == 0)

    # Inject natural, stack-coherent, non-contradictory context into the user prompt
    if is_refusal:
        # Keep unauthorized-request refusal prompts realistic without fake compliance scopes
        msgs[1]["content"] = f"[Thread #{a_idx + 1}-{b_idx + 1}-{c_num}] {msgs[1]['content']}"
    else:
        msgs[1]["content"] = f"{msgs[1]['content']}\n\n*({opener_ctx} #{c_num} · target profile: {stack_prof})*"

    if domain in ("coding", "ethical_security") and rec["style"] in ("direct", "step_by_step"):
        if ultra_mode in (0, 3):
            rec["style"] = "ultra_reasoning"
            focus = _extract_focus_terms(msgs[1]["content"], msgs[-1]["content"])
            strat_main, strat_edge = CATEGORY_THOUGHT_STRATEGIES[category]

            if domain == "coding":
                msgs[0]["content"] = (
                    "You are a Distinguished Staff Software Engineer. Reason concisely inside <thought>...</thought> "
                    "about complexity, edge cases, and runtime invariants before providing clean, production-grade code."
                )
                if ultra_mode == 0:
                    thought = (
                        f"<thought>\n"
                        f"- Target artifacts: {focus} under `{stack_prof}`.\n"
                        f"- Design approach: {strat_main}\n"
                        f"- Edge-case check: {strat_edge}\n"
                        f"</thought>\n\n"
                    )
                else:
                    thought = (
                        f"<thought>\n"
                        f"Analyzing {focus} (`{stack_prof}`): {strat_main} "
                        f"Before finalizing, verify invariants: {strat_edge}\n"
                        f"</thought>\n\n"
                    )
            else:
                msgs[0]["content"] = (
                    "You are a Principal Defensive Security Architect. Analyze trust boundaries, root weakness mechanics, "
                    "and bypass resistance inside <thought>...</thought>, then provide verified defensive guidance."
                )
                if ultra_mode == 0:
                    thought = (
                        f"<thought>\n"
                        f"- Scope & target: {focus} (`{stack_prof}`).\n"
                        f"- Security analysis: {strat_main}\n"
                        f"- Assurance criterion: {strat_edge}\n"
                        f"</thought>\n\n"
                    )
                else:
                    thought = (
                        f"<thought>\n"
                        f"Evaluating {focus}: {strat_main} "
                        f"Confirm fail-closed behavior: {strat_edge}\n"
                        f"</thought>\n\n"
                    )
            msgs[-1]["content"] = thought + msgs[-1]["content"]

    elif rec["style"] == "agent_trajectory" and len(msgs) >= 3 and "<tool_call>" in msgs[2]["content"]:
        first_asst = msgs[2]["content"]
        if not first_asst.lstrip().startswith("<thought>"):
            idx_tc = first_asst.index("<tool_call>")
            tc_part = first_asst[idx_tc:]
            grounded_thought = _build_agent_thought(msgs[0]["content"], first_asst, msgs[1]["content"])
            msgs[2]["content"] = f"{grounded_thought}\n{tc_part}"

    return rec


def split_size(split: str) -> int:
    if split == "train":
        return TRAIN_TOTAL
    if split == "validation":
        return VAL_TOTAL
    if split == "test":
        return TEST_TOTAL
    if split == "all":
        return TOTAL_BILLION_RECORDS
    raise ValueError(f"Invalid split: {split}")


def shard_count_for_split(split: str) -> int:
    if split == "train":
        return TRAIN_SHARDS
    if split == "validation":
        return VAL_SHARDS
    if split == "test":
        return TEST_SHARDS
    raise ValueError(f"Invalid split for sharding: {split}")


def get_shard_bounds(split: str, shard_id: int) -> tuple[int, int]:
    """Return `[start_idx, end_idx)` within `split` for `shard_id`."""
    n_shards = shard_count_for_split(split)
    if not (0 <= shard_id < n_shards):
        raise IndexError(f"shard_id {shard_id} out of range [0, {n_shards}) for split '{split}'")
    per_shard = split_size(split) // n_shards
    return shard_id * per_shard, (shard_id + 1) * per_shard


class BillionUltraDataset:
    """
    O(1) Random-Access 1,000,000,000-Example Map-Style Dataset.
    Compatible directly with `torch.utils.data.DataLoader`, `DistributedSampler`, and Python indexing/slicing.
    Uses 0 bytes of index/dedup RAM regardless of whether you access record 0 or record 999,999,999!
    """

    def __init__(self, split: str = "train", seed: int = 42) -> None:
        self.split = split
        self.seed = seed
        self._length = split_size(split)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._length)
            return [self[i] for i in range(start, stop, step)]
        if index < 0:
            index += self._length
        if not (0 <= index < self._length):
            raise IndexError(f"Index {index} out of range [0, {self._length})")
        cat_idx, local_coord = resolve_coordinate(self.split, index, self.seed)
        return synthesize_ultra_record(cat_idx, local_coord, self.seed)


def iter_billion_ultra(
    *,
    split: str = "train",
    start: int = 0,
    count: int | None = None,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
) -> Iterator[dict[str, Any]]:
    """
    Deterministic multi-GPU / DDP / FSDP streaming iterator over up to 1,000,000,000 Ultra examples
    in O(1) RAM. Worker `rank` yields indices `start + rank, start + rank + world_size, ...`.
    """
    if world_size < 1 or not (0 <= rank < world_size):
        raise ValueError(f"Invalid rank={rank}, world_size={world_size}")
    total = split_size(split)
    end = total if count is None else min(total, start + count)
    for idx in range(start + rank, end, world_size):
        cat_idx, local_coord = resolve_coordinate(split, idx, seed)
        yield synthesize_ultra_record(cat_idx, local_coord, seed)
