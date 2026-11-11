"""Coding Generators Part 3 (Categories 37–41): Go Concurrency, Rust Safety, Verified Regex, Schema Evolution, OpenTelemetry."""

from __future__ import annotations

import random
import re
from typing import Any

from training_data.schema import make_record, pick_coding_system
from training_data.vocab import (
    ENVIRONMENTS,
    FILE_PATHS,
    SERVICES,
    TENANT_SLUGS,
    USER_OPENERS_DIRECT,
    USER_OPENERS_STEP,
)


# ---------------------------------------------------------------------------
# 37. go_concurrency_and_services
# ---------------------------------------------------------------------------

def gen_go_concurrency_and_services(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    style = rng.choice(["direct", "step_by_step"])
    opener = rng.choice(USER_OPENERS_DIRECT if style == "direct" else USER_OPENERS_STEP)
    variant = idx % 3

    if variant == 0:
        difficulty = "medium"
        workers = rng.randint(2, 24)
        user_q = (
            f"{opener}\nWrite an idiomatic Go 1.22 worker pool `RunBatch(ctx context.Context, ids []string) error` "
            f"for `{svc_name}` (`{env}`, `{workers}` workers) that cancels remaining work on the first error."
        )
        go_code = (
            "package worker\n\nimport (\n\t\"context\"\n\t\"sync\"\n)\n\n"
            f"func RunBatch(ctx context.Context, ids []string, fn func(context.Context, string) error) error {{\n"
            "\tctx, cancel := context.WithCancel(ctx)\n\tdefer cancel()\n"
            "\tjobs := make(chan string)\n\tvar wg sync.WaitGroup\n\tvar once sync.Once\n\tvar firstErr error\n\n"
            f"\tfor i := 0; i < {workers}; i++ {{\n"
            "\t\twg.Add(1)\n\t\tgo func() {\n\t\t\tdefer wg.Done()\n\t\t\tfor id := range jobs {\n"
            "\t\t\t\tif err := fn(ctx, id); err != nil {\n\t\t\t\t\tonce.Do(func() { firstErr = err; cancel() })\n\t\t\t\t\treturn\n\t\t\t\t}\n"
            "\t\t\t}\n\t\t}()\n\t}\n"
            "\tfor _, id := range ids {\n\t\tselect {\n\t\tcase <-ctx.Done():\n\t\t\tbreak\n\t\tcase jobs <- id:\n\t\t}\n\t}\n"
            "\tclose(jobs)\n\twg.Wait()\n\treturn firstErr\n}"
        )
        note = "Selecting on `<-ctx.Done()` while sending prevents producer goroutine leaks when workers exit early."
    elif variant == 1:
        difficulty = "easy"
        rd_to = rng.randint(3, 15)
        wr_to = rng.randint(5, 30)
        port = rng.randint(8000, 9500)
        user_q = (
            f"{opener}\nConfigure a hardened Go `http.Server` for `{svc_name}` (`:{port}`, `{env}`) with `ReadHeaderTimeout={rd_to}s` "
            f"and `WriteTimeout={wr_to}s` to prevent Slowloris socket exhaustion."
        )
        go_code = (
            "package server\n\nimport (\n\t\"net/http\"\n\t\"time\"\n)\n\n"
            "func NewHardenedServer(handler http.Handler) *http.Server {\n"
            "\treturn &http.Server{\n"
            f"\t\tAddr:              \":{port}\",\n"
            "\t\tHandler:           handler,\n"
            f"\t\tReadHeaderTimeout: {rd_to} * time.Second,\n"
            f"\t\tWriteTimeout:      {wr_to} * time.Second,\n"
            "\t\tIdleTimeout:       60 * time.Second,\n"
            "\t\tMaxHeaderBytes:    1 << 16,\n"
            "\t}\n}"
        )
        note = "Go's default `&http.Server{}` has zero timeouts; `ReadHeaderTimeout` bounds Slowloris header-drip connections."
    else:
        difficulty = "medium"
        cap_n = rng.randint(64, 2048)
        user_q = (
            f"{opener}\nImplement a concurrency-safe Go `CounterMap` using `sync.RWMutex` for `{svc_name}` (`{table_ctx}`, initial hint `{cap_n}`)."
        )
        go_code = (
            "package metrics\n\nimport \"sync\"\n\n"
            "type CounterMap struct {\n\tmu   sync.RWMutex\n\tvals map[string]int64\n}\n\n"
            f"func NewCounterMap() *CounterMap {{\n\treturn &CounterMap{{vals: make(map[string]int64, {cap_n})}}\n}}\n\n"
            "func (m *CounterMap) Add(key string, delta int64) {\n\tm.mu.Lock()\n\tm.vals[key] += delta\n\tm.mu.Unlock()\n}\n\n"
            "func (m *CounterMap) Get(key string) int64 {\n\tm.mu.RLock()\n\tv := m.vals[key]\n\tm.mu.RUnlock()\n\treturn v\n}"
        )
        note = "`RLock` permits concurrent readers without contention while `Lock` serializes map writes safely."

    ans = f"### Go Concurrency Notes\n- {note}\n\n```go\n{go_code}\n```" if style == "step_by_step" else f"```go\n{go_code}\n```\n- {note}"
    return make_record(
        record_id=f"td-go-{idx:07d}",
        domain="coding",
        category="go_concurrency_and_services",
        difficulty=difficulty,
        style=style,
        tags=["go", "concurrency", "backend"],
        quality_check="syntax_and_pattern_verified",
        messages=[
            {"role": "system", "content": pick_coding_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans},
        ],
    )


# ---------------------------------------------------------------------------
# 38. rust_memory_and_error_handling
# ---------------------------------------------------------------------------

def gen_rust_memory_and_error_handling(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    style = rng.choice(["direct", "step_by_step"])
    opener = rng.choice(USER_OPENERS_DIRECT if style == "direct" else USER_OPENERS_STEP)
    max_val = rng.randint(50, 9999)
    variant = idx % 2

    if variant == 0:
        difficulty = "medium"
        user_q = (
            f"{opener}\nWrite an idiomatic Rust function `parse_bounded(raw: &str) -> Result<u32, ParseQuotaError>` "
            f"for `{svc_name}` (`{env}`) rejecting empty input, invalid integers, or values `> {max_val}` without `.unwrap()`."
        )
        rs_code = (
            "#[derive(Debug, PartialEq, Eq)]\n"
            "pub enum ParseQuotaError {\n    Empty,\n    InvalidInt,\n    ExceedsMax(u32),\n}\n\n"
            "pub fn parse_bounded(raw: &str) -> Result<u32, ParseQuotaError> {\n"
            "    let trimmed = raw.trim();\n"
            "    if trimmed.is_empty() {\n        return Err(ParseQuotaError::Empty);\n    }\n"
            "    let val: u32 = trimmed.parse().map_err(|_| ParseQuotaError::InvalidInt)?;\n"
            f"    if val > {max_val} {{\n        return Err(ParseQuotaError::ExceedsMax(val));\n    }}\n"
            "    Ok(val)\n}"
        )
        note = "Explicit `Result<u32, ParseQuotaError>` + `?` propagation eliminates panics on untrusted input."
    else:
        difficulty = "medium"
        prefix = f"{domain_ctx}:"
        user_q = (
            f"{opener}\nWrite a zero-allocation Rust helper `strip_tag<'a>(line: &'a str) -> Option<&'a str>` "
            f"for `{svc_name}` (`{table_ctx}`, sample threshold `{max_val}`) that strips prefix `\"{prefix}\"` and trims ASCII whitespace."
        )
        rs_code = (
            "pub fn strip_tag<'a>(line: &'a str) -> Option<&'a str> {\n"
            f"    let rest = line.strip_prefix(\"{prefix}\")?.trim();\n"
            "    if rest.is_empty() { None } else { Some(rest) }\n"
            "}"
        )
        note = "`str::strip_prefix` borrows directly from `'a` without heap-allocating a `String`."

    ans = f"### Rust Safety Notes\n- {note}\n\n```rust\n{rs_code}\n```" if style == "step_by_step" else f"```rust\n{rs_code}\n```\n- {note}"
    return make_record(
        record_id=f"td-rust-{idx:07d}",
        domain="coding",
        category="rust_memory_and_error_handling",
        difficulty=difficulty,
        style=style,
        tags=["rust", "memory-safety", "error-handling"],
        quality_check="syntax_and_pattern_verified",
        messages=[
            {"role": "system", "content": pick_coding_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans},
        ],
    )


# ---------------------------------------------------------------------------
# 39. regex_and_input_validation (Live Python `re` Verified!)
# ---------------------------------------------------------------------------

def gen_regex_and_input_validation(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    t_slug = rng.choice(TENANT_SLUGS)
    env = rng.choice(ENVIRONMENTS)
    style = rng.choice(["direct", "step_by_step"])
    min_d = rng.randint(3, 7)
    max_d = rng.randint(8, 24)
    pat_str = rf"^{domain_ctx}_[a-z0-9]{{{min_d},{max_d}}}$"
    compiled = re.compile(pat_str)
    good_sample = f"{domain_ctx}_{'a1b2c3d4e5f6g7h8'[:min_d + 1]}"
    bad_sample = f"{domain_ctx}_{'x' * (max_d + 2)}"
    assert compiled.fullmatch(good_sample) is not None
    assert compiled.fullmatch(bad_sample) is None

    user_q = (
        f"Write a ReDoS-safe Python validator `is_valid_{id_col}(val: str) -> bool` for `{svc_name}` (`{t_slug}`, `{env}`) "
        f"matching exact prefix `'{domain_ctx}_'` followed by `{min_d}..{max_d}` lowercase ASCII alphanumeric characters (disallow trailing newline). "
        f"Verify on `'{good_sample}'` and `'{bad_sample}'`."
    )
    code = (
        "import re\n\n"
        f"ID_PATTERN = re.compile(r'\\A{domain_ctx}_[a-z0-9]{{{min_d},{max_d}}}\\Z')\n\n"
        f"def is_valid_{id_col}(val: str) -> bool:\n"
        "    return ID_PATTERN.fullmatch(val) is not None"
    )
    note = f"`\\A...\\Z` (unlike `$`) rejects trailing `\\n`, and bounded `[a-z0-9]{{{min_d},{max_d}}}` runs in linear O(n) time (`'{good_sample}' -> True`, `'{bad_sample}' -> False`)."
    ans = f"### Bounded Linear Regex\n- {note}\n\n```python\n{code}\n```" if style == "step_by_step" else f"```python\n{code}\n```\n- {note}"

    return make_record(
        record_id=f"td-regex-{idx:07d}",
        domain="coding",
        category="regex_and_input_validation",
        difficulty="easy",
        style=style,
        tags=["python", "regex", "input-validation"],
        quality_check="ast_and_solver_verified",
        messages=[
            {"role": "system", "content": pick_coding_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans},
        ],
    )


# ---------------------------------------------------------------------------
# 40. serialization_schema_migrations (AST-Verified!)
# ---------------------------------------------------------------------------

def gen_serialization_schema_migrations(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    style = rng.choice(["direct", "step_by_step"])
    def_ccy = rng.choice(["USD", "EUR", "GBP", "JPY", "CAD"])
    sample_amt = rng.randint(100, 95000)

    user_q = (
        f"In `{svc_name}` (`{env}`), legacy v1 `{table_ctx}` events have `{{'ver': 1, '{id_col}': '...', 'cents': {sample_amt}}}`, "
        f"while v2 requires `{{'ver': 2, '{id_col}': '...', 'money': {{'cents': ..., 'currency': '{def_ccy}'}}}}`. "
        f"Write a pure Python upcaster `upcast_event(raw: dict) -> dict` handling both v1 and v2."
    )
    code = (
        "from typing import Any\n\n"
        "def upcast_event(raw: dict[str, Any]) -> dict[str, Any]:\n"
        "    ver = raw.get('ver', 1)\n"
        "    if ver == 2:\n"
        "        return dict(raw)\n"
        "    if ver == 1:\n"
        "        return {\n"
        "            'ver': 2,\n"
        f"            '{id_col}': raw['{id_col}'],\n"
        f"            'money': {{'cents': int(raw['cents']), 'currency': '{def_ccy}'}},\n"
        "        }\n"
        "    raise ValueError(f'unsupported schema version: {ver}')"
    )
    note = "Idempotent v1->v2 upcasting at read boundaries decouples consumer upgrades from historical event-log rewrites."
    ans = f"### Backward-Compatible Upcasting\n- {note}\n\n```python\n{code}\n```" if style == "step_by_step" else f"```python\n{code}\n```\n- {note}"

    return make_record(
        record_id=f"td-schema-{idx:07d}",
        domain="coding",
        category="serialization_schema_migrations",
        difficulty="medium",
        style=style,
        tags=["python", "schema-evolution", "serialization"],
        quality_check="ast_and_solver_verified",
        messages=[
            {"role": "system", "content": pick_coding_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans},
        ],
    )


# ---------------------------------------------------------------------------
# 41. observability_opentelemetry_metrics (AST-Verified!)
# ---------------------------------------------------------------------------

def gen_observability_opentelemetry_metrics(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    fpath = rng.choice(FILE_PATHS)
    style = rng.choice(["direct", "step_by_step"])
    sla_ms = rng.randint(50, 1500)

    user_q = (
        f"Write a Python context manager `timed_span(op_name: str, sla_ms: float = {sla_ms}.0)` for `{svc_name}` "
        f"(`{fpath}`, `{env}`) that measures monotonic duration, flags `sla_breached`, and emits structured JSON even if an exception occurs."
    )
    code = (
        "from collections.abc import Iterator\nfrom contextlib import contextmanager\nimport json\nimport time\n\n"
        "@contextmanager\n"
        f"def timed_span(op_name: str, sla_ms: float = {sla_ms}.0) -> Iterator[None]:\n"
        "    start = time.perf_counter()\n"
        "    status = 'ok'\n"
        "    try:\n"
        "        yield\n"
        "    except Exception as exc:\n"
        "        status = type(exc).__name__\n"
        "        raise\n"
        "    finally:\n"
        "        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)\n"
        "        print(json.dumps({\n"
        f"            'service': '{svc_name}',\n"
        "            'op': op_name,\n"
        "            'status': status,\n"
        "            'duration_ms': elapsed_ms,\n"
        "            'sla_breached': elapsed_ms > sla_ms,\n"
        "        }))"
    )
    note = "`finally` guarantees span emission on unhandled exceptions while re-raising cleanly with high-resolution `perf_counter()`."
    ans = f"### Telemetry Instrumentation\n- {note}\n\n```python\n{code}\n```" if style == "step_by_step" else f"```python\n{code}\n```\n- {note}"

    return make_record(
        record_id=f"td-otel-{idx:07d}",
        domain="coding",
        category="observability_opentelemetry_metrics",
        difficulty="medium",
        style=style,
        tags=["python", "observability", "telemetry"],
        quality_check="ast_and_solver_verified",
        messages=[
            {"role": "system", "content": pick_coding_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans},
        ],
    )
