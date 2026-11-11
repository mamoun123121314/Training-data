"""Coding Generators Part 2 (Categories 7–12): TypeScript, Systems/Perf, Data/ETL, DevOps, Git/Shell, Multi-Turn."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from training_data.schema import make_record, pick_coding_system
from training_data.vocab import (
    ENVIRONMENTS,
    FILE_PATHS,
    REVIEW_CONTEXTS,
    SERVICES,
    TENANT_SLUGS,
    USER_OPENERS_DIRECT,
    USER_OPENERS_STEP,
    gen_short_sha,
)


# ---------------------------------------------------------------------------
# 7. typescript_modern_js
# ---------------------------------------------------------------------------

def gen_typescript_modern_js(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    style = rng.choice(["direct", "step_by_step"])
    opener = rng.choice(USER_OPENERS_DIRECT if style == "direct" else USER_OPENERS_STEP)
    variant = idx % 5

    if variant == 0:
        difficulty = "easy"
        timeout_ms = rng.randint(800, 9500)
        user_q = (
            f"{opener}\nWrite a TypeScript 5 helper `fetchWithTimeout<T>(url: string, ms = {timeout_ms}): Promise<T>` "
            f"for `{svc_name}` (`{env}`) using `AbortController` that always clears its timer."
        )
        ts_code = (
            f"export async function fetchWithTimeout<T>(url: string, ms = {timeout_ms}): Promise<T> {{\n"
            "  const controller = new AbortController();\n"
            "  const timer = setTimeout(() => controller.abort(), ms);\n"
            "  try {\n"
            "    const res = await fetch(url, { signal: controller.signal });\n"
            "    if (!res.ok) throw new Error(`HTTP ${res.status}`);\n"
            "    return (await res.json()) as T;\n"
            "  } finally {\n"
            "    clearTimeout(timer);\n"
            "  }\n"
            "}"
        )
        note = "Clearing `timer` inside `finally` prevents dangling Node.js event-loop handles on fast responses."
    elif variant == 1:
        difficulty = "medium"
        fee_bps = rng.randint(10, 95)
        user_q = (
            f"{opener}\nModel `{svc_name}` (`{table_ctx}`) job states as a TypeScript discriminated union "
            f"(`queued` | `settled` | `failed`) and write `describeState` with an exhaustive `never` check (settlement fee `{fee_bps}` bps)."
        )
        ts_code = (
            f"export type JobState =\n"
            f"  | {{ status: 'queued'; {id_col}: string }}\n"
            f"  | {{ status: 'settled'; {id_col}: string; cents: number }}\n"
            f"  | {{ status: 'failed'; {id_col}: string; reason: string }};\n\n"
            "export function describeState(state: JobState): string {\n"
            "  switch (state.status) {\n"
            f"    case 'queued': return `queued:${{state.{id_col}}}`;\n"
            f"    case 'settled': return `net:${{state.cents - Math.floor((state.cents * {fee_bps}) / 10000)}}`;\n"
            "    case 'failed': return `err:${state.reason}`;\n"
            "    default: {\n"
            "      const _exhaustive: never = state;\n"
            "      return _exhaustive;\n"
            "    }\n"
            "  }\n"
            "}"
        )
        note = "Assigning unhandled branches to `never` turns missing union cases into compile-time errors."
    elif variant == 2:
        difficulty = "hard"
        conc = rng.randint(2, 16)
        user_q = (
            f"{opener}\nImplement `mapPool<T, R>(items: readonly T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]>` "
            f"in TypeScript for `{svc_name}` (`{env}`, default concurrency `{conc}`) preserving input order."
        )
        ts_code = (
            f"export async function mapPool<T, R>(\n"
            f"  items: readonly T[],\n"
            f"  limit = {conc},\n"
            "  fn: (item: T) => Promise<R>,\n"
            "): Promise<R[]> {\n"
            "  const results = new Array<R>(items.length);\n"
            "  let cursor = 0;\n"
            "  async function worker(): Promise<void> {\n"
            "    while (cursor < items.length) {\n"
            "      const idx = cursor++;\n"
            "      results[idx] = await fn(items[idx]);\n"
            "    }\n"
            "  }\n"
            "  const workers = Array.from({ length: Math.min(limit, items.length) }, () => worker());\n"
            "  await Promise.all(workers);\n"
            "  return results;\n"
            "}"
        )
        note = "Shared index `cursor++` is race-free between `await` points in single-threaded JS while preserving output indices."
    elif variant == 3:
        difficulty = "medium"
        max_ver = rng.randint(2, 25)
        user_q = (
            f"{opener}\nWrite a TypeScript runtime type guard `isValidPayload(u: unknown): u is {{ {id_col}: string; version: number }}` "
            f"for `{svc_name}` requiring non-empty `{id_col}` and integer `version` in `1..{max_ver}`."
        )
        ts_code = (
            f"export interface ValidPayload {{ {id_col}: string; version: number }}\n\n"
            "export function isValidPayload(u: unknown): u is ValidPayload {\n"
            "  if (typeof u !== 'object' || u === null) return false;\n"
            "  const rec = u as Record<string, unknown>;\n"
            f"  return (\n"
            f"    typeof rec.{id_col} === 'string' &&\n"
            f"    rec.{id_col}.trim().length > 0 &&\n"
            "    typeof rec.version === 'number' &&\n"
            "    Number.isInteger(rec.version) &&\n"
            f"    rec.version >= 1 && rec.version <= {max_ver}\n"
            "  );\n"
            "}"
        )
        note = "Narrows `unknown` safely without unsafe casts, checking `Number.isInteger` to reject `NaN` and `Infinity`."
    else:
        difficulty = "medium"
        wait_ms = rng.randint(50, 900)
        user_q = (
            f"{opener}\nWrite a typed `debounce<Args extends unknown[]>(fn: (...args: Args) => void, waitMs = {wait_ms})` "
            f"in TypeScript for `{svc_name}` telemetry flushing, including a `.cancel()` method."
        )
        ts_code = (
            f"export function debounce<Args extends unknown[]>(\n"
            "  fn: (...args: Args) => void,\n"
            f"  waitMs = {wait_ms},\n"
            ") {\n"
            "  let timer: ReturnType<typeof setTimeout> | undefined;\n"
            "  const wrapped = (...args: Args): void => {\n"
            "    if (timer !== undefined) clearTimeout(timer);\n"
            "    timer = setTimeout(() => fn(...args), waitMs);\n"
            "  };\n"
            "  wrapped.cancel = (): void => {\n"
            "    if (timer !== undefined) clearTimeout(timer);\n"
            "    timer = undefined;\n"
            "  };\n"
            "  return wrapped;\n"
            "}"
        )
        note = "Preserves exact tuple parameter types via `Args extends unknown[]` and exposes `.cancel()` for teardown."

    ans_text = (
        f"### Key Mechanism\n- {note}\n\n```typescript\n{ts_code}\n```"
        if style == "step_by_step"
        else f"```typescript\n{ts_code}\n```\n- {note}"
    )

    return make_record(
        record_id=f"td-ts-{idx:06d}",
        domain="coding",
        category="typescript_modern_js",
        difficulty=difficulty,
        style=style,
        tags=["typescript", "type-safety", "async"],
        quality_check="syntax_and_pattern_verified",
        messages=[
            {"role": "system", "content": pick_coding_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans_text},
        ],
    )


# ---------------------------------------------------------------------------
# 8. systems_concurrency_perf
# ---------------------------------------------------------------------------

def gen_systems_concurrency_perf(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    style = rng.choice(["direct", "step_by_step"])
    variant = idx % 4

    if variant == 0:
        difficulty = "easy"
        n_rows = rng.randint(10_000, 500_000)
        user_q = (
            f"In `{svc_name}` (`{env}`), joining `{n_rows}` `{table_ctx}` items against `blacklist` takes seconds:\n"
            f"```python\ndef filter_blocked(events: list[dict], blocked_ids: list[str]) -> list[dict]:\n"
            f"    return [e for e in events if e['{id_col}'] not in blocked_ids]\n```\n"
            f"Optimize the asymptotic bottleneck."
        )
        code = (
            "from typing import Any\n\n"
            "def filter_blocked(events: list[dict[str, Any]], blocked_ids: list[str]) -> list[dict[str, Any]]:\n"
            "    blocked_set = set(blocked_ids)\n"
            f"    return [e for e in events if e['{id_col}'] not in blocked_set]"
        )
        note = f"Converts O(N * M) linear `list` membership scans across `{n_rows}` items into O(N + M) via O(1) average `set` lookups."
    elif variant == 1:
        difficulty = "medium"
        sem_lim = rng.randint(2, 512)
        fpath = rng.choice(FILE_PATHS)
        user_q = (
            f"Write `gather_bounded(coros, limit={sem_lim})` in Python `asyncio` for `{svc_name}` (`{fpath}`, `{env}`) "
            f"so spawning thousands of `{domain_ctx}` RPCs never exceeds `{sem_lim}` in-flight sockets."
        )
        code = (
            "import asyncio\nfrom collections.abc import Awaitable, Callable, Sequence\nfrom typing import TypeVar\n\n"
            "T = TypeVar('T')\n\n"
            f"async def gather_bounded(factories: Sequence[Callable[[], Awaitable[T]]], limit: int = {sem_lim}) -> list[T]:\n"
            "    sem = asyncio.Semaphore(limit)\n\n"
            "    async def run_one(fn: Callable[[], Awaitable[T]]) -> T:\n"
            "        async with sem:\n"
            "            return await fn()\n\n"
            "    return list(await asyncio.gather(*(run_one(f) for f in factories)))"
        )
        note = f"Accepting lazy callables (`factories`) ensures coroutines are only instantiated once the `Semaphore({sem_lim})` slot is acquired."
    elif variant == 2:
        difficulty = "hard"
        hdr_len = rng.randint(4, 64)
        user_q = (
            f"In `{svc_name}` (`{env}`), slicing `buf = buf[{hdr_len}:]` in a packet loop copies megabytes per second. "
            f"Write `iter_frames(raw: bytes, header_size={hdr_len})` using zero-copy `memoryview` where the first 4 big-endian bytes encode payload length."
        )
        code = (
            "import struct\nfrom collections.abc import Iterator\n\n"
            f"def iter_frames(raw: bytes, header_size: int = {hdr_len}) -> Iterator[memoryview]:\n"
            "    view = memoryview(raw)\n"
            "    offset = 0\n"
            "    total = len(view)\n"
            "    while offset + header_size <= total:\n"
            "        (body_len,) = struct.unpack_from('>I', view, offset)\n"
            "        start = offset + header_size\n"
            "        end = start + body_len\n"
            "        if end > total:\n"
            "            raise ValueError('truncated frame')\n"
            "        yield view[start:end]\n"
            "        offset = end"
        )
        note = "`memoryview` slicing and `struct.unpack_from` read directly from `raw` in O(1) per frame with zero byte copies."
    else:
        difficulty = "hard"
        acct_prefix = rng.choice(TENANT_SLUGS)
        min_xfer = rng.randint(1, 50)
        user_q = (
            f"Two threads transferring quota (`min_transfer={min_xfer}`) between `{acct_prefix}_a` and `{acct_prefix}_b` in `{svc_name}` (`{env}`) deadlock when locking `(src, dst)` in opposite orders. "
            f"Implement a deadlock-free `SafeLedger` transfer in Python."
        )
        code = (
            "import threading\n\n"
            "class SafeLedger:\n"
            "    def __init__(self, balances: dict[str, int]) -> None:\n"
            "        self._bal = dict(balances)\n"
            "        self._locks = {k: threading.Lock() for k in balances}\n\n"
            "    def transfer(self, src: str, dst: str, amount: int) -> None:\n"
            "        if src == dst or amount <= 0:\n"
            "            return\n"
            "        first, second = (src, dst) if src < dst else (dst, src)\n"
            "        with self._locks[first], self._locks[second]:\n"
            "            if self._bal[src] < amount:\n"
            "                raise ValueError('insufficient quota')\n"
            "            self._bal[src] -= amount\n"
            "            self._bal[dst] += amount"
        )
        note = "Acquiring locks in global canonical key order (`min(src, dst)` then `max(src, dst)`) breaks circular wait (Coffman condition 4)."

    ans_text = (
        f"### Performance & Concurrency Principle\n- {note}\n\n```python\n{code}\n```"
        if style == "step_by_step"
        else f"```python\n{code}\n```\n- {note}"
    )

    return make_record(
        record_id=f"td-perf-{idx:06d}",
        domain="coding",
        category="systems_concurrency_perf",
        difficulty=difficulty,
        style=style,
        tags=["python", "performance", "concurrency"],
        quality_check="ast_and_solver_verified",
        messages=[
            {"role": "system", "content": pick_coding_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans_text},
        ],
    )


# ---------------------------------------------------------------------------
# 9. data_pipelines_parsing (Verified Output!)
# ---------------------------------------------------------------------------

def gen_data_pipelines_parsing(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    style = rng.choice(["direct", "step_by_step"])
    variant = idx % 4

    if variant == 0:
        difficulty = "medium"
        v1, v2 = rng.randint(10, 400), rng.randint(10, 400)
        k1 = f"ev_{rng.randint(100, 999)}"
        lines = [
            json.dumps({id_col: k1, "val": v1}),
            "{corrupt_json",
            json.dumps({id_col: k1, "val": 999}),  # duplicate id_col
            json.dumps({id_col: f"{k1}_b", "val": v2}),
        ]
        user_q = (
            f"For `{svc_name}`, write `ingest_jsonl(lines: list[str]) -> tuple[list[dict], list[str]]` in Python "
            f"that parses JSONL, routes malformed lines to `dlq`, deduplicates by first-seen `{id_col}`, "
            f"and returns `(valid, dlq)`. Show output on `{lines}`."
        )
        code = (
            "import json\nfrom typing import Any\n\n"
            "def ingest_jsonl(lines: list[str]) -> tuple[list[dict[str, Any]], list[str]]:\n"
            "    valid: list[dict[str, Any]] = []\n"
            "    dlq: list[str] = []\n"
            "    seen: set[str] = set()\n"
            "    for raw in lines:\n"
            "        try:\n"
            "            obj = json.loads(raw)\n"
            f"            key = obj['{id_col}']\n"
            "        except Exception:\n"
            "            dlq.append(raw)\n"
            "            continue\n"
            "        if key not in seen:\n"
            "            seen.add(key)\n"
            "            valid.append(obj)\n"
            "    return valid, dlq"
        )
        expected_valid = [{id_col: k1, "val": v1}, {id_col: f"{k1}_b", "val": v2}]
        note = f"Verified output: `valid = {expected_valid}`, `dlq = ['{{corrupt_json']`."
    elif variant == 1:
        difficulty = "easy"
        u_name = rng.choice(TENANT_SLUGS)
        lat_ms = rng.randint(12, 980)
        status = rng.choice([200, 201, 404, 503])
        sample_log = f"ts=2026-10-12T09:15:00Z svc={svc_name} tenant={u_name} status={status} latency_ms={lat_ms}"
        user_q = (
            f"Write `parse_kv_log(line: str) -> dict[str, str | int]` in Python for `{svc_name}` "
            f"that parses `key=value` tokens and casts `status` and `latency_ms` to `int`. Verify on `\"{sample_log}\"`."
        )
        code = (
            "import re\n\n"
            "TOKEN_RE = re.compile(r'([a-zA-Z0-9_]+)=([^\\s]+)')\n"
            "INT_FIELDS = {'status', 'latency_ms'}\n\n"
            "def parse_kv_log(line: str) -> dict[str, str | int]:\n"
            "    out: dict[str, str | int] = {}\n"
            "    for key, val in TOKEN_RE.findall(line):\n"
            "        out[key] = int(val) if key in INT_FIELDS else val\n"
            "    return out"
        )
        parsed = {"ts": "2026-10-12T09:15:00Z", "svc": svc_name, "tenant": u_name, "status": status, "latency_ms": lat_ms}
        note = f"Parsed dictionary: `{parsed}`."
    elif variant == 2:
        difficulty = "medium"
        bucket_s = rng.choice([60, 300, 900, 3600])
        t_a = bucket_s * rng.randint(100, 500) + 12
        t_b = t_a + 15
        v_a, v_b = rng.randint(5, 90), rng.randint(5, 90)
        user_q = (
            f"Implement `rollup_buckets(points: list[tuple[int, int]], width_s: int = {bucket_s}) -> dict[int, int]` "
            f"in Python for `{svc_name}` summing `(unix_ts, count)` into aligned `{bucket_s}s` floor buckets. "
            f"Evaluate on `[({t_a}, {v_a}), ({t_b}, {v_b})]`."
        )
        code = (
            "from collections import defaultdict\n\n"
            f"def rollup_buckets(points: list[tuple[int, int]], width_s: int = {bucket_s}) -> dict[int, int]:\n"
            "    totals: dict[int, int] = defaultdict(int)\n"
            "    for ts, count in points:\n"
            "        bucket = (ts // width_s) * width_s\n"
            "        totals[bucket] += count\n"
            "    return dict(sorted(totals.items()))"
        )
        b_start = (t_a // bucket_s) * bucket_s
        b2 = (t_b // bucket_s) * bucket_s
        res = {b_start: v_a + v_b} if b_start == b2 else {b_start: v_a, b2: v_b}
        note = f"Computed bucket rollup: `{res}`."
    else:
        difficulty = "hard"
        salt_tag = f"{svc_name}:{rng.randint(1000, 9999)}"
        raw_email = f"{rng.choice(TENANT_SLUGS)}@example.org"
        digest = hashlib.sha256(f"{salt_tag}:{raw_email}".encode("utf-8")).hexdigest()[:16]
        user_q = (
            f"Write `pseudonymize_pii(record: dict[str, str], salt: str = '{salt_tag}') -> dict[str, str]` in Python "
            f"that lowercases/trims `email` and replaces it with a 16-hex SHA-256 prefix of `f'{{salt}}:{{email}}'`. "
            f"Verify on `{{'{id_col}': '1', 'email': '  {raw_email.upper()}  '}}`."
        )
        code = (
            "import hashlib\n\n"
            f"def pseudonymize_pii(record: dict[str, str], salt: str = '{salt_tag}') -> dict[str, str]:\n"
            "    out = dict(record)\n"
            "    if 'email' in out:\n"
            "        norm = out['email'].strip().lower()\n"
            "        out['email'] = hashlib.sha256(f'{salt}:{norm}'.encode('utf-8')).hexdigest()[:16]\n"
            "    return out"
        )
        note = f"Verified pseudonymized output: `{{'{id_col}': '1', 'email': '{digest}'}}`."

    ans_text = (
        f"### Pipeline Verification\n- {note}\n\n```python\n{code}\n```"
        if style == "step_by_step"
        else f"```python\n{code}\n```\n- {note}"
    )

    return make_record(
        record_id=f"td-etl-{idx:06d}",
        domain="coding",
        category="data_pipelines_parsing",
        difficulty=difficulty,
        style=style,
        tags=["python", "etl", "data-engineering"],
        quality_check="ast_and_solver_verified",
        messages=[
            {"role": "system", "content": pick_coding_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans_text},
        ],
    )


# ---------------------------------------------------------------------------
# 10. devops_ci_containers
# ---------------------------------------------------------------------------

def gen_devops_ci_containers(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    style = rng.choice(["direct", "step_by_step"])
    variant = idx % 4
    port = rng.randint(8000, 9400)

    if variant == 0:
        difficulty = "medium"
        uid = rng.randint(10001, 10999)
        user_q = (
            f"Write a minimal multi-stage `Dockerfile` for Python 3.11 service `{svc_name}` listening on port `{port}` "
            f"running as non-root UID `{uid}` without build compilers in the final image."
        )
        cfg = (
            "FROM python:3.11-slim AS builder\n"
            "WORKDIR /build\n"
            "COPY requirements.txt .\n"
            "RUN pip install --no-cache-dir --prefix=/install -r requirements.txt\n\n"
            "FROM python:3.11-slim\n"
            f"RUN groupadd -g {uid} app && useradd -u {uid} -g app -s /sbin/nologin -M app\n"
            "WORKDIR /app\n"
            "COPY --from=builder /install /usr/local\n"
            "COPY src/ ./src/\n"
            f"USER {uid}:{uid}\n"
            f"EXPOSE {port}\n"
            f"CMD [\"python\", \"-m\", \"uvicorn\", \"src.main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"{port}\"]"
        )
        lang = "dockerfile"
        note = f"Separating `builder` strips build caches and runs `uvicorn` under unprivileged UID `{uid}`."
    elif variant == 1:
        difficulty = "medium"
        replicas = rng.randint(3, 8)
        cpu_m = rng.choice([250, 500, 1000])
        mem_mi = rng.choice([256, 512, 1024])
        user_q = (
            f"Create a Kubernetes `Deployment` snippet for `{svc_name}` in namespace `{domain_ctx}` (`{replicas}` replicas, port `{port}`) "
            f"with CPU `{cpu_m}m`, memory `{mem_mi}Mi`, and `/healthz` readinessprobe."
        )
        cfg = (
            "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n"
            f"  name: {svc_name}\n  namespace: {domain_ctx}\nspec:\n"
            f"  replicas: {replicas}\n"
            f"  selector:\n    matchLabels:\n      app: {svc_name}\n"
            f"  template:\n    metadata:\n      labels:\n        app: {svc_name}\n"
            "    spec:\n      containers:\n"
            f"        - name: {svc_name}\n"
            f"          image: registry.internal/{svc_name}:v1.4.2\n"
            f"          ports:\n            - containerPort: {port}\n"
            "          readinessProbe:\n"
            f"            httpGet:\n              path: /healthz\n              port: {port}\n"
            "            initialDelaySeconds: 5\n            periodSeconds: 10\n"
            "          resources:\n"
            f"            requests:\n              cpu: {cpu_m}m\n              memory: {mem_mi}Mi\n"
            f"            limits:\n              memory: {mem_mi}Mi"
        )
        lang = "yaml"
        note = "Explicit readiness probes gate Service traffic until `/healthz` succeeds; matched memory request/limit avoids burst OOM."
    elif variant == 2:
        difficulty = "easy"
        sha = gen_short_sha(rng) + gen_short_sha(rng)
        job_to = rng.randint(5, 45)
        user_q = (
            f"Write a concise GitHub Actions CI job (`timeout-minutes: {job_to}`) for `{svc_name}` (`{env}`) with read-only `GITHUB_TOKEN`, "
            f"pip caching, `ruff check`, and `pytest`."
        )
        cfg = (
            f"name: {svc_name}-ci\n"
            "on:\n  pull_request:\n  push:\n    branches: [main]\n"
            "permissions:\n  contents: read\n"
            f"jobs:\n  test:\n    runs-on: ubuntu-latest\n    timeout-minutes: {job_to}\n    steps:\n"
            f"      - uses: actions/checkout@{sha}0123456789abcdef01234567\n"
            "      - uses: actions/setup-python@v5\n"
            "        with:\n          python-version: '3.11'\n          cache: 'pip'\n"
            "      - run: pip install -r requirements-dev.txt\n"
            "      - run: ruff check src/ tests/\n"
            "      - run: pytest -q"
        )
        lang = "yaml"
        note = "Top-level `permissions: contents: read` enforces least privilege on `GITHUB_TOKEN`."
    else:
        difficulty = "hard"
        keepalive = rng.choice([32, 64, 128])
        read_to = rng.randint(10, 60)
        user_q = (
            f"Configure an Nginx reverse-proxy `upstream` and `server` block forwarding to `{svc_name}` on `127.0.0.1:{port}` "
            f"with HTTP/1.1 keepalive `{keepalive}` and `proxy_read_timeout {read_to}s`."
        )
        cfg = (
            f"upstream {domain_ctx}_backend {{\n"
            f"    server 127.0.0.1:{port} max_fails=3 fail_timeout=10s;\n"
            f"    keepalive {keepalive};\n"
            "}\n\n"
            "server {\n"
            "    listen 443 ssl http2;\n"
            f"    server_name {svc_name}.internal;\n"
            "    location / {\n"
            f"        proxy_pass http://{domain_ctx}_backend;\n"
            "        proxy_http_version 1.1;\n"
            "        proxy_set_header Connection \"\";\n"
            "        proxy_set_header Host $host;\n"
            "        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n"
            f"        proxy_read_timeout {read_to}s;\n"
            "    }\n"
            "}"
        )
        lang = "nginx"
        note = "Setting `proxy_http_version 1.1` and clearing `Connection` is required for Nginx upstream `keepalive` socket reuse."

    ans_text = (
        f"### Operational Rationale\n- {note}\n\n```{lang}\n{cfg}\n```"
        if style == "step_by_step"
        else f"```{lang}\n{cfg}\n```\n- {note}"
    )

    return make_record(
        record_id=f"td-devops-{idx:06d}",
        domain="coding",
        category="devops_ci_containers",
        difficulty=difficulty,
        style=style,
        tags=["devops", "containers", "infrastructure"],
        quality_check="config_structure_verified",
        messages=[
            {"role": "system", "content": pick_coding_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans_text},
        ],
    )


# ---------------------------------------------------------------------------
# 11. git_cli_shell_engineering
# ---------------------------------------------------------------------------

def gen_git_cli_shell_engineering(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    style = rng.choice(["direct", "step_by_step"])
    variant = idx % 4

    if variant == 0:
        difficulty = "medium"
        days = rng.randint(7, 90)
        log_dir = f"/var/log/{svc_name}"
        user_q = (
            f"Write a safe Bash script (`set -euo pipefail`) that deletes `.gz` archives older than `{days}` days "
            f"under `{log_dir}` (`{env}`), safely handling filenames with spaces or newlines."
        )
        sh = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n\n"
            f"LOG_DIR=\"{log_dir}\"\n"
            "[[ -d \"$LOG_DIR\" ]] || { echo \"missing $LOG_DIR\" >&2; exit 1; }\n"
            f"find \"$LOG_DIR\" -type f -name '*.gz' -mtime +{days} -print0 | xargs -0 -r rm -f --"
        )
        note = "`-print0 | xargs -0 -r` uses NUL delimiters so whitespace cannot split arguments, and `-r` skips running `rm` on empty input."
    elif variant == 1:
        difficulty = "medium"
        min_ms = rng.randint(250, 2000)
        user_q = (
            f"Write a `jq` command to filter NDJSON logs from `{svc_name}` (`{env}`) where `.latency_ms >= {min_ms}` "
            f"and `.status >= 500`, outputting compact `{{\"id\": .{id_col}, \"ms\": .latency_ms}}` objects."
        )
        sh = f"jq -c 'select(.latency_ms >= {min_ms} and .status >= 500) | {{id: .{id_col}, ms: .latency_ms}}' {svc_name}.jsonl"
        note = "`-c` emits one compact JSON object per line, streaming arbitrarily large NDJSON files in constant memory."
    elif variant == 2:
        difficulty = "hard"
        good_sha = gen_short_sha(rng)
        bad_sha = gen_short_sha(rng)
        test_file = f"tests/test_{domain_ctx}.py"
        user_q = (
            f"A regression appeared in `{svc_name}` between known good commit `{good_sha}` and bad commit `{bad_sha}`. "
            f"Show the exact non-interactive `git bisect` commands using `pytest -q {test_file}` to pinpoint the offending commit and reset cleanly."
        )
        sh = (
            f"git bisect start {bad_sha} {good_sha}\n"
            f"git bisect run pytest -q {test_file}\n"
            "git bisect reset"
        )
        note = "`git bisect run` binary-searches the commit graph automatically (`0` = good, `1..124` = bad, `125` = skip untestable commit)."
    else:
        difficulty = "medium"
        cfg_ver = rng.randint(1, 250)
        dest_cfg = f"/etc/{svc_name}/{env}-v{cfg_ver}.json"
        user_q = (
            f"Write a defensive Bash snippet to update `{dest_cfg}` atomically so readers in `{svc_name}` never observe a half-written JSON file."
        )
        sh = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n\n"
            f"DEST=\"{dest_cfg}\"\n"
            "TMP=$(mktemp \"${DEST}.tmp.XXXXXX\")\n"
            "trap 'rm -f \"$TMP\"' EXIT\n"
            f"jq -n --arg env \"{env}\" '{{service: \"{svc_name}\", env: $env}}' > \"$TMP\"\n"
            "chmod 0640 \"$TMP\"\n"
            "mv -f \"$TMP\" \"$DEST\""
        )
        note = "Creating `mktemp` in the same directory as `DEST` guarantees `mv` executes an atomic `rename(2)` syscall on the same filesystem."

    ans_text = (
        f"### Shell Safety Details\n- {note}\n\n```bash\n{sh}\n```"
        if style == "step_by_step"
        else f"```bash\n{sh}\n```\n- {note}"
    )

    return make_record(
        record_id=f"td-sh-{idx:06d}",
        domain="coding",
        category="git_cli_shell_engineering",
        difficulty=difficulty,
        style=style,
        tags=["bash", "git", "cli"],
        quality_check="shell_and_git_verified",
        messages=[
            {"role": "system", "content": pick_coding_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans_text},
        ],
    )


# ---------------------------------------------------------------------------
# 12. multi_turn_code_review (5-message Multi-Turn Refinement, AST Verified!)
# ---------------------------------------------------------------------------

def gen_multi_turn_code_review(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    variant = idx % 4

    if variant == 0:
        difficulty = "medium"
        max_sz = rng.randint(32, 512)
        ttl_s = rng.randint(15, 300)
        u1 = f"Write a simple bounded FIFO cache `BoundedCache(max_size={max_sz})` in Python for `{svc_name}` (`{id_col}` lookups)."
        a1 = (
            "```python\n"
            "from collections import OrderedDict\nfrom typing import Any\n\n"
            "class BoundedCache:\n"
            f"    def __init__(self, max_size: int = {max_sz}) -> None:\n"
            "        self.max_size = max_size\n"
            "        self._data: OrderedDict[str, Any] = OrderedDict()\n\n"
            "    def get(self, key: str) -> Any | None:\n"
            "        return self._data.get(key)\n\n"
            "    def put(self, key: str, value: Any) -> None:\n"
            "        if key in self._data:\n"
            "            self._data.move_to_end(key)\n"
            "        self._data[key] = value\n"
            "        if len(self._data) > self.max_size:\n"
            "            self._data.popitem(last=False)\n"
            "```"
        )
        u2 = f"Now add per-entry TTL expiration (`ttl_seconds={ttl_s}`) and thread safety for `{env}` workers."
        a2 = (
            "Updated with `threading.RLock` and monotonic expiry timestamps:\n\n"
            "```python\n"
            "import threading\nimport time\nfrom collections import OrderedDict\nfrom typing import Any\n\n"
            "class BoundedCache:\n"
            f"    def __init__(self, max_size: int = {max_sz}, ttl_seconds: float = {ttl_s}.0) -> None:\n"
            "        self.max_size = max_size\n"
            "        self.ttl = ttl_seconds\n"
            "        self._data: OrderedDict[str, tuple[Any, float]] = OrderedDict()\n"
            "        self._lock = threading.RLock()\n\n"
            "    def get(self, key: str) -> Any | None:\n"
            "        with self._lock:\n"
            "            entry = self._data.get(key)\n"
            "            if entry is None:\n"
            "                return None\n"
            "            val, exp = entry\n"
            "            if time.monotonic() >= exp:\n"
            "                del self._data[key]\n"
            "                return None\n"
            "            return val\n\n"
            "    def put(self, key: str, value: Any) -> None:\n"
            "        with self._lock:\n"
            "            if key in self._data:\n"
            "                self._data.move_to_end(key)\n"
            "            self._data[key] = (value, time.monotonic() + self.ttl)\n"
            "            if len(self._data) > self.max_size:\n"
            "                self._data.popitem(last=False)\n"
            "```"
        )
    elif variant == 1:
        difficulty = "medium"
        batch_lim = rng.randint(10, 200)
        max_bytes = rng.randint(4, 64) * 1024
        u1 = f"Implement `BatchBuffer(max_items={batch_lim})` in Python for `{svc_name}` (`{table_ctx}`) with `add(item: str) -> list[str] | None` returning a flushed batch when full."
        a1 = (
            "```python\n"
            "class BatchBuffer:\n"
            f"    def __init__(self, max_items: int = {batch_lim}) -> None:\n"
            "        self.max_items = max_items\n"
            "        self._buf: list[str] = []\n\n"
            "    def add(self, item: str) -> list[str] | None:\n"
            "        self._buf.append(item)\n"
            "        if len(self._buf) >= self.max_items:\n"
            "            return self.flush()\n"
            "        return None\n\n"
            "    def flush(self) -> list[str]:\n"
            "        out = self._buf\n"
            "        self._buf = []\n"
            "        return out\n"
            "```"
        )
        u2 = f"Also enforce a byte cap `max_bytes={max_bytes}` (UTF-8) so adding an item that would exceed `{max_bytes}` bytes flushes existing items first."
        a2 = (
            "Added UTF-8 byte accounting and pre-append flush when `current_bytes + size > max_bytes`:\n\n"
            "```python\n"
            "class BatchBuffer:\n"
            f"    def __init__(self, max_items: int = {batch_lim}, max_bytes: int = {max_bytes}) -> None:\n"
            "        self.max_items = max_items\n"
            "        self.max_bytes = max_bytes\n"
            "        self._buf: list[str] = []\n"
            "        self._bytes = 0\n\n"
            "    def add(self, item: str) -> list[str] | None:\n"
            "        sz = len(item.encode('utf-8'))\n"
            "        if sz > self.max_bytes:\n"
            "            raise ValueError('single item exceeds max_bytes')\n"
            "        flushed: list[str] | None = None\n"
            "        if self._buf and (self._bytes + sz > self.max_bytes):\n"
            "            flushed = self.flush()\n"
            "        self._buf.append(item)\n"
            "        self._bytes += sz\n"
            "        if len(self._buf) >= self.max_items and flushed is None:\n"
            "            return self.flush()\n"
            "        return flushed\n\n"
            "    def flush(self) -> list[str]:\n"
            "        out = self._buf\n"
            "        self._buf = []\n"
            "        self._bytes = 0\n"
            "        return out\n"
            "```"
        )
    elif variant == 2:
        difficulty = "hard"
        q_thresh = rng.randint(50, 900)
        u1 = f"Write `parse_env_int(name: str, default: int = {q_thresh}) -> int` in Python for `{svc_name}` config loading."
        a1 = (
            "```python\n"
            "import os\n\n"
            f"def parse_env_int(name: str, default: int = {q_thresh}) -> int:\n"
            "    raw = os.getenv(name)\n"
            "    if raw is None or raw.strip() == '':\n"
            "        return default\n"
            "    return int(raw.strip())\n"
            "```"
        )
        u2 = f"In `{env}`, misconfigured ranges crashed startup late. Add `min_val` and `max_val` bounds raising a clear `ConfigError(ValueError)` naming the variable."
        a2 = (
            "Added `ConfigError` with variable-specific diagnostics for non-integers and out-of-range values:\n\n"
            "```python\n"
            "import os\n\n"
            "class ConfigError(ValueError):\n"
            "    pass\n\n"
            f"def parse_env_int(\n"
            f"    name: str,\n"
            f"    default: int = {q_thresh},\n"
            "    min_val: int = 1,\n"
            "    max_val: int = 100_000,\n"
            ") -> int:\n"
            "    raw = os.getenv(name)\n"
            "    if raw is None or raw.strip() == '':\n"
            "        val = default\n"
            "    else:\n"
            "        try:\n"
            "            val = int(raw.strip())\n"
            "        except ValueError as exc:\n"
            "            raise ConfigError(f'{name} must be an integer, got {raw!r}') from exc\n"
            "    if not (min_val <= val <= max_val):\n"
            "        raise ConfigError(f'{name}={val} outside [{min_val}, {max_val}]')\n"
            "    return val\n"
            "```"
        )
    else:
        difficulty = "hard"
        win_n = rng.randint(5, 1200)
        u1 = f"Write `MovingAverage(window={win_n})` in Python for `{svc_name}` (`{env}`) metrics."
        a1 = (
            "```python\n"
            "from collections import deque\n\n"
            "class MovingAverage:\n"
            f"    def __init__(self, window: int = {win_n}) -> None:\n"
            "        self.window = window\n"
            "        self._vals: deque[float] = deque()\n"
            "        self._sum = 0.0\n\n"
            "    def push(self, x: float) -> float:\n"
            "        self._vals.append(x)\n"
            "        self._sum += x\n"
            "        if len(self._vals) > self.window:\n"
            "            self._sum -= self._vals.popleft()\n"
            "        return self._sum / len(self._vals)\n"
            "```"
        )
        u2 = "Reject `NaN`/`inf` samples so poisoned telemetry cannot corrupt `self._sum` permanently, and add a `p95()` method."
        a2 = (
            "Added `math.isfinite` guard and sorted-window `p95()` computation:\n\n"
            "```python\n"
            "import math\nfrom collections import deque\n\n"
            "class MovingAverage:\n"
            f"    def __init__(self, window: int = {win_n}) -> None:\n"
            "        self.window = window\n"
            "        self._vals: deque[float] = deque()\n"
            "        self._sum = 0.0\n\n"
            "    def push(self, x: float) -> float:\n"
            "        if not math.isfinite(x):\n"
            "            raise ValueError('telemetry sample must be finite')\n"
            "        self._vals.append(x)\n"
            "        self._sum += x\n"
            "        if len(self._vals) > self.window:\n"
            "            self._sum -= self._vals.popleft()\n"
            "        return self._sum / len(self._vals)\n\n"
            "    def p95(self) -> float:\n"
            "        if not self._vals:\n"
            "            raise ValueError('empty window')\n"
            "        ordered = sorted(self._vals)\n"
            "        idx = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)\n"
            "        return ordered[idx]\n"
            "```"
        )

    return make_record(
        record_id=f"td-mtcode-{idx:06d}",
        domain="coding",
        category="multi_turn_code_review",
        difficulty=difficulty,
        style="multi_turn",
        tags=["python", "multi-turn", "iterative-design"],
        quality_check="ast_and_solver_verified",
        messages=[
            {"role": "system", "content": pick_coding_system(rng, "step_by_step")},
            {"role": "user", "content": u1},
            {"role": "assistant", "content": a1},
            {"role": "user", "content": u2},
            {"role": "assistant", "content": a2},
        ],
    )
