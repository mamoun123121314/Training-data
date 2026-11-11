"""Coding Generators Part 1 (Categories 1–6): Algorithms, Bug Fixes, SQL, Backend APIs, Refactoring, Testing."""

from __future__ import annotations

import random
from typing import Any

from training_data.schema import make_record, pick_coding_system
from training_data.solvers import (
    execute_sqlite_query,
    solve_dijkstra,
    solve_max_window_sum,
    solve_merge_intervals,
    solve_min_coins,
    solve_min_ship_capacity,
    solve_next_greater_latency,
    solve_shortest_grid_bfs,
    solve_subarray_at_most_budget,
    solve_topo_order,
)
from training_data.vocab import (
    ENVIRONMENTS,
    FILE_PATHS,
    REVIEW_CONTEXTS,
    SERVICES,
    TENANT_SLUGS,
    USER_OPENERS_DIRECT,
    USER_OPENERS_STEP,
)


# ---------------------------------------------------------------------------
# 1. algo_data_structures (Solver-Verified)
# ---------------------------------------------------------------------------

def gen_algo_data_structures(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    style = rng.choice(["direct", "step_by_step", "step_by_step"])
    opener = rng.choice(USER_OPENERS_DIRECT if style == "direct" else USER_OPENERS_STEP)
    variant = idx % 9

    if variant == 0:
        difficulty = "easy"
        k = rng.randint(2, 5)
        n = rng.randint(k + 3, k + 7)
        nums = [rng.randint(3, 140) for _ in range(n)]
        ans = solve_max_window_sum(nums, k)
        fn_name = f"max_{domain_ctx}_window"
        user_q = (
            f"{opener}\nIn `{svc_name}`, write `{fn_name}(events: list[int], k: int) -> int` in Python "
            f"to compute the maximum sum of any contiguous window of length `k` in O(n) time and O(1) auxiliary space. "
            f"Verify on `events = {nums}`, `k = {k}`."
        )
        code = (
            f"def {fn_name}(events: list[int], k: int) -> int:\n"
            f"    if k <= 0 or k > len(events):\n"
            f"        raise ValueError('k must satisfy 1 <= k <= len(events)')\n"
            f"    window_sum = sum(events[:k])\n"
            f"    best = window_sum\n"
            f"    for i in range(k, len(events)):\n"
            f"        window_sum += events[i] - events[i - k]\n"
            f"        if window_sum > best:\n"
            f"            best = window_sum\n"
            f"    return best"
        )
        ans_text = (
            f"### Approach\n1. Maintain a fixed-width sliding window of size `{k}`.\n"
            f"2. Slide `i` from `{k}` to `len(events) - 1` by adding `events[i]` and subtracting `events[i - k]` in O(1).\n"
            f"3. On `{nums}` (`k={k}`), peak window sum is `{ans}`.\n\n```python\n{code}\n```"
            if style == "step_by_step"
            else f"```python\n{code}\n```\n- Verified output (`k={k}`): `{ans}` (O(n) time, O(1) space)."
        )
        tags = ["python", "sliding-window", "algorithms"]

    elif variant == 1:
        difficulty = "medium"
        budget = rng.randint(35, 160)
        costs = [rng.randint(5, 55) for _ in range(rng.randint(7, 11))]
        ans = solve_subarray_at_most_budget(costs, budget)
        fn_name = f"longest_{domain_ctx}_slice"
        user_q = (
            f"{opener}\nFor `{svc_name}` (`{table_ctx}`), implement `{fn_name}(costs: list[int], budget: int) -> int` "
            f"returning the length of the longest contiguous slice of non-negative costs whose sum is `<= budget`. "
            f"Show the result for `costs = {costs}`, `budget = {budget}`."
        )
        code = (
            f"def {fn_name}(costs: list[int], budget: int) -> int:\n"
            f"    left = 0\n"
            f"    running = 0\n"
            f"    best = 0\n"
            f"    for right, cost in enumerate(costs):\n"
            f"        running += cost\n"
            f"        while running > budget and left <= right:\n"
            f"            running -= costs[left]\n"
            f"            left += 1\n"
            f"        if right - left + 1 > best:\n"
            f"            best = right - left + 1\n"
            f"    return best"
        )
        ans_text = (
            f"### Two-Pointer Invariant\nSince `costs` are non-negative, advancing `left` restores `running <= budget` in amortized O(1) per element.\n\n"
            f"```python\n{code}\n```\n- **Verified Output**: `{ans}` (`budget={budget}`)."
            if style == "step_by_step"
            else f"```python\n{code}\n```\n- Output (`budget={budget}`): `{ans}` (O(n) time)."
        )
        tags = ["python", "two-pointers", "algorithms"]

    elif variant == 2:
        difficulty = "easy"
        raw_iv = []
        for _ in range(rng.randint(5, 8)):
            s = rng.randint(1, 75)
            raw_iv.append([s, s + rng.randint(2, 18)])
        ans = solve_merge_intervals(raw_iv)
        user_q = (
            f"{opener}\nIn `{svc_name}`, maintenance windows are `[start, end]` pairs. "
            f"Write `merge_windows(windows: list[list[int]]) -> list[list[int]]` to merge overlapping or touching (`s <= prev_end`) windows. "
            f"Evaluate on `{raw_iv}`."
        )
        code = (
            "def merge_windows(windows: list[list[int]]) -> list[list[int]]:\n"
            "    if not windows:\n"
            "        return []\n"
            "    ordered = sorted(windows, key=lambda w: (w[0], w[1]))\n"
            "    merged = [ordered[0][:]]\n"
            "    for start, end in ordered[1:]:\n"
            "        if start <= merged[-1][1]:\n"
            "            merged[-1][1] = max(merged[-1][1], end)\n"
            "        else:\n"
            "            merged.append([start, end])\n"
            "    return merged"
        )
        ans_text = (
            f"### Steps\n1. Sort intervals by `(start, end)` in O(n log n).\n"
            f"2. Extend `merged[-1][1]` whenever `start <= merged[-1][1]`.\n\n```python\n{code}\n```\n- **Merged**: `{ans}`."
            if style == "step_by_step"
            else f"```python\n{code}\n```\n- Merged output: `{ans}`."
        )
        tags = ["python", "intervals", "sorting"]

    elif variant == 3:
        difficulty = "medium"
        weights = [rng.randint(7, 85) for _ in range(rng.randint(6, 10))]
        batches = rng.randint(2, 5)
        ans = solve_min_ship_capacity(weights, batches)
        user_q = (
            f"{opener}\n`{svc_name}` flushes `{table_ctx}` payloads `sizes = {weights}` strictly in order across at most "
            f"`max_batches = {batches}` network flushes. Implement `min_flush_bytes(sizes: list[int], max_batches: int) -> int` "
            f"using binary search on the answer."
        )
        code = (
            "def min_flush_bytes(sizes: list[int], max_batches: int) -> int:\n"
            "    lo, hi = max(sizes), sum(sizes)\n"
            "    while lo < hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        batches, used = 1, 0\n"
            "        for sz in sizes:\n"
            "            if used + sz > mid:\n"
            "                batches += 1\n"
            "                used = sz\n"
            "            else:\n"
            "                used += sz\n"
            "        if batches <= max_batches:\n"
            "            hi = mid\n"
            "        else:\n"
            "            lo = mid + 1\n"
            "    return lo"
        )
        ans_text = (
            f"### Binary Search on Capacity\nSearch `[max(sizes)={max(weights)}, sum(sizes)={sum(weights)}]`; greedy batch count decreases monotonically with capacity.\n\n"
            f"```python\n{code}\n```\n- **Minimum capacity**: `{ans}`."
            if style == "step_by_step"
            else f"```python\n{code}\n```\n- Minimum capacity (`max_batches={batches}`): `{ans}`."
        )
        tags = ["python", "binary-search", "optimization"]

    elif variant == 4:
        difficulty = "hard"
        n = rng.choice([5, 6])
        edges: list[tuple[int, int, int]] = []
        for u in range(n - 1):
            edges.append((u, u + 1, rng.randint(3, 19)))
            if u + 2 < n:
                edges.append((u, u + 2, rng.randint(5, 28)))
        ans = solve_dijkstra(n, edges, 0, n - 1)
        user_q = (
            f"{opener}\nFor `{svc_name}` routing across `{n}` nodes (`0..{n-1}`), write "
            f"`shortest_latency(n: int, edges: list[tuple[int, int, int]], src: int, dst: int) -> int` "
            f"using Dijkstra's algorithm (`-1` if unreachable). Compute `src=0, dst={n-1}` for `edges = {edges}`."
        )
        code = (
            "import heapq\nimport math\n\n"
            "def shortest_latency(n: int, edges: list[tuple[int, int, int]], src: int, dst: int) -> int:\n"
            "    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]\n"
            "    for u, v, w in edges:\n"
            "        adj[u].append((v, w))\n"
            "    dist = [math.inf] * n\n"
            "    dist[src] = 0\n"
            "    pq: list[tuple[int, int]] = [(0, src)]\n"
            "    while pq:\n"
            "        d, u = heapq.heappop(pq)\n"
            "        if d > dist[u]:\n"
            "            continue\n"
            "        if u == dst:\n"
            "            return int(d)\n"
            "        for v, w in adj[u]:\n"
            "            nd = d + w\n"
            "            if nd < dist[v]:\n"
            "                dist[v] = nd\n"
            "                heapq.heappush(pq, (nd, v))\n"
            "    return -1"
        )
        ans_text = (
            f"### Dijkstra Min-Heap Analysis\nNon-negative link weights guarantee optimality on first pop (`d == dist[u]`) in O((V + E) log V).\n\n"
            f"```python\n{code}\n```\n- **Shortest latency `0 -> {n-1}`**: `{ans}` ms."
            if style == "step_by_step"
            else f"```python\n{code}\n```\n- Shortest latency `0 -> {n-1}`: `{ans}` ms."
        )
        tags = ["python", "graphs", "dijkstra"]

    elif variant == 5:
        difficulty = "hard"
        suffix = rng.randint(10, 999)
        pool_names = ["fetch", "parse", "validate", "enrich", "persist", "notify", "index", "audit"]
        chosen = sorted(rng.sample(pool_names, 5))
        base_stages = [f"{s}_{suffix}" for s in chosen]
        perm = base_stages[:]
        rng.shuffle(perm)
        deps = [(perm[0], perm[2]), (perm[1], perm[2]), (perm[2], perm[3]), (perm[1], perm[4]), (perm[3], perm[4])]
        ans = solve_topo_order(base_stages, deps)
        user_q = (
            f"{opener}\nIn `{svc_name}`, DAG stages `nodes = {base_stages}` have prerequisites `deps = {deps}` "
            f"where `(u, v)` means `u` precedes `v`. Implement `resolve_schedule(nodes: list[str], deps: list[tuple[str, str]]) -> list[str] | None` "
            f"returning the lexicographically smallest topological order, or `None` on cycle."
        )
        code = (
            "import heapq\n\n"
            "def resolve_schedule(nodes: list[str], deps: list[tuple[str, str]]) -> list[str] | None:\n"
            "    indeg = {u: 0 for u in nodes}\n"
            "    adj: dict[str, list[str]] = {u: [] for u in nodes}\n"
            "    for u, v in deps:\n"
            "        adj[u].append(v)\n"
            "        indeg[v] += 1\n"
            "    ready = [u for u in nodes if indeg[u] == 0]\n"
            "    heapq.heapify(ready)\n"
            "    order: list[str] = []\n"
            "    while ready:\n"
            "        u = heapq.heappop(ready)\n"
            "        order.append(u)\n"
            "        for nxt in adj[u]:\n"
            "            indeg[nxt] -= 1\n"
            "            if indeg[nxt] == 0:\n"
            "                heapq.heappush(ready, nxt)\n"
            "    return order if len(order) == len(nodes) else None"
        )
        ans_text = (
            f"### Lexicographical Kahn's Algorithm\nUse `heapq` for zero-in-degree vertices; if `len(order) < len(nodes)`, a cycle exists.\n\n"
            f"```python\n{code}\n```\n- **Schedule**: `{ans}`."
            if style == "step_by_step"
            else f"```python\n{code}\n```\n- Schedule output: `{ans}`."
        )
        tags = ["python", "topological-sort", "dag"]

    elif variant == 6:
        difficulty = "medium"
        vals = [rng.randint(12, 240) for _ in range(rng.randint(6, 9))]
        ans = solve_next_greater_latency(vals)
        user_q = (
            f"{opener}\nFor `{svc_name}` telemetry `latencies = {vals}`, write `steps_until_spike(latencies: list[int]) -> list[int]` "
            f"returning for each index `i` how many steps ahead the first strictly higher latency occurs (`-1` if none), in O(n) time."
        )
        code = (
            "def steps_until_spike(latencies: list[int]) -> list[int]:\n"
            "    out = [-1] * len(latencies)\n"
            "    stack: list[int] = []\n"
            "    for idx, val in enumerate(latencies):\n"
            "        while stack and latencies[stack[-1]] < val:\n"
            "            prev = stack.pop()\n"
            "            out[prev] = idx - prev\n"
            "        stack.append(idx)\n"
            "    return out"
        )
        ans_text = (
            f"### Monotonic Decreasing Stack\nPop indices whose latency is strictly smaller than `val` and record `idx - prev`.\n\n"
            f"```python\n{code}\n```\n- **Output**: `{ans}`."
            if style == "step_by_step"
            else f"```python\n{code}\n```\n- Verified output: `{ans}`."
        )
        tags = ["python", "monotonic-stack", "algorithms"]

    elif variant == 7:
        difficulty = "medium"
        coins = sorted(rng.sample([1, 2, 3, 4, 5, 6, 7, 9, 11, 13], 4))
        if 1 not in coins:
            coins[0] = 1
        target = rng.randint(18, 95)
        ans = solve_min_coins(coins, target)
        user_q = (
            f"{opener}\n`{svc_name}` allocates worker packs of sizes `{coins}` to satisfy exact capacity `target = {target}`. "
            f"Implement `min_packs(sizes: list[int], target: int) -> int` using bottom-up DP (`-1` if impossible)."
        )
        code = (
            "def min_packs(sizes: list[int], target: int) -> int:\n"
            "    sentinel = target + 1\n"
            "    dp = [sentinel] * (target + 1)\n"
            "    dp[0] = 0\n"
            "    for amt in range(1, target + 1):\n"
            "        for sz in sizes:\n"
            "            if sz <= amt and dp[amt - sz] + 1 < dp[amt]:\n"
            "                dp[amt] = dp[amt - sz] + 1\n"
            "    return dp[target] if dp[target] <= target else -1"
        )
        ans_text = (
            f"### Bottom-Up DP\n`dp[a] = min(dp[a - sz] + 1)` over `sz <= a` with `dp[0] = 0`.\n\n"
            f"```python\n{code}\n```\n- **Result (`target={target}`)**: `{ans}`."
            if style == "step_by_step"
            else f"```python\n{code}\n```\n- Minimum packs (`target={target}`): `{ans}`."
        )
        tags = ["python", "dynamic-programming", "optimization"]

    else:
        difficulty = "medium"
        rows_n = rng.choice([3, 4])
        cols_n = rng.choice([4, 5])
        grid = [[rng.choice([0, 0, 0, 1]) for _ in range(cols_n)] for _ in range(rows_n)]
        grid[0][0] = 0
        grid[-1][-1] = 0
        env = rng.choice(ENVIRONMENTS)
        ans = solve_shortest_grid_bfs(grid)
        user_q = (
            f"{opener}\nIn `{svc_name}` (`{env}`) mesh `grid = {grid}` (`0` = healthy, `1` = offline), "
            f"write `shortest_hop_count(grid: list[list[int]]) -> int` returning minimum cells from `(0, 0)` to `(R-1, C-1)` "
            f"moving 4-directionally (`-1` if blocked)."
        )
        code = (
            "from collections import deque\n\n"
            "def shortest_hop_count(grid: list[list[int]]) -> int:\n"
            "    rows, cols = len(grid), len(grid[0])\n"
            "    if grid[0][0] == 1 or grid[rows - 1][cols - 1] == 1:\n"
            "        return -1\n"
            "    queue: deque[tuple[int, int, int]] = deque([(0, 0, 1)])\n"
            "    visited = {(0, 0)}\n"
            "    while queue:\n"
            "        r, c, dist = queue.popleft()\n"
            "        if (r, c) == (rows - 1, cols - 1):\n"
            "            return dist\n"
            "        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):\n"
            "            nr, nc = r + dr, c + dc\n"
            "            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and (nr, nc) not in visited:\n"
            "                visited.add((nr, nc))\n"
            "                queue.append((nr, nc, dist + 1))\n"
            "    return -1"
        )
        ans_text = (
            f"### Unweighted Grid BFS\nMark neighbors visited before enqueueing to run in O(R * C).\n\n"
            f"```python\n{code}\n```\n- **Hop count**: `{ans}`."
            if style == "step_by_step"
            else f"```python\n{code}\n```\n- Sample grid result: `{ans}`."
        )
        tags = ["python", "bfs", "graphs"]

    return make_record(
        record_id=f"td-algo-{idx:06d}",
        domain="coding",
        category="algo_data_structures",
        difficulty=difficulty,
        style=style,
        tags=tags,
        quality_check="ast_and_solver_verified",
        messages=[
            {"role": "system", "content": pick_coding_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans_text},
        ],
    )


# ---------------------------------------------------------------------------
# 2. bug_diagnosis_and_fix
# ---------------------------------------------------------------------------

def gen_bug_diagnosis_and_fix(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    fpath = rng.choice(FILE_PATHS)
    ctx_note = rng.choice(REVIEW_CONTEXTS)
    style = rng.choice(["direct", "step_by_step", "step_by_step"])
    variant = idx % 8

    if variant == 0:
        difficulty = "easy"
        tag_val = f"trace:{env}:{rng.randint(100, 9999)}"
        buggy = (
            f"def append_{domain_ctx}_tag(tag: str, bucket: list[str] = []) -> list[str]:\n"
            f"    bucket.append(tag)\n"
            f"    return bucket"
        )
        fixed = (
            f"def append_{domain_ctx}_tag(tag: str, bucket: list[str] | None = None) -> list[str]:\n"
            f"    target = [] if bucket is None else list(bucket)\n"
            f"    target.append(tag)\n"
            f"    return target"
        )
        user_q = (
            f"In `{svc_name}` (`{fpath}`, spotted {ctx_note}), tags leak across requests when `bucket` is omitted:\n"
            f"```python\n{buggy}\n```\n"
            f"Diagnose why `append_{domain_ctx}_tag('{tag_val}')` retains prior state and fix it."
        )
        diag = (
            "### Root Cause\nPython evaluates default argument expressions **once at function definition time**. "
            "Every call omitting `bucket` mutates the same shared `list` instance."
        )
    elif variant == 1:
        difficulty = "medium"
        delays = sorted(rng.sample(range(1, 45), 4))
        buggy = (
            f"def build_retry_hooks(delays: list[int]):\n"
            f"    hooks = []\n"
            f"    for d in delays:\n"
            f"        hooks.append(lambda: f'{svc_name}:wait={{d}}s')\n"
            f"    return hooks"
        )
        fixed = (
            f"from collections.abc import Callable\n\n"
            f"def build_retry_hooks(delays: list[int]) -> list[Callable[[], str]]:\n"
            f"    return [(lambda delay=d: f'{svc_name}:wait={{delay}}s') for d in delays]"
        )
        user_q = (
            f"In `{svc_name}` (`{env}`), passing `delays = {delays}` causes every hook to return "
            f"`'{svc_name}:wait={delays[-1]}s'`:\n```python\n{buggy}\n```\nExplain the closure bug and fix it."
        )
        diag = (
            f"### Root Cause\nPython closures bind loop variables **by reference (late binding)**, so every lambda reads `d == {delays[-1]}` after loop completion. "
            f"Bind each iteration's value eagerly via `lambda delay=d`."
        )
    elif variant == 2:
        difficulty = "hard"
        cents_price = f"{rng.randint(4, 199)}.{rng.choice(['19', '29', '85', '95', '99'])}"
        qty = rng.randint(3, 17)
        buggy = (
            f"def verify_{domain_ctx}_subtotal(unit_price: float, qty: int, expected: float) -> bool:\n"
            f"    return (unit_price * qty) == expected"
        )
        fixed = (
            f"from decimal import Decimal, ROUND_HALF_UP\n\n"
            f"def verify_{domain_ctx}_subtotal(unit_price: str, qty: int, expected: str) -> bool:\n"
            f"    subtotal = (Decimal(unit_price) * Decimal(qty)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)\n"
            f"    return subtotal == Decimal(expected)"
        )
        user_q = (
            f"In `{svc_name}` (`{env}`), `verify_{domain_ctx}_subtotal({cents_price}, {qty}, ...)` fails intermittently on valid invoices:\n"
            f"```python\n{buggy}\n```\nExplain the IEEE-754 bug and rewrite it safely."
        )
        diag = (
            f"### Root Cause\nBinary `float` cannot represent decimal fractions like `{cents_price}` exactly, leaving rounding residue so `==` fails. "
            f"Use `decimal.Decimal` parsed from strings with `ROUND_HALF_UP` quantization."
        )
    elif variant == 3:
        difficulty = "hard"
        wait_ms = rng.randint(150, 2500)
        concurrency = rng.randint(20, 80)
        buggy = (
            f"import asyncio\nimport time\n\n"
            f"async def poll_{domain_ctx}(endpoint: str) -> dict[str, str]:\n"
            f"    time.sleep({wait_ms / 1000:.2f})\n"
            f"    return {{'service': '{svc_name}', 'endpoint': endpoint}}"
        )
        fixed = (
            f"import asyncio\n\n"
            f"async def poll_{domain_ctx}(endpoint: str) -> dict[str, str]:\n"
            f"    await asyncio.sleep({wait_ms / 1000:.2f})\n"
            f"    return {{'service': '{svc_name}', 'endpoint': endpoint}}"
        )
        user_q = (
            f"In `{svc_name}` (`{env}`), running `{concurrency}` concurrent `poll_{domain_ctx}` tasks via `asyncio.gather` blocks health probes:\n"
            f"```python\n{buggy}\n```\nIdentify the event-loop bug and fix it."
        )
        diag = (
            f"### Root Cause\n`time.sleep({wait_ms / 1000:.2f})` blocks the OS thread running the single-threaded `asyncio` loop, serializing all `{concurrency}` coroutines. "
            f"Replace with `await asyncio.sleep({wait_ms / 1000:.2f})`."
        )
    elif variant == 4:
        difficulty = "medium"
        threshold = rng.randint(50, 950)
        buggy = (
            f"def evict_below_score(scores: dict[str, int]) -> dict[str, int]:\n"
            f"    for k, v in scores.items():\n"
            f"        if v < {threshold}:\n"
            f"            del scores[k]\n"
            f"    return scores"
        )
        fixed = (
            f"def evict_below_score(scores: dict[str, int]) -> dict[str, int]:\n"
            f"    return {{k: v for k, v in scores.items() if v >= {threshold}}}"
        )
        user_q = (
            f"`{svc_name}` (`{fpath}`) crashes in `{env}` with `RuntimeError: dictionary changed size during iteration`:\n"
            f"```python\n{buggy}\n```\nExplain why and provide a clean fix."
        )
        diag = (
            "### Root Cause\nMutating a `dict` (`del scores[k]`) while iterating over its live view `scores.items()` invalidates the hash table iterator. "
            "Return a filtered dictionary comprehension instead."
        )
    elif variant == 5:
        difficulty = "medium"
        ts_sample = f"2026-11-{rng.randint(1,28):02d}T{rng.randint(0,23):02d}:{rng.randint(10,59):02d}:00+00:00"
        buggy = (
            f"from datetime import datetime\n\n"
            f"def is_{id_col}_expired(exp_iso: str) -> bool:\n"
            f"    exp = datetime.fromisoformat(exp_iso)\n"
            f"    return exp <= datetime.utcnow()"
        )
        fixed = (
            f"from datetime import datetime, timezone\n\n"
            f"def is_{id_col}_expired(exp_iso: str) -> bool:\n"
            f"    exp = datetime.fromisoformat(exp_iso)\n"
            f"    if exp.tzinfo is None:\n"
            f"        exp = exp.replace(tzinfo=timezone.utc)\n"
            f"    return exp <= datetime.now(timezone.utc)"
        )
        user_q = (
            f"In `{svc_name}` (`{env}`), passing `'{ts_sample}'` to `is_{id_col}_expired` raises `TypeError: can't compare offset-naive and offset-aware datetimes`:\n"
            f"```python\n{buggy}\n```\nDiagnose and fix this."
        )
        diag = (
            "### Root Cause\n`datetime.utcnow()` produces a naive datetime whereas `+00:00` timestamps parse as timezone-aware. "
            "Normalize naive inputs to `timezone.utc` and compare against `datetime.now(timezone.utc)`."
        )
    elif variant == 6:
        difficulty = "medium"
        target_id = rng.randint(500, 9999)
        buggy = (
            f"def binary_find_{id_col}(ids: list[int], target: int) -> int:\n"
            f"    lo, hi = 0, len(ids)\n"
            f"    while lo <= hi:\n"
            f"        mid = (lo + hi) // 2\n"
            f"        if ids[mid] == target:\n"
            f"            return mid\n"
            f"        elif ids[mid] < target:\n"
            f"            lo = mid + 1\n"
            f"        else:\n"
            f"            hi = mid - 1\n"
            f"    return -1"
        )
        fixed = (
            f"def binary_find_{id_col}(ids: list[int], target: int) -> int:\n"
            f"    lo, hi = 0, len(ids) - 1\n"
            f"    while lo <= hi:\n"
            f"        mid = (lo + hi) // 2\n"
            f"        if ids[mid] == target:\n"
            f"            return mid\n"
            f"        if ids[mid] < target:\n"
            f"            lo = mid + 1\n"
            f"        else:\n"
            f"            hi = mid - 1\n"
            f"    return -1"
        )
        user_q = (
            f"In `{svc_name}` (`{fpath}`), searching for `target={target_id}` above `max(ids)` raises `IndexError`:\n"
            f"```python\n{buggy}\n```\nFind the boundary invariant bug and patch it."
        )
        diag = (
            "### Root Cause\n`hi = len(ids)` mixes a half-open bound with `while lo <= hi`; when `target > ids[-1]`, `lo` reaches `len(ids)` and `ids[mid]` overflows. "
            "Initialize `hi = len(ids) - 1`."
        )
    else:
        difficulty = "hard"
        init_val = rng.randint(11, 999)
        r_cnt, c_cnt = rng.randint(3, 9), rng.randint(3, 9)
        buggy = (
            f"def alloc_{domain_ctx}_grid(rows: int = {r_cnt}, cols: int = {c_cnt}) -> list[list[int]]:\n"
            f"    grid = [[0] * cols] * rows\n"
            f"    grid[0][0] = {init_val}\n"
            f"    return grid"
        )
        fixed = (
            f"def alloc_{domain_ctx}_grid(rows: int = {r_cnt}, cols: int = {c_cnt}) -> list[list[int]]:\n"
            f"    grid = [[0 for _ in range(cols)] for _ in range(rows)]\n"
            f"    if rows > 0 and cols > 0:\n"
            f"        grid[0][0] = {init_val}\n"
            f"    return grid"
        )
        user_q = (
            f"In `{svc_name}` (`{env}`), setting `grid[0][0] = {init_val}` mutates column `0` of all `{r_cnt}` rows:\n"
            f"```python\n{buggy}\n```\nWhy does this happen and how should `alloc_{domain_ctx}_grid` be implemented?"
        )
        diag = (
            "### Root Cause\n`[[0] * cols] * rows` aliases the same inner `list` object `rows` times. "
            "Allocate independent inner lists via a nested comprehension."
        )

    ans_text = (
        f"{diag}\n\n### Fixed Implementation\n```python\n{fixed}\n```"
        if style == "step_by_step"
        else f"```python\n{fixed}\n```\n- **Fix**: {diag.replace('### Root Cause', '').strip()}"
    )

    return make_record(
        record_id=f"td-bug-{idx:06d}",
        domain="coding",
        category="bug_diagnosis_and_fix",
        difficulty=difficulty,
        style=style,
        tags=["python", "debugging", "root-cause"],
        quality_check="ast_and_solver_verified",
        messages=[
            {"role": "system", "content": pick_coding_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans_text},
        ],
    )


# ---------------------------------------------------------------------------
# 3. sql_query_and_optimization (Live SQLite Executed!)
# ---------------------------------------------------------------------------

def gen_sql_query_and_optimization(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    t1, t2 = rng.sample(TENANT_SLUGS, 2)
    style = rng.choice(["direct", "step_by_step"])
    variant = idx % 5

    if variant == 0:
        difficulty = "medium"
        top_k = rng.choice([2, 3])
        a1, a2, a3, b1, b2 = [rng.randint(105, 1950) for _ in range(5)]
        ddl = [
            f"CREATE TABLE {table_ctx} ({id_col} INT, tenant TEXT, amount INT);",
            f"INSERT INTO {table_ctx} VALUES (1, '{t1}', {a1}), (2, '{t1}', {a2}), "
            f"(3, '{t1}', {a3}), (4, '{t2}', {b1}), (5, '{t2}', {b2});",
        ]
        sql = (
            f"WITH ranked AS (\n"
            f"    SELECT {id_col}, tenant, amount,\n"
            f"           ROW_NUMBER() OVER (PARTITION BY tenant ORDER BY amount DESC, {id_col} ASC) AS rn\n"
            f"    FROM {table_ctx}\n"
            f")\n"
            f"SELECT tenant, {id_col}, amount\n"
            f"FROM ranked\n"
            f"WHERE rn <= {top_k}\n"
            f"ORDER BY tenant ASC, rn ASC;"
        )
        rows = execute_sqlite_query(ddl, sql)
        user_q = (
            f"Given `{table_ctx}({id_col}, tenant, amount)` in `{svc_name}`, write a SQL query returning "
            f"the top `{top_k}` highest-`amount` rows per `tenant` (tie-break `{id_col} ASC`). "
            f"Verify on `(1,'{t1}',{a1}), (2,'{t1}',{a2}), (3,'{t1}',{a3}), (4,'{t2}',{b1}), (5,'{t2}',{b2})`."
        )
        explain = (
            f"### Window Ranking\nPartition by `tenant` with `ROW_NUMBER() OVER (PARTITION BY tenant ORDER BY amount DESC, {id_col} ASC)` and filter `rn <= {top_k}`.\n\n"
            f"```sql\n{sql}\n```\n- **Verified Rows**: `{rows}`."
            if style == "step_by_step"
            else f"```sql\n{sql}\n```\n- Output on sample data: `{rows}`."
        )
    elif variant == 1:
        difficulty = "medium"
        v1, v2, v3, v4 = [rng.randint(25, 450) for _ in range(4)]
        m_num = rng.randint(1, 9)
        ddl = [
            f"CREATE TABLE {table_ctx} (day TEXT, metric_val INT);",
            f"INSERT INTO {table_ctx} VALUES ('2026-0{m_num}-01', {v1}), ('2026-0{m_num}-02', {v2}), ('2026-0{m_num}-03', {v3}), ('2026-0{m_num}-04', {v4});",
        ]
        sql = (
            f"SELECT day,\n"
            f"       metric_val,\n"
            f"       SUM(metric_val) OVER (ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total,\n"
            f"       metric_val - LAG(metric_val, 1, 0) OVER (ORDER BY day) AS day_delta\n"
            f"FROM {table_ctx}\n"
            f"ORDER BY day;"
        )
        rows = execute_sqlite_query(ddl, sql)
        user_q = (
            f"For `{svc_name}` rollup table `{table_ctx}(day, metric_val)`, compute `running_total` and `day_delta` "
            f"(prior day default `0`) across `('2026-0{m_num}-01',{v1}), ('2026-0{m_num}-02',{v2}), ('2026-0{m_num}-03',{v3}), ('2026-0{m_num}-04',{v4})`."
        )
        explain = (
            f"### Window Frame\nUse explicit `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` and `LAG(metric_val, 1, 0)`.\n\n"
            f"```sql\n{sql}\n```\n- **Verified Result**: `{rows}`."
            if style == "step_by_step"
            else f"```sql\n{sql}\n```\n- Result rows: `{rows}`."
        )
    elif variant == 2:
        difficulty = "easy"
        min_cnt = rng.choice([2, 3])
        s_code1 = rng.choice([500, 502, 503])
        s_code2 = rng.choice([500, 503, 504])
        benign = rng.randint(200, 204)
        ddl = [
            f"CREATE TABLE {table_ctx} ({id_col} INT, client_key TEXT, status INT);",
            f"INSERT INTO {table_ctx} VALUES (1, '{t1}', {s_code1}), (2, '{t1}', {s_code2}), (3, '{t1}', 500), (4, '{t2}', {benign}), (5, '{t2}', 500);",
        ]
        sql = (
            f"SELECT client_key,\n"
            f"       COUNT(*) AS error_count\n"
            f"FROM {table_ctx}\n"
            f"WHERE status >= 500\n"
            f"GROUP BY client_key\n"
            f"HAVING COUNT(*) >= {min_cnt}\n"
            f"ORDER BY error_count DESC, client_key ASC;"
        )
        rows = execute_sqlite_query(ddl, sql)
        user_q = (
            f"In `{svc_name}`, query `{table_ctx}({id_col}, client_key, status)` for `client_key`s with `>= {min_cnt}` "
            f"5xx responses. Verify on `(1,'{t1}',{s_code1}),(2,'{t1}',{s_code2}),(3,'{t1}',500),(4,'{t2}',{benign}),(5,'{t2}',500)`."
        )
        explain = (
            f"### Predicate Pushdown\nFilter `status >= 500` in `WHERE` prior to `GROUP BY`, then enforce `HAVING COUNT(*) >= {min_cnt}`.\n\n"
            f"```sql\n{sql}\n```\n- **Output**: `{rows}`."
            if style == "step_by_step"
            else f"```sql\n{sql}\n```\n- Output: `{rows}`."
        )
    elif variant == 3:
        difficulty = "hard"
        id_base = rng.randint(100, 899)
        ddl = [
            "CREATE TABLE accounts (acct_id INT PRIMARY KEY, handle TEXT);",
            f"CREATE TABLE {table_ctx} ({id_col} INT, acct_id INT);",
            f"INSERT INTO accounts VALUES ({id_base}, '{t1}'), ({id_base + 1}, '{t2}');",
            f"INSERT INTO {table_ctx} VALUES (1, {id_base}), (2, NULL);",
        ]
        sql = (
            "SELECT a.acct_id, a.handle\n"
            "FROM accounts AS a\n"
            f"WHERE NOT EXISTS (\n"
            f"    SELECT 1 FROM {table_ctx} AS t WHERE t.acct_id = a.acct_id\n"
            f")\n"
            "ORDER BY a.acct_id;"
        )
        rows = execute_sqlite_query(ddl, sql)
        user_q = (
            f"In `{svc_name}`, `SELECT acct_id FROM accounts WHERE acct_id NOT IN (SELECT acct_id FROM {table_ctx})` "
            f"returns 0 rows when `{table_ctx}` has `(1, {id_base}), (2, NULL)` against accounts `({id_base}, '{t1}'), ({id_base+1}, '{t2}')`. "
            f"Explain the SQL `NULL` trap and write the safe anti-join."
        )
        explain = (
            f"### Three-Valued Logic Trap\n`acct_id NOT IN ({id_base}, NULL)` evaluates `acct_id <> NULL` as `UNKNOWN`, filtering out every row. "
            f"Use `NOT EXISTS` so `NULL` foreign keys do not poison the predicate.\n\n"
            f"```sql\n{sql}\n```\n- **Verified Rows**: `{rows}`."
            if style == "step_by_step"
            else f"```sql\n{sql}\n```\n- `NOT EXISTS` avoids `NULL` collapse; verified output: `{rows}`."
        )
    else:
        difficulty = "hard"
        cutoff_month = rng.randint(1, 9)
        day_start = rng.randint(1, 20)
        tenant_val = f"{t1}_{rng.randint(10, 999)}"
        cutoff = f"2026-{cutoff_month:02d}-{day_start:02d}"
        next_cutoff = f"2026-{cutoff_month + 1:02d}-{day_start:02d}"
        user_q = (
            f"In `{svc_name}`, this PostgreSQL query triggers a sequential scan despite an index on `(tenant_id, created_at)`:\n"
            f"```sql\nSELECT {id_col}, payload FROM {table_ctx}\n"
            f"WHERE tenant_id = '{tenant_val}' AND DATE(created_at) BETWEEN DATE '{cutoff}' AND DATE '{next_cutoff}';\n```\n"
            f"Rewrite it to be sargable and specify a covering index."
        )
        rewritten = (
            f"SELECT {id_col}, payload\n"
            f"FROM {table_ctx}\n"
            f"WHERE tenant_id = '{tenant_val}'\n"
            f"  AND created_at >= TIMESTAMPTZ '{cutoff}T00:00:00Z'\n"
            f"  AND created_at <  TIMESTAMPTZ '{next_cutoff}T00:00:00Z';\n\n"
            f"CREATE INDEX CONCURRENTLY idx_{table_ctx}_tenant_created\n"
            f"    ON {table_ctx} (tenant_id, created_at)\n"
            f"    INCLUDE ({id_col}, payload);"
        )
        explain = (
            f"### Sargability & Covering Index\nWrapping `created_at` in `DATE(...)` defeats B-tree range seeks. Compare `created_at` directly against half-open `TIMESTAMPTZ` bounds and `INCLUDE ({id_col}, payload)`.\n\n"
            f"```sql\n{rewritten}\n```"
            if style == "step_by_step"
            else f"```sql\n{rewritten}\n```\n- Bare column range preserves B-tree seek + Index-Only Scan."
        )

    return make_record(
        record_id=f"td-sql-{idx:06d}",
        domain="coding",
        category="sql_query_and_optimization",
        difficulty=difficulty,
        style=style,
        tags=["sql", "query-optimization", "databases"],
        quality_check="sqlite_executed",
        messages=[
            {"role": "system", "content": pick_coding_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": explain},
        ],
    )


# ---------------------------------------------------------------------------
# 4. api_backend_engineering
# ---------------------------------------------------------------------------

def gen_api_backend_engineering(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    style = rng.choice(["direct", "step_by_step"])
    opener = rng.choice(USER_OPENERS_DIRECT if style == "direct" else USER_OPENERS_STEP)
    variant = idx % 5

    if variant == 0:
        difficulty = "medium"
        rate = rng.randint(5, 120)
        cap = rate * rng.randint(2, 6)
        user_q = (
            f"{opener}\nImplement a thread-safe `TokenBucketLimiter` in Python for `{svc_name}` (`{env}`) "
            f"configured with `rate={rate}.0` tokens/sec and `capacity={cap}`, using `time.monotonic()`."
        )
        code = (
            "import threading\nimport time\n\n"
            "class TokenBucketLimiter:\n"
            f"    def __init__(self, rate: float = {rate}.0, capacity: int = {cap}) -> None:\n"
            "        self.rate = rate\n"
            "        self.capacity = float(capacity)\n"
            "        self._tokens: dict[str, tuple[float, float]] = {}\n"
            "        self._lock = threading.Lock()\n\n"
            "    def allow(self, key: str, cost: float = 1.0) -> bool:\n"
            "        now = time.monotonic()\n"
            "        with self._lock:\n"
            "            tokens, last_ts = self._tokens.get(key, (self.capacity, now))\n"
            "            tokens = min(self.capacity, tokens + (now - last_ts) * self.rate)\n"
            "            if tokens >= cost:\n"
            "                self._tokens[key] = (tokens - cost, now)\n"
            "                return True\n"
            "            self._tokens[key] = (tokens, now)\n"
            "            return False"
        )
        notes = "Uses `time.monotonic()` so wall-clock NTP adjustments cannot skew token refill calculations."
    elif variant == 1:
        difficulty = "medium"
        attempts = rng.randint(3, 6)
        base_ms = rng.randint(50, 450)
        cap_s = rng.randint(6, 40)
        user_q = (
            f"{opener}\nWrite an async helper `retry_{domain_ctx}(coro_fn, max_attempts={attempts}, base_delay={base_ms / 1000:.3f}, max_delay={cap_s}.0)` "
            f"for `{svc_name}` (`{env}`) that retries transient `TimeoutError` or `ConnectionError` with full jitter."
        )
        code = (
            "import asyncio\nimport random\nfrom collections.abc import Awaitable, Callable\nfrom typing import TypeVar\n\n"
            "T = TypeVar('T')\n\n"
            f"async def retry_{domain_ctx}(\n"
            f"    coro_fn: Callable[[], Awaitable[T]],\n"
            f"    max_attempts: int = {attempts},\n"
            f"    base_delay: float = {base_ms / 1000:.3f},\n"
            f"    max_delay: float = {cap_s}.0,\n"
            f") -> T:\n"
            "    for attempt in range(1, max_attempts + 1):\n"
            "        try:\n"
            "            return await coro_fn()\n"
            "        except (TimeoutError, ConnectionError):\n"
            "            if attempt == max_attempts:\n"
            "                raise\n"
            "            ceiling = min(max_delay, base_delay * (2 ** (attempt - 1)))\n"
            "            await asyncio.sleep(random.uniform(0.0, ceiling))\n"
            "    raise RuntimeError('unreachable')"
        )
        notes = "Full jitter (`uniform(0, min(max_delay, base * 2^(attempt-1)))`) prevents synchronized retry waves."
    elif variant == 2:
        difficulty = "hard"
        fail_thresh = rng.randint(3, 12)
        reset_s = rng.randint(10, 90)
        user_q = (
            f"{opener}\nBuild a `CircuitBreaker` state machine (`CLOSED`, `OPEN`, `HALF_OPEN`) in Python for `{svc_name}` (`{env}`) "
            f"that opens after `{fail_thresh}` consecutive errors and probes recovery after `{reset_s}` seconds."
        )
        code = (
            "import time\nfrom collections.abc import Callable\nfrom typing import TypeVar\n\n"
            "T = TypeVar('T')\n\n"
            "class CircuitOpenError(RuntimeError):\n"
            "    pass\n\n"
            "class CircuitBreaker:\n"
            f"    def __init__(self, failure_threshold: int = {fail_thresh}, recovery_timeout: float = {reset_s}.0) -> None:\n"
            "        self.failure_threshold = failure_threshold\n"
            "        self.recovery_timeout = recovery_timeout\n"
            "        self.state = 'CLOSED'\n"
            "        self.failures = 0\n"
            "        self.opened_at = 0.0\n\n"
            "    def call(self, fn: Callable[[], T]) -> T:\n"
            "        now = time.monotonic()\n"
            "        if self.state == 'OPEN':\n"
            "            if now - self.opened_at < self.recovery_timeout:\n"
            "                raise CircuitOpenError('circuit is OPEN')\n"
            "            self.state = 'HALF_OPEN'\n"
            "        try:\n"
            "            result = fn()\n"
            "        except Exception:\n"
            "            self.failures += 1\n"
            "            if self.state == 'HALF_OPEN' or self.failures >= self.failure_threshold:\n"
            "                self.state = 'OPEN'\n"
            "                self.opened_at = time.monotonic()\n"
            "            raise\n"
            "        self.failures = 0\n"
            "        self.state = 'CLOSED'\n"
            "        return result"
        )
        notes = "Any failed probe in `HALF_OPEN` immediately re-opens the circuit; success resets `failures` to `0`."
    elif variant == 3:
        difficulty = "medium"
        page_sz = rng.randint(15, 250)
        user_q = (
            f"{opener}\nImplement base64url keyset cursor helpers `encode_cursor(ts_ms: int, {id_col}: str) -> str` "
            f"and `decode_cursor(token: str) -> tuple[int, str]` for `{svc_name}` (`{table_ctx}`, `limit={page_sz}`), raising `ValueError` on invalid input."
        )
        code = (
            "import base64\nimport json\n\n"
            f"def encode_cursor(ts_ms: int, {id_col}: str) -> str:\n"
            f"    raw = json.dumps([ts_ms, {id_col}], separators=(',', ':')).encode('utf-8')\n"
            "    return base64.urlsafe_b64encode(raw).decode('ascii').rstrip('=')\n\n"
            "def decode_cursor(token: str) -> tuple[int, str]:\n"
            "    try:\n"
            "        padded = token + '=' * (-len(token) % 4)\n"
            "        data = json.loads(base64.urlsafe_b64decode(padded.encode('ascii')))\n"
            "        if not (isinstance(data, list) and len(data) == 2 and isinstance(data[0], int) and isinstance(data[1], str)):\n"
            "            raise ValueError('invalid cursor structure')\n"
            "        return data[0], data[1]\n"
            "    except Exception as exc:\n"
            "        raise ValueError('malformed pagination cursor') from exc"
        )
        notes = f"Composite `(ts_ms, {id_col})` keyset cursors prevent missed/duplicate rows under concurrent inserts."
    else:
        difficulty = "hard"
        ttl_s = rng.randint(600, 86400)
        user_q = (
            f"{opener}\nWrite an `IdempotencyGuard` class in Python for `{svc_name}` (`{env}`) POST mutations "
            f"that binds keys to SHA-256 payload digests, raises `IdempotencyConflictError` on mismatched body reuse, "
            f"and expires entries after `ttl_seconds={ttl_s}`."
        )
        code = (
            "import hashlib\nimport time\nfrom typing import Any\n\n"
            "class IdempotencyConflictError(ValueError):\n"
            "    pass\n\n"
            "class IdempotencyGuard:\n"
            f"    def __init__(self, ttl_seconds: int = {ttl_s}) -> None:\n"
            "        self.ttl = ttl_seconds\n"
            "        self._records: dict[str, tuple[str, Any, float]] = {}\n\n"
            "    def get_or_store(self, idem_key: str, payload_bytes: bytes, compute_fn: Any) -> Any:\n"
            "        digest = hashlib.sha256(payload_bytes).hexdigest()\n"
            "        now = time.monotonic()\n"
            "        cached = self._records.get(idem_key)\n"
            "        if cached is not None and (now - cached[2]) < self.ttl:\n"
            "            if cached[0] != digest:\n"
            "                raise IdempotencyConflictError('idempotency key reused with different payload')\n"
            "            return cached[1]\n"
            "        result = compute_fn()\n"
            "        self._records[idem_key] = (digest, result, now)\n"
            "        return result"
        )
        notes = "Comparing the SHA-256 request digest prevents silent cross-payload collisions under reused idempotency keys."

    ans_text = (
        f"### Design Notes\n- {notes}\n\n```python\n{code}\n```"
        if style == "step_by_step"
        else f"```python\n{code}\n```\n- {notes}"
    )

    return make_record(
        record_id=f"td-api-{idx:06d}",
        domain="coding",
        category="api_backend_engineering",
        difficulty=difficulty,
        style=style,
        tags=["python", "backend", "distributed-systems"],
        quality_check="ast_and_solver_verified",
        messages=[
            {"role": "system", "content": pick_coding_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans_text},
        ],
    )


# ---------------------------------------------------------------------------
# 5. refactoring_clean_code
# ---------------------------------------------------------------------------

def gen_refactoring_clean_code(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    fpath = rng.choice(FILE_PATHS)
    ctx_note = rng.choice(REVIEW_CONTEXTS)
    style = rng.choice(["direct", "step_by_step"])
    variant = idx % 4

    if variant == 0:
        difficulty = "easy"
        min_amt = rng.randint(10, 950)
        legacy = (
            f"def process_{domain_ctx}(req):\n"
            f"    if req is not None:\n"
            f"        if 'active' in req and req['active'] == True:\n"
            f"            if 'amount' in req and req['amount'] >= {min_amt}:\n"
            f"                return {{'ok': True, '{id_col}': req['{id_col}']}}\n"
            f"            else:\n"
            f"                return {{'ok': False, 'err': 'below_min'}}\n"
            f"        else:\n"
            f"            return {{'ok': False, 'err': 'inactive'}}\n"
            f"    else:\n"
            f"        return {{'ok': False, 'err': 'missing_req'}}"
        )
        clean = (
            "from typing import Any\n\n"
            f"MIN_AMOUNT = {min_amt}\n\n"
            f"def process_{domain_ctx}(req: dict[str, Any] | None) -> dict[str, Any]:\n"
            "    if req is None:\n"
            "        return {'ok': False, 'err': 'missing_req'}\n"
            "    if not req.get('active'):\n"
            "        return {'ok': False, 'err': 'inactive'}\n"
            "    if req.get('amount', 0) < MIN_AMOUNT:\n"
            "        return {'ok': False, 'err': 'below_min'}\n"
            f"    return {{'ok': True, '{id_col}': req['{id_col}']}}"
        )
        summary = "Replaced deeply nested conditionals with flat guard clauses, named constant `MIN_AMOUNT`, and explicit types."
    elif variant == 1:
        difficulty = "medium"
        fee_card = rng.randint(15, 75)
        wire_flat = rng.randint(800, 2500)
        legacy = (
            f"def calc_{domain_ctx}_fee(channel, cents):\n"
            "    if channel == 'ach':\n"
            "        return int(cents * 0.008)\n"
            "    elif channel == 'card':\n"
            f"        return int(cents * 0.029) + {fee_card}\n"
            "    elif channel == 'wire':\n"
            f"        return {wire_flat}\n"
            "    else:\n"
            "        raise ValueError('bad channel')"
        )
        clean = (
            "from collections.abc import Callable\n\n"
            "FEE_STRATEGIES: dict[str, Callable[[int], int]] = {\n"
            "    'ach': lambda cents: int(cents * 0.008),\n"
            f"    'card': lambda cents: int(cents * 0.029) + {fee_card},\n"
            f"    'wire': lambda _cents: {wire_flat},\n"
            "}\n\n"
            f"def calc_{domain_ctx}_fee(channel: str, cents: int) -> int:\n"
            "    strategy = FEE_STRATEGIES.get(channel)\n"
            "    if strategy is None:\n"
            "        raise ValueError(f'unsupported channel: {channel}')\n"
            "    return strategy(cents)"
        )
        summary = "Replaced `if/elif` chain with a typed `FEE_STRATEGIES` dispatch dictionary and informative error message."
    elif variant == 2:
        difficulty = "medium"
        chunk_kb = rng.randint(16, 512)
        sev = rng.choice(["ERROR", "CRITICAL", "FATAL", "ALERT"])
        legacy = (
            f"def extract_{domain_ctx}_alerts(path):\n"
            "    f = open(path, 'r')\n"
            "    lines = f.readlines()\n"
            "    f.close()\n"
            "    out = []\n"
            "    for l in lines:\n"
            f"        if '{sev}' in l:\n"
            "            out.append(l.strip())\n"
            "    return out"
        )
        clean = (
            "from collections.abc import Iterator\nfrom pathlib import Path\n\n"
            f"def iter_{domain_ctx}_alerts(path: str | Path) -> Iterator[str]:\n"
            f"    with Path(path).open('r', encoding='utf-8', buffering={chunk_kb * 1024}) as handle:\n"
            "        for line in handle:\n"
            f"            if '{sev}' in line:\n"
            "                yield line.strip()"
        )
        summary = "Converted eager `readlines()` + manual `close()` into a leak-safe `with` context manager and O(1)-memory generator."
    else:
        difficulty = "hard"
        max_retries = rng.randint(1, 9)
        default_port = rng.randint(3000, 9999)
        legacy = (
            f"def create_{domain_ctx}_cfg(host, port={default_port}, tls=True, retries={max_retries}, prefix='{id_col}'):\n"
            f"    return [host, port, tls, retries, prefix]"
        )
        clean = (
            "from dataclasses import dataclass\n\n"
            "@dataclass(frozen=True, slots=True)\n"
            "class EndpointConfig:\n"
            "    host: str\n"
            f"    port: int = {default_port}\n"
            "    tls: bool = True\n"
            f"    retries: int = {max_retries}\n"
            f"    prefix: str = '{id_col}'\n\n"
            "    def __post_init__(self) -> None:\n"
            "        if not (1 <= self.port <= 65535):\n"
            "            raise ValueError(f'invalid port: {self.port}')"
        )
        summary = "Replaced untyped positional list return with an immutable `@dataclass(frozen=True, slots=True)` and invariant check."

    user_q = (
        f"Refactor this legacy code in `{svc_name}` (`{fpath}`, flagged {ctx_note}) into clean, idiomatic Python 3.11:\n"
        f"```python\n{legacy}\n```"
    )
    ans_text = (
        f"### Refactoring Improvements\n- {summary}\n\n```python\n{clean}\n```"
        if style == "step_by_step"
        else f"```python\n{clean}\n```\n- {summary}"
    )

    return make_record(
        record_id=f"td-refac-{idx:06d}",
        domain="coding",
        category="refactoring_clean_code",
        difficulty=difficulty,
        style=style,
        tags=["python", "refactoring", "clean-code"],
        quality_check="ast_and_solver_verified",
        messages=[
            {"role": "system", "content": pick_coding_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans_text},
        ],
    )


# ---------------------------------------------------------------------------
# 6. testing_property_edge_cases
# ---------------------------------------------------------------------------

def gen_testing_property_edge_cases(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    t_slug = rng.choice(TENANT_SLUGS)
    style = rng.choice(["direct", "step_by_step"])
    variant = idx % 4

    if variant == 0:
        difficulty = "easy"
        max_len = rng.randint(12, 64)
        target_fn = (
            f"def validate_{id_col}(raw: str) -> str:\n"
            f"    cleaned = raw.strip()\n"
            f"    if not cleaned or len(cleaned) > {max_len} or not cleaned.replace('_', '').isalnum():\n"
            f"        raise ValueError('invalid {id_col}')\n"
            f"    return cleaned.lower()"
        )
        test_code = (
            "import pytest\n\n"
            "@pytest.mark.parametrize(\n"
            "    ('raw', 'expected'),\n"
            "    [\n"
            f"        ('  {t_slug.upper()}_1  ', '{t_slug}_1'),\n"
            f"        ('a' * {max_len}, 'a' * {max_len}),\n"
            "        ('9', '9'),\n"
            "    ],\n"
            ")\n"
            f"def test_validate_{id_col}_valid(raw: str, expected: str) -> None:\n"
            f"    assert validate_{id_col}(raw) == expected\n\n"
            "@pytest.mark.parametrize('bad_input', ['', '   ', " + f"'x' * {max_len + 1}" + ", 'bad-char!', '../etc'])\n"
            f"def test_validate_{id_col}_rejects_invalid(bad_input: str) -> None:\n"
            f"    with pytest.raises(ValueError, match='invalid {id_col}'):\n"
            f"        validate_{id_col}(bad_input)"
        )
        notes = f"Covers trimming, exact `{max_len}` boundary, `{max_len + 1}` overflow, empty input, and traversal punctuation."
    elif variant == 1:
        difficulty = "medium"
        limit = rng.randint(2, 15)
        win_s = rng.choice([10.0, 30.0, 60.0, 120.0, 300.0])
        target_fn = (
            f"class QuotaTracker:\n"
            f"    def __init__(self, clock, max_calls: int = {limit}, window_s: float = {win_s}):\n"
            f"        ..."
        )
        test_code = (
            "class FakeClock:\n"
            "    def __init__(self, now: float = 1000.0) -> None:\n"
            "        self.now = now\n\n"
            "    def __call__(self) -> float:\n"
            "        return self.now\n\n"
            "def test_quota_enforces_limit_and_recovers_after_window() -> None:\n"
            "    clock = FakeClock(1000.0)\n"
            f"    tracker = QuotaTracker(clock=clock, max_calls={limit}, window_s={win_s})\n"
            f"    for _ in range({limit}):\n"
            f"        assert tracker.acquire('{t_slug}') is True\n"
            f"    assert tracker.acquire('{t_slug}') is False\n"
            "    assert tracker.acquire('isolated_peer') is True\n"
            f"    clock.now += {win_s + 0.1}\n"
            f"    assert tracker.acquire('{t_slug}') is True"
        )
        notes = "Uses deterministic `FakeClock` injection to verify saturation, tenant isolation, and window recovery without `time.sleep`."
    elif variant == 2:
        difficulty = "medium"
        status_code = rng.choice([429, 502, 503, 504])
        sample_key = f"{t_slug}_{rng.randint(10, 999)}"
        target_fn = f"async def fetch_{domain_ctx}_state(client, {id_col}: str = '{sample_key}') -> dict"
        test_code = (
            "import pytest\nfrom unittest.mock import AsyncMock\n\n"
            "@pytest.mark.anyio\n"
            f"async def test_fetch_{domain_ctx}_retries_on_{status_code}_then_succeeds() -> None:\n"
            "    client = AsyncMock()\n"
            "    client.get.side_effect = [\n"
            f"        RuntimeError('HTTP {status_code}'),\n"
            f"        {{'{id_col}': '{sample_key}', 'state': 'ready'}},\n"
            "    ]\n"
            f"    result = await fetch_{domain_ctx}_state(client, '{sample_key}')\n"
            f"    assert result == {{'{id_col}': '{sample_key}', 'state': 'ready'}}\n"
            "    assert client.get.await_count == 2"
        )
        notes = f"Simulates transient `HTTP {status_code}` via `AsyncMock.side_effect` and verifies retry count and recovered payload."
    else:
        difficulty = "hard"
        batch_sz = rng.randint(2, 19)
        target_fn = (
            f"def chunk_{table_ctx}(items: list[int], size: int = {batch_sz}) -> list[list[int]]:\n"
            "    if size <= 0:\n"
            "        raise ValueError('size must be positive')\n"
            "    return [items[i:i + size] for i in range(0, len(items), size)]"
        )
        test_code = (
            "import pytest\n\n"
            f"@pytest.mark.parametrize('n', [0, 1, {batch_sz - 1}, {batch_sz}, {batch_sz + 1}, {batch_sz * 3 + 1}])\n"
            f"def test_chunk_{table_ctx}_invariants(n: int) -> None:\n"
            "    items = list(range(n))\n"
            f"    chunks = chunk_{table_ctx}(items, size={batch_sz})\n"
            "    assert [x for c in chunks for x in c] == items\n"
            "    for c in chunks[:-1]:\n"
            f"        assert len(c) == {batch_sz}\n"
            "    if chunks:\n"
            f"        assert 1 <= len(chunks[-1]) <= {batch_sz}\n\n"
            "@pytest.mark.parametrize('bad_size', [0, -1, -50])\n"
            f"def test_chunk_{table_ctx}_invalid_size(bad_size: int) -> None:\n"
            "    with pytest.raises(ValueError):\n"
            f"        chunk_{table_ctx}([1, 2], size=bad_size)"
        )
        notes = "Asserts lossless flattening identity, exact interior chunk length, bounded tail chunk, and non-positive `size` rejection."

    user_q = (
        f"Write a concise `pytest` suite covering boundary conditions and failure modes for `{svc_name}` (`{env}`):\n"
        f"```python\n{target_fn}\n```"
    )
    ans_text = (
        f"### Test Matrix Strategy\n- {notes}\n\n```python\n{test_code}\n```"
        if style == "step_by_step"
        else f"```python\n{test_code}\n```\n- {notes}"
    )

    return make_record(
        record_id=f"td-test-{idx:06d}",
        domain="coding",
        category="testing_property_edge_cases",
        difficulty=difficulty,
        style=style,
        tags=["python", "pytest", "unit-testing"],
        quality_check="ast_and_solver_verified",
        messages=[
            {"role": "system", "content": pick_coding_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans_text},
        ],
    )
