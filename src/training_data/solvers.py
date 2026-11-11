"""Programmatic ground-truth solvers for verifying math, code, SQL, CIDR/ACL, and CVSS v3.1 scores."""

from __future__ import annotations

import ast
import heapq
import ipaddress
import math
import sqlite3
from collections import deque
from typing import Any


# ---------------------------------------------------------------------------
# 1. Official FIRST.org CVSS v3.1 Base Score Engine
# ---------------------------------------------------------------------------

CVSS_AV = {"N": (0.85, "Network"), "A": (0.62, "Adjacent"), "L": (0.55, "Local"), "P": (0.20, "Physical")}
CVSS_AC = {"L": (0.77, "Low"), "H": (0.44, "High")}
CVSS_PR_U = {"N": (0.85, "None"), "L": (0.62, "Low"), "H": (0.27, "High")}
CVSS_PR_C = {"N": (0.85, "None"), "L": (0.68, "Low"), "H": (0.50, "High")}
CVSS_UI = {"N": (0.85, "None"), "R": (0.62, "Required")}
CVSS_S = {"U": "Unchanged", "C": "Changed"}
CVSS_CIA = {"N": (0.00, "None"), "L": (0.22, "Low"), "H": (0.56, "High")}


def cvss_roundup(val: float) -> float:
    """Official CVSS v3.1 Roundup function using integer arithmetic to avoid float drift."""
    int_input = round(val * 100000)
    if (int_input % 10000) == 0:
        return int_input / 100000.0
    return (math.floor(int_input / 10000) + 1) / 10.0


def compute_cvss31(
    av: str, ac: str, pr: str, ui: str, s: str, c: str, i: str, a: str
) -> dict[str, Any]:
    """Compute exact CVSS v3.1 vector, sub-scores, base score, and severity."""
    av_w, av_name = CVSS_AV[av]
    ac_w, ac_name = CVSS_AC[ac]
    pr_w, pr_name = (CVSS_PR_C if s == "C" else CVSS_PR_U)[pr]
    ui_w, ui_name = CVSS_UI[ui]
    c_w, c_name = CVSS_CIA[c]
    i_w, i_name = CVSS_CIA[i]
    a_w, a_name = CVSS_CIA[a]

    iss = 1.0 - ((1.0 - c_w) * (1.0 - i_w) * (1.0 - a_w))
    if s == "U":
        impact = 6.42 * iss
    else:
        impact = 7.52 * (iss - 0.029) - 3.25 * ((iss - 0.02) ** 15)

    exploitability = 8.22 * av_w * ac_w * pr_w * ui_w

    if impact <= 0:
        base_score = 0.0
    elif s == "U":
        base_score = cvss_roundup(min(impact + exploitability, 10.0))
    else:
        base_score = cvss_roundup(min(1.08 * (impact + exploitability), 10.0))

    if base_score == 0.0:
        severity = "None"
    elif base_score <= 3.9:
        severity = "Low"
    elif base_score <= 6.9:
        severity = "Medium"
    elif base_score <= 8.9:
        severity = "High"
    else:
        severity = "Critical"

    vector = f"CVSS:3.1/AV:{av}/AC:{ac}/PR:{pr}/UI:{ui}/S:{s}/C:{c}/I:{i}/A:{a}"
    return {
        "vector": vector,
        "score": round(base_score, 1),
        "severity": severity,
        "iss": round(iss, 4),
        "impact": round(impact, 2),
        "exploitability": round(exploitability, 2),
        "labels": {
            "AV": av_name,
            "AC": ac_name,
            "PR": pr_name,
            "UI": ui_name,
            "S": CVSS_S[s],
            "C": c_name,
            "I": i_name,
            "A": a_name,
        },
    }


# ---------------------------------------------------------------------------
# 2. Network CIDR & Firewall ACL Ground-Truth Solver
# ---------------------------------------------------------------------------

def analyze_ipv4_cidr(cidr: str) -> dict[str, Any]:
    net = ipaddress.IPv4Network(cidr, strict=False)
    total = net.num_addresses
    if net.prefixlen <= 30:
        usable = total - 2
        first_host = str(net.network_address + 1)
        last_host = str(net.broadcast_address - 1)
    elif net.prefixlen == 31:
        usable = 2
        first_host = str(net.network_address)
        last_host = str(net.broadcast_address)
    else:
        usable = 1
        first_host = str(net.network_address)
        last_host = str(net.network_address)
    return {
        "canonical": str(net),
        "network": str(net.network_address),
        "broadcast": str(net.broadcast_address),
        "netmask": str(net.netmask),
        "prefixlen": net.prefixlen,
        "total_addresses": total,
        "usable_hosts": usable,
        "first_usable": first_host,
        "last_usable": last_host,
    }


def evaluate_firewall_rules(
    rules: list[dict[str, Any]], src_ip: str, dst_port: int, proto: str
) -> tuple[str, int | None, list[tuple[int, int]]]:
    """
    Evaluate ordered ACL rules and also detect shadowed rules.
    Each rule: {"id": int, "action": "ALLOW"|"DENY", "src": cidr, "port": int|"ANY", "proto": "TCP"|"UDP"|"ANY"}
    Returns (verdict, matched_rule_id_or_None_for_default_deny, shadowed_pairs[(later_id, earlier_id)]).
    """
    pkt_ip = ipaddress.IPv4Address(src_ip)
    matched_verdict = "DENY"
    matched_id: int | None = None

    parsed_nets = [ipaddress.IPv4Network(r["src"], strict=False) for r in rules]

    for r, net in zip(rules, parsed_nets):
        proto_ok = r["proto"] == "ANY" or r["proto"] == proto
        port_ok = r["port"] == "ANY" or r["port"] == dst_port
        if proto_ok and port_ok and (pkt_ip in net):
            matched_verdict = r["action"]
            matched_id = r["id"]
            break

    shadowed: list[tuple[int, int]] = []
    for idx_b in range(1, len(rules)):
        rb = rules[idx_b]
        net_b = parsed_nets[idx_b]
        for idx_a in range(idx_b):
            ra = rules[idx_a]
            net_a = parsed_nets[idx_a]
            proto_covers = ra["proto"] == "ANY" or ra["proto"] == rb["proto"]
            port_covers = ra["port"] == "ANY" or ra["port"] == rb["port"]
            if proto_covers and port_covers and net_b.subnet_of(net_a):
                shadowed.append((rb["id"], ra["id"]))
                break

    return matched_verdict, matched_id, shadowed


# ---------------------------------------------------------------------------
# 3. Python AST & Execution Verification Helpers
# ---------------------------------------------------------------------------

def verify_python_ast(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


_SQLITE_CONN: sqlite3.Connection | None = None


def execute_sqlite_query(ddl_and_seed: list[str], query: str) -> list[tuple[Any, ...]]:
    global _SQLITE_CONN
    if _SQLITE_CONN is None:
        _SQLITE_CONN = sqlite3.connect(":memory:")
    cur = _SQLITE_CONN.cursor()
    cur.execute("SAVEPOINT sp_run;")
    try:
        for stmt in ddl_and_seed:
            cur.execute(stmt)
        cur.execute(query)
        return cur.fetchall()
    finally:
        cur.execute("ROLLBACK TO sp_run;")
        cur.execute("RELEASE sp_run;")


# ---------------------------------------------------------------------------
# 4. Exact Algorithm Solvers (Ground-Truth Output Generators)
# ---------------------------------------------------------------------------

def solve_max_window_sum(nums: list[int], k: int) -> int:
    curr = sum(nums[:k])
    best = curr
    for idx in range(k, len(nums)):
        curr += nums[idx] - nums[idx - k]
        if curr > best:
            best = curr
    return best


def solve_subarray_at_most_budget(costs: list[int], budget: int) -> int:
    left = 0
    running = 0
    best = 0
    for right, val in enumerate(costs):
        running += val
        while running > budget and left <= right:
            running -= costs[left]
            left += 1
        best = max(best, right - left + 1)
    return best


def solve_merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [ordered[0][:]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def solve_min_ship_capacity(weights: list[int], max_batches: int) -> int:
    lo, hi = max(weights), sum(weights)
    while lo < hi:
        mid = (lo + hi) // 2
        batches = 1
        curr = 0
        for w in weights:
            if curr + w > mid:
                batches += 1
                curr = w
            else:
                curr += w
        if batches <= max_batches:
            hi = mid
        else:
            lo = mid + 1
    return lo


def solve_dijkstra(n: int, edges: list[tuple[int, int, int]], src: int, dst: int) -> int:
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
    dist = [math.inf] * n
    dist[src] = 0
    pq: list[tuple[int, int]] = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == dst:
            return int(d)
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return -1 if math.isinf(dist[dst]) else int(dist[dst])


def solve_topo_order(nodes: list[str], deps: list[tuple[str, str]]) -> list[str] | None:
    """Return deterministic lexicographical topological order where (a, b) means a before b."""
    indeg = {u: 0 for u in nodes}
    adj: dict[str, list[str]] = {u: [] for u in nodes}
    for u, v in deps:
        adj[u].append(v)
        indeg[v] += 1
    heap = [u for u in nodes if indeg[u] == 0]
    heapq.heapify(heap)
    order: list[str] = []
    while heap:
        u = heapq.heappop(heap)
        order.append(u)
        for nxt in adj[u]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                heapq.heappush(heap, nxt)
    return order if len(order) == len(nodes) else None


def solve_next_greater_latency(vals: list[int]) -> list[int]:
    res = [-1] * len(vals)
    stack: list[int] = []
    for idx, val in enumerate(vals):
        while stack and vals[stack[-1]] < val:
            prev = stack.pop()
            res[prev] = idx - prev
        stack.append(idx)
    return res


def solve_min_coins(coins: list[int], target: int) -> int:
    dp = [target + 1] * (target + 1)
    dp[0] = 0
    for amt in range(1, target + 1):
        for c in coins:
            if c <= amt and dp[amt - c] + 1 < dp[amt]:
                dp[amt] = dp[amt - c] + 1
    return dp[target] if dp[target] <= target else -1


def solve_shortest_grid_bfs(grid: list[list[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    if grid[0][0] == 1 or grid[rows - 1][cols - 1] == 1:
        return -1
    q: deque[tuple[int, int, int]] = deque([(0, 0, 1)])
    seen = {(0, 0)}
    while q:
        r, c, d = q.popleft()
        if (r, c) == (rows - 1, cols - 1):
            return d
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc, d + 1))
    return -1
