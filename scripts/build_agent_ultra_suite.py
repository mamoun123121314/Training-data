#!/usr/bin/env python3
"""
Synthesize & Compress the Pure Autonomous Agent Fine-Tuning Suite (SFT + DPO):
  1. Persisted In-Repo Pack (`data/agent_ultra/`):
       - 450,000 Pure-Agent Ultra SFT trajectories across all 42 Agent categories (~41 MiB .jsonl.xz)
       - 200,000 Agent DPO / ORPO Preference pairs across 8 failure taxonomies (~7 MiB .jsonl.xz)
       -> Fits cleanly inside Git/snapshot headroom (~48 MiB added -> ~111 MiB total persisted repo).
  2. Massive Local-Disk Pack (`out/agent-ultra-pack/`):
       - 4,500,000 Pure-Agent Ultra SFT trajectories + 500,000 Agent DPO pairs = 5,000,000 records (~10.8 GB raw).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training_data.agent_dpo import DPO_TAXONOMIES, validate_dpo_pair  # noqa: E402
from training_data.agent_ultra import (  # noqa: E402
    AGENT_DPO_TOTAL,
    AGENT_SFT_TOTAL,
    AgentDPODataset,
    AgentUltraDataset,
    NUM_AGENT_CATEGORIES,
)
from training_data.generators import AGENT_GENERATORS  # noqa: E402
from training_data.validator import validate_record  # noqa: E402


def _write_sft_shard(args: tuple[str, str, int, int, int, int]) -> dict[str, Any]:
    split, out_path_s, start_idx, count, seed, xz_preset = args
    out_path = Path(out_path_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = AgentUltraDataset(split=split, seed=seed)

    raw_bytes = 0
    with out_path.open("wb") as out_fh:
        proc = subprocess.Popen(
            ["xz", f"--lzma2=preset={xz_preset},dict=32MiB", "-T1", "--check=crc64"],
            stdin=subprocess.PIPE,
            stdout=out_fh,
        )
        assert proc.stdin is not None
        buf: list[bytes] = []
        try:
            for idx in range(start_idx, start_idx + count):
                rec = ds[idx]
                ok, reason = validate_record(rec)
                if not ok:
                    raise ValueError(f"Agent SFT validation failed at {split}[{idx}] ({rec['category']}): {reason}")
                b = (json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
                raw_bytes += len(b)
                buf.append(b)
                if len(buf) >= 2048:
                    proc.stdin.write(b"".join(buf))
                    buf.clear()
            if buf:
                proc.stdin.write(b"".join(buf))
                buf.clear()
        finally:
            proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"xz failed on {out_path}: {ret}")

    h = hashlib.sha256()
    comp_bytes = 0
    with out_path.open("rb") as rfh:
        while chunk := rfh.read(1 << 20):
            comp_bytes += len(chunk)
            h.update(chunk)

    return {
        "kind": "agent_sft",
        "split": split,
        "file": out_path.name,
        "start_index": start_idx,
        "records": count,
        "uncompressed_bytes": raw_bytes,
        "compressed_bytes": comp_bytes,
        "sha256": h.hexdigest(),
    }


def _write_dpo_shard(args: tuple[str, str, int, int, int, int]) -> dict[str, Any]:
    split_label, out_path_s, start_idx, count, seed, xz_preset = args
    out_path = Path(out_path_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = AgentDPODataset(seed=seed)

    raw_bytes = 0
    with out_path.open("wb") as out_fh:
        proc = subprocess.Popen(
            ["xz", f"--lzma2=preset={xz_preset},dict=32MiB", "-T1", "--check=crc64"],
            stdin=subprocess.PIPE,
            stdout=out_fh,
        )
        assert proc.stdin is not None
        buf: list[bytes] = []
        try:
            for idx in range(start_idx, start_idx + count):
                pair = ds[idx]
                ok, reason = validate_dpo_pair(pair)
                if not ok:
                    raise ValueError(f"Agent DPO validation failed at [{idx}]: {reason}")
                b = (json.dumps(pair, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
                raw_bytes += len(b)
                buf.append(b)
                if len(buf) >= 2048:
                    proc.stdin.write(b"".join(buf))
                    buf.clear()
            if buf:
                proc.stdin.write(b"".join(buf))
                buf.clear()
        finally:
            proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"xz failed on {out_path}: {ret}")

    h = hashlib.sha256()
    comp_bytes = 0
    with out_path.open("rb") as rfh:
        while chunk := rfh.read(1 << 20):
            comp_bytes += len(chunk)
            h.update(chunk)

    return {
        "kind": "agent_dpo",
        "split": split_label,
        "file": out_path.name,
        "start_index": start_idx,
        "records": count,
        "uncompressed_bytes": raw_bytes,
        "compressed_bytes": comp_bytes,
        "sha256": h.hexdigest(),
    }


def _write_agent_showcase(path: Path, seed: int = 42) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds_sft = AgentUltraDataset("train", seed=seed)
    ds_dpo = AgentDPODataset(seed=seed)

    wanted_cats = [
        "swe_multi_file_bugfix_trajectory",
        "mcp_server_discovery_and_invocation",
        "agent_browser_dom_navigation_workflow",
        "agent_idempotency_and_rollback_safety",
        "agent_no_tool_needed_restraint",
        "agent_trajectory_judge_and_reward_critique",
    ]
    picked_sft: dict[str, dict[str, Any]] = {}
    for i in range(4000):
        rec = ds_sft[i]
        c = rec["category"]
        if c in wanted_cats and c not in picked_sft:
            picked_sft[c] = rec
        if len(picked_sft) == len(wanted_cats):
            break

    lines = [
        "# Autonomous Agent Training & Fine-Tuning Showcase (`v5.0.0`)",
        "",
        "Curated examples from **`AgentUltraDataset` (`1,050,000,000` SFT trajectories across 42 Agent categories)**",
        "and **`AgentDPODataset` (`100,000,000` `chosen` vs `rejected` preference alignment pairs)**.",
        "",
        "---",
        "",
        "## Part 1 — Agent SFT Multi-Turn Trajectories & Reasoning",
        "",
    ]
    for idx_n, cat in enumerate(wanted_cats, start=1):
        rec = picked_sft[cat]
        lines.append(f"### {idx_n}. `{cat}` (`id: {rec['id']}` | `difficulty: {rec['difficulty']}` | `messages: {len(rec['messages'])}`)")
        lines.append("")
        for msg in rec["messages"]:
            lines.append(f"**[{msg['role'].upper()}]**")
            if msg["role"] == "tool":
                lines.append(f"```json\n{msg['content']}\n```")
            else:
                lines.append(msg["content"])
            lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Part 2 — Agent DPO / ORPO Preference Pairs (`chosen` vs `rejected`)")
    lines.append("")
    seen_tax: set[str] = set()
    for i in range(400):
        pair = ds_dpo[i]
        tax = pair["rejection_taxonomy"]
        if tax in seen_tax:
            continue
        seen_tax.add(tax)
        lines.append(f"### DPO Taxonomy: `{tax}` (`id: {pair['id']}`)")
        lines.append("")
        lines.append("**[CONVERSATION PROMPT]**")
        for pm in pair["prompt"]:
            lines.append(f"- `{pm['role'].upper()}`: {pm['content']}")
        lines.append("")
        lines.append("**[CHOSEN (Preferred Agent Turn)]**")
        lines.append(pair["chosen"][0]["content"])
        lines.append("")
        lines.append("**[REJECTED (Penalized Agent Failure Mode)]**")
        lines.append(pair["rejected"][0]["content"])
        lines.append("")
        lines.append("---")
        lines.append("")
        if len(seen_tax) == 4:
            break

    path.write_text("\n".join(lines), encoding="utf-8")


def run_pack(
    out_dir: Path,
    *,
    sft_train_shards: int,
    sft_per_train_shard: int,
    sft_val: int,
    sft_test: int,
    sft_offset: int,
    dpo_train_shards: int,
    dpo_per_train_shard: int,
    dpo_val: int,
    dpo_test: int,
    dpo_offset: int,
    xz_preset: int,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jsonl.xz"):
        old.unlink()

    sft_tasks = []
    for s in range(sft_train_shards):
        fp = out_dir / f"sft_train-{s:05d}-of-{sft_train_shards:05d}.jsonl.xz"
        sft_tasks.append(("train", str(fp), sft_offset + s * sft_per_train_shard, sft_per_train_shard, seed, xz_preset))
    sft_tasks.append(("validation", str(out_dir / "sft_validation.jsonl.xz"), 0, sft_val, seed, xz_preset))
    sft_tasks.append(("test", str(out_dir / "sft_test.jsonl.xz"), 0, sft_test, seed, xz_preset))

    dpo_tasks = []
    for s in range(dpo_train_shards):
        fp = out_dir / f"dpo_train-{s:05d}-of-{dpo_train_shards:05d}.jsonl.xz"
        dpo_tasks.append(("dpo_train", str(fp), dpo_offset + s * dpo_per_train_shard, dpo_per_train_shard, seed, xz_preset))
    cur_dpo = dpo_offset + dpo_train_shards * dpo_per_train_shard
    dpo_tasks.append(("dpo_validation", str(out_dir / "dpo_validation.jsonl.xz"), cur_dpo, dpo_val, seed, xz_preset))
    dpo_tasks.append(("dpo_test", str(out_dir / "dpo_test.jsonl.xz"), cur_dpo + dpo_val, dpo_test, seed, xz_preset))

    files: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(_write_sft_shard, sft_tasks):
            files.append(res)
        for res in pool.map(_write_dpo_shard, dpo_tasks):
            files.append(res)

    sft_rec = sum(f["records"] for f in files if f["kind"] == "agent_sft")
    dpo_rec = sum(f["records"] for f in files if f["kind"] == "agent_dpo")
    tot_raw = sum(f["uncompressed_bytes"] for f in files)
    tot_xz = sum(f["compressed_bytes"] for f in files)
    elapsed = round(time.time() - t0, 2)

    stats = {
        "suite_name": "Autonomous Agent Ultra Fine-Tuning Suite (SFT + DPO)",
        "version": "5.0.0",
        "addressable_agent_sft_universe": AGENT_SFT_TOTAL,
        "addressable_agent_dpo_universe": AGENT_DPO_TOTAL,
        "num_agent_categories": NUM_AGENT_CATEGORIES,
        "agent_categories": [c for _, c, _ in AGENT_GENERATORS],
        "dpo_failure_taxonomies": list(DPO_TAXONOMIES),
        "materialized_summary": {
            "output_dir": str(out_dir),
            "total_records": sft_rec + dpo_rec,
            "agent_sft_records": sft_rec,
            "agent_dpo_pairs": dpo_rec,
            "uncompressed_bytes": tot_raw,
            "uncompressed_gb": round(tot_raw / 1_000_000_000, 3),
            "compressed_bytes": tot_xz,
            "compressed_mib": round(tot_xz / (1024 * 1024), 2),
            "compression_ratio": round(tot_raw / max(1, tot_xz), 2),
            "seconds": elapsed,
        },
        "files": files,
    }
    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build persisted in-repo + massive local-disk Agent Fine-Tuning packs.")
    parser.add_argument("--mode", choices=["in-repo", "local-massive", "both"], default="both")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _write_agent_showcase(ROOT / "samples" / "agent_ultra_showcase.md", seed=args.seed)

    results = {}
    if args.mode in ("in-repo", "both"):
        # ~450,000 Agent SFT + 200,000 Agent DPO = 650,000 persisted records (~46-49 MiB .jsonl.xz)
        st_repo = run_pack(
            ROOT / "data" / "agent_ultra",
            sft_train_shards=5,
            sft_per_train_shard=80_000,
            sft_val=25_000,
            sft_test=25_000,
            sft_offset=0,
            dpo_train_shards=2,
            dpo_per_train_shard=90_000,
            dpo_val=10_000,
            dpo_test=10_000,
            dpo_offset=0,
            xz_preset=3,
            workers=args.workers,
            seed=args.seed,
        )
        results["in_repo_persisted"] = st_repo["materialized_summary"]

    if args.mode in ("local-massive", "both"):
        # Disjoint 3,000,000 Pure-Agent Ultra pack in out/agent-ultra-pack/ (2.5M SFT + 500k DPO)
        st_out = run_pack(
            ROOT / "out" / "agent-ultra-pack",
            sft_train_shards=12,
            sft_per_train_shard=200_000,
            sft_val=50_000,
            sft_test=50_000,
            sft_offset=500_000,
            dpo_train_shards=4,
            dpo_per_train_shard=115_000,
            dpo_val=20_000,
            dpo_test=20_000,
            dpo_offset=250_000,
            xz_preset=2,
            workers=args.workers,
            seed=args.seed,
        )
        (ROOT / "data" / "manifest_agent_ultra_3m.json").write_text(json.dumps(st_out, indent=2) + "\n", encoding="utf-8")
        results["local_disk_massive"] = st_out["materialized_summary"]

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
