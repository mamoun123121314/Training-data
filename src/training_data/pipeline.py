"""10-Million-scale parallel synthesis, validation, global deduplication, and streaming LZMA2 (.jsonl.xz) pipeline."""

from __future__ import annotations

from array import array
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time
from typing import Any
import zlib

from training_data.generators import ALL_GENERATORS
from training_data.validator import normalize_text, validate_record


def _hash64_pair(record: dict[str, Any]) -> tuple[int, int]:
    """Compute 64-bit SHA-256 prefixes for normalized user prompt and full conversation."""
    msgs = record["messages"]
    if len(msgs) == 3:
        u_txt = msgs[1]["content"]
        c_txt = f"user:{u_txt} || assistant:{msgs[2]['content']}"
    else:
        u_txt = " || ".join(m["content"] for m in msgs if m["role"] == "user")
        c_txt = " || ".join(f"{m['role']}:{m['content']}" for m in msgs[1:])
    uh = int.from_bytes(hashlib.sha256(normalize_text(u_txt).encode("utf-8")).digest()[:8], "little")
    ch = int.from_bytes(hashlib.sha256(normalize_text(c_txt).encode("utf-8")).digest()[:8], "little")
    return uh, ch


def _generate_category_worker(args: tuple[int, int, int, int, int, int, str]) -> dict[str, Any]:
    """
    Worker process: deterministically generate, validate, locally deduplicate `per_category` records,
    and write pre-partitioned zlib-compressed shard slices directly to `bucket_dir`.
    """
    cat_idx, per_category, val_per_cat, test_per_cat, train_shards, base_seed, bucket_dir_s = args
    expected_domain, expected_cat, gen_fn = ALL_GENERATORS[cat_idx]
    rng = random.Random(base_seed * 100_003 + cat_idx)
    bucket_dir = Path(bucket_dir_s)

    seen_u: set[int] = set()
    seen_c: set[int] = set()
    u_arr = array("Q")

    lines: list[bytes] = []
    preview_objs: list[dict[str, Any]] = []
    diff_counts: Counter[str] = Counter()
    style_counts: Counter[str] = Counter()
    qual_counts: Counter[str] = Counter()
    turns_counts: Counter[str] = Counter()
    collisions = 0

    attempt = 0
    max_attempts = per_category * 6
    while len(lines) < per_category:
        if attempt >= max_attempts:
            raise RuntimeError(f"Exceeded max attempts ({max_attempts}) for {expected_cat} at {len(lines)}")
        rec = gen_fn(rng, attempt)
        attempt += 1

        ok, reason = validate_record(rec)
        if not ok:
            raise ValueError(f"Validation failed in {expected_cat} (attempt {attempt}): {reason}")
        if rec["domain"] != expected_domain or rec["category"] != expected_cat:
            raise ValueError(f"Metadata mismatch in {expected_cat}")

        uh, ch = _hash64_pair(rec)
        if uh in seen_u or ch in seen_c:
            collisions += 1
            continue
        seen_u.add(uh)
        seen_c.add(ch)
        u_arr.append(uh)

        encoded = (json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
        lines.append(encoded)
        if len(preview_objs) < 30:
            preview_objs.append(rec)

        diff_counts[rec["difficulty"]] += 1
        style_counts[rec["style"]] += 1
        qual_counts[rec["quality_check"]] += 1
        turns_counts[str(len(rec["messages"]))] += 1

    # Free hash sets immediately before shuffling & slicing
    del seen_u
    del seen_c

    rng.shuffle(lines)
    train_per_cat = per_category - val_per_cat - test_per_cat
    per_shard = train_per_cat // train_shards

    for s in range(train_shards):
        st = s * per_shard
        en = (s + 1) * per_shard
        comp = zlib.compress(b"".join(lines[st:en]), level=1)
        (bucket_dir / f"tr_s{s:02d}_c{cat_idx:02d}.zlib").write_bytes(comp)

    v_st = train_per_cat
    v_en = train_per_cat + val_per_cat
    (bucket_dir / f"val_c{cat_idx:02d}.zlib").write_bytes(zlib.compress(b"".join(lines[v_st:v_en]), level=1))
    (bucket_dir / f"tst_c{cat_idx:02d}.zlib").write_bytes(zlib.compress(b"".join(lines[v_en:]), level=1))
    del lines

    hash_path = bucket_dir / f"uhash_c{cat_idx:02d}.bin"
    with hash_path.open("wb") as hfh:
        u_arr.tofile(hfh)
    del u_arr

    return {
        "cat_idx": cat_idx,
        "domain": expected_domain,
        "category": expected_cat,
        "count": per_category,
        "collisions": collisions,
        "preview": preview_objs,
        "diff_counts": dict(diff_counts),
        "style_counts": dict(style_counts),
        "qual_counts": dict(qual_counts),
        "turns_counts": dict(turns_counts),
    }


def _stream_compress_shard(
    out_path: Path, slice_paths: list[Path], rng: random.Random, xz_preset: int = 2
) -> tuple[int, int, int, str]:
    """
    Stream category slices in randomized order directly into `xz -T2` stdin with O(1) slice RAM (~15 MB peak).
    Returns (record_count, raw_bytes, compressed_bytes, sha256_hex).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    order = list(range(len(slice_paths)))
    rng.shuffle(order)

    raw_bytes = 0
    rec_count = 0
    with out_path.open("wb") as out_fh:
        proc = subprocess.Popen(
            ["xz", f"--lzma2=preset={xz_preset},dict=32MiB", "-T2", "--check=crc64"],
            stdin=subprocess.PIPE,
            stdout=out_fh,
        )
        assert proc.stdin is not None
        try:
            for idx in order:
                sp = slice_paths[idx]
                raw_chunk = zlib.decompress(sp.read_bytes())
                sp.unlink(missing_ok=True)
                raw_bytes += len(raw_chunk)
                rec_count += raw_chunk.count(b"\n")
                proc.stdin.write(raw_chunk)
                del raw_chunk
        finally:
            proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"xz compression failed on {out_path} with exit code {ret}")

    h = hashlib.sha256()
    comp_bytes = 0
    with out_path.open("rb") as rfh:
        while chunk := rfh.read(1 << 20):
            comp_bytes += len(chunk)
            h.update(chunk)
    return rec_count, raw_bytes, comp_bytes, h.hexdigest()


def _verify_global_hashes(bucket_dir: Path, num_cats: int, total_expected: int) -> int:
    """Verify cross-category uniqueness across all 10,000,000+ user prompt 64-bit hashes in ~80 MB RAM."""
    all_hashes = array("Q")
    for c in range(num_cats):
        hp = bucket_dir / f"uhash_c{c:02d}.bin"
        raw = hp.read_bytes()
        hp.unlink(missing_ok=True)
        part = array("Q")
        part.frombytes(raw)
        all_hashes.extend(part)
        del part, raw

    if len(all_hashes) != total_expected:
        raise RuntimeError(f"Hash count mismatch: {len(all_hashes)} != {total_expected}")

    # Check cross-category uniqueness via compact C-level sort on memoryview/list chunks or bucketed check
    # Partition into 16 radix buckets by top 4 bits so peak sort memory stays < 140 MB!
    cross_collisions = 0
    for nib in range(16):
        sub = sorted(h for h in all_hashes if (h >> 60) == nib)
        for i in range(1, len(sub)):
            if sub[i] == sub[i - 1]:
                cross_collisions += 1
        del sub
    del all_hashes
    return cross_collisions


def _write_plain_jsonl(path: Path, records: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            line = json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n"
            total += len(line.encode("utf-8"))
            fh.write(line)
    return total


def _write_showcase_markdown(path: Path, showcase_items: list[dict[str, Any]], total_records: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Showcase Training Examples (Curated Sample Across All 3 Pillars)",
        "",
        f"Below are representative records drawn directly from the generated **{total_records:,}-example** ChatML dataset,",
        "spanning **Coding & Software Engineering**, **Ethical Security Engineering**, and **Autonomous Agent Use**.",
        "",
    ]
    for idx, rec in enumerate(showcase_items, start=1):
        lines.append(f"## {idx}. `{rec['category']}` (`{rec['domain']}` | difficulty: `{rec['difficulty']}` | style: `{rec['style']}`)")
        lines.append(f"- **Record ID**: `{rec['id']}` | **Quality Gate**: `{rec['quality_check']}` | **Tags**: `{', '.join(rec['tags'])}`")
        lines.append("")
        for msg in rec["messages"]:
            role = msg["role"].upper()
            lines.append(f"**[{role}]**")
            if msg["role"] == "tool":
                lines.append(f"```json\n{msg['content']}\n```")
            else:
                lines.append(msg["content"])
            lines.append("")
        lines.append("---")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_dataset(
    output_dir: Path,
    samples_dir: Path | None = None,
    *,
    per_category: int = 200_000,
    val_per_category: int = 5_000,
    test_per_category: int = 5_000,
    train_shards: int = 20,
    seed: int = 42,
    workers: int = 2,
    xz_preset: int = 2,
    stats_mirror_path: Path | None = None,
) -> dict[str, Any]:
    """Generate, validate, deduplicate, and stream-compress up to 10,000,000+ records."""
    if val_per_category + test_per_category >= per_category:
        raise ValueError("val_per_category + test_per_category must be strictly less than per_category")

    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in list(output_dir.glob("*.jsonl.gz")) + list(output_dir.glob("*.jsonl.xz")):
        old_file.unlink()

    num_cats = len(ALL_GENERATORS)
    train_per_cat = per_category - val_per_category - test_per_category
    if train_per_cat % train_shards != 0:
        raise ValueError(f"train_per_cat ({train_per_cat}) must be divisible by train_shards ({train_shards})")

    bucket_dir = Path(tempfile.mkdtemp(prefix="td_10m_"))
    try:
        tasks = [
            (cat_idx, per_category, val_per_category, test_per_category, train_shards, seed, str(bucket_dir))
            for cat_idx in range(num_cats)
        ]
        results: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for res in pool.map(_generate_category_worker, tasks):
                results.append(res)

        results.sort(key=lambda r: r["cat_idx"])

        domain_counts: Counter[str] = Counter()
        category_counts: Counter[str] = Counter()
        difficulty_counts: Counter[str] = Counter()
        style_counts: Counter[str] = Counter()
        quality_counts: Counter[str] = Counter()
        turns_counts: Counter[str] = Counter()
        collisions_skipped = 0
        preview_pool: list[dict[str, Any]] = []
        showcase_pool: list[dict[str, Any]] = []

        showcase_categories = {
            "algo_data_structures",
            "sql_query_and_optimization",
            "go_concurrency_and_services",
            "multi_turn_code_review",
            "secure_code_remediation_web",
            "cvss_triage_and_vuln_reporting",
            "log_forensics_incident_response",
            "network_acl_firewall_analysis",
            "code_repo_navigation_agent",
            "parallel_multi_tool_fanout",
            "prompt_injection_resistant_agent",
            "guardrail_policy_check_agent",
        }

        for res in results:
            domain_counts[res["domain"]] += res["count"]
            category_counts[res["category"]] += res["count"]
            difficulty_counts.update(res["diff_counts"])
            style_counts.update(res["style_counts"])
            quality_counts.update(res["qual_counts"])
            turns_counts.update(res["turns_counts"])
            collisions_skipped += res["collisions"]
            preview_pool.extend(res["preview"])
            if res["category"] in showcase_categories and res["preview"]:
                showcase_pool.append(res["preview"][0])

        total_expected = per_category * num_cats
        cross_collisions = _verify_global_hashes(bucket_dir, num_cats, total_expected)
        if cross_collisions != 0:
            raise RuntimeError(f"Detected {cross_collisions} unexpected cross-category hash collisions!")

        rng = random.Random(seed)
        shard_files: list[dict[str, Any]] = []
        total_raw_bytes = 0
        total_xz_bytes = 0
        total_train_records = 0

        for s in range(train_shards):
            s_paths = [bucket_dir / f"tr_s{s:02d}_c{c:02d}.zlib" for c in range(num_cats)]
            shard_path = output_dir / f"train-{s:05d}-of-{train_shards:05d}.jsonl.xz"
            cnt, raw_b, xz_b, sha = _stream_compress_shard(shard_path, s_paths, rng, xz_preset=xz_preset)
            total_train_records += cnt
            total_raw_bytes += raw_b
            total_xz_bytes += xz_b
            shard_files.append(
                {
                    "split": "train",
                    "file": shard_path.name,
                    "records": cnt,
                    "uncompressed_bytes": raw_b,
                    "compressed_bytes": xz_b,
                    "sha256": sha,
                }
            )

        for split_name, prefix in (("validation", "val"), ("test", "tst")):
            s_paths = [bucket_dir / f"{prefix}_c{c:02d}.zlib" for c in range(num_cats)]
            split_path = output_dir / f"{split_name}.jsonl.xz"
            cnt, raw_b, xz_b, sha = _stream_compress_shard(split_path, s_paths, rng, xz_preset=xz_preset)
            total_raw_bytes += raw_b
            total_xz_bytes += xz_b
            shard_files.append(
                {
                    "split": split_name,
                    "file": split_path.name,
                    "records": cnt,
                    "uncompressed_bytes": raw_b,
                    "compressed_bytes": xz_b,
                    "sha256": sha,
                }
            )

        preview_bytes = 0
        if samples_dir is not None:
            samples_dir.mkdir(parents=True, exist_ok=True)
            for old_prev in samples_dir.glob("preview_*.jsonl"):
                old_prev.unlink()
            rng.shuffle(preview_pool)
            preview_path = samples_dir / f"preview_{len(preview_pool)}.jsonl"
            preview_bytes = _write_plain_jsonl(preview_path, preview_pool)
            _write_showcase_markdown(samples_dir / "showcase_examples.md", showcase_pool, total_expected)

        elapsed = round(time.time() - t0, 2)
        stats: dict[str, Any] = {
            "dataset_name": "Training-data: 10-Million-Scale Coding, Ethical Security & Autonomous Agent SFT Suite",
            "version": "3.0.0",
            "language": "en",
            "format": "ChatML / OpenAI Messages (.jsonl.xz LZMA2 shards + streaming API)",
            "seed": seed,
            "total_records": total_expected,
            "num_categories": num_cats,
            "splits": {
                "train": total_train_records,
                "validation": val_per_category * num_cats,
                "test": test_per_category * num_cats,
            },
            "storage": {
                "total_uncompressed_bytes": total_raw_bytes,
                "total_uncompressed_mb": round(total_raw_bytes / 1_000_000, 2),
                "total_uncompressed_gb": round(total_raw_bytes / 1_000_000_000, 3),
                "total_compressed_bytes": total_xz_bytes,
                "total_compressed_mb": round(total_xz_bytes / 1_000_000, 2),
                "compression_ratio": round(total_raw_bytes / max(1, total_xz_bytes), 2),
                "preview_plain_jsonl_bytes": preview_bytes,
            },
            "quality_assurance": {
                "validation_pass_rate": 1.0,
                "unique_user_prompts_rate": 1.0,
                "unique_conversations_rate": 1.0,
                "cross_category_collisions": cross_collisions,
                "duplicates_rejected_during_synthesis": collisions_skipped,
                "generation_seconds": elapsed,
            },
            "domain_distribution": dict(domain_counts),
            "difficulty_distribution": dict(difficulty_counts),
            "style_distribution": dict(style_counts),
            "quality_check_distribution": dict(quality_counts),
            "message_count_distribution": dict(sorted(turns_counts.items(), key=lambda kv: int(kv[0]))),
            "category_distribution": dict(category_counts),
            "files": shard_files,
        }

        stats_json = json.dumps(stats, indent=2) + "\n"
        (output_dir / "dataset_stats.json").write_text(stats_json, encoding="utf-8")
        if stats_mirror_path is not None:
            stats_mirror_path.parent.mkdir(parents=True, exist_ok=True)
            stats_mirror_path.write_text(stats_json, encoding="utf-8")
        return stats
    finally:
        shutil.rmtree(bucket_dir, ignore_errors=True)
