#!/usr/bin/env python3
"""Independent zero-trust quality & cryptographic audit for 1.25M and 10,000,000+ (.jsonl.xz) datasets."""

from __future__ import annotations

import argparse
from array import array
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training_data.validator import normalize_text, validate_record  # noqa: E402


def _audit_shard_worker(args: tuple[str, str]) -> dict[str, object]:
    """Stream-decompress and validate every record in a single shard, writing 64-bit prompt hashes to tmp."""
    fpath_s, tmp_dir_s = args
    fpath = Path(fpath_s)
    tmp_dir = Path(tmp_dir_s)

    h_sha = hashlib.sha256()
    with fpath.open("rb") as bfh:
        while chunk := bfh.read(1 << 20):
            h_sha.update(chunk)
    actual_sha = h_sha.hexdigest()

    u_arr = array("Q")
    count = 0
    errors: list[str] = []

    if fpath.name.endswith(".xz"):
        proc = subprocess.Popen(["xz", "-dc", str(fpath)], stdout=subprocess.PIPE)
        assert proc.stdout is not None
        for line_no, raw_line in enumerate(proc.stdout, start=1):
            rec = json.loads(raw_line)
            count += 1
            ok, reason = validate_record(rec)
            if not ok:
                errors.append(f"{fpath.name}:{line_no}: {reason}")
                if len(errors) >= 10:
                    break
            msgs = rec["messages"]
            u_txt = msgs[1]["content"] if len(msgs) == 3 else " || ".join(
                m["content"] for m in msgs if m["role"] == "user"
            )
            uh = int.from_bytes(hashlib.sha256(normalize_text(u_txt).encode("utf-8")).digest()[:8], "little")
            u_arr.append(uh)
        proc.stdout.close()
        ret = proc.wait()
        if ret != 0 and not errors:
            errors.append(f"{fpath.name}: xz exited with {ret}")
    else:
        opener = gzip.open if fpath.name.endswith(".gz") else open
        with opener(fpath, "rt", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                rec = json.loads(raw_line)
                count += 1
                ok, reason = validate_record(rec)
                if not ok:
                    errors.append(f"{fpath.name}:{line_no}: {reason}")
                    if len(errors) >= 10:
                        break
                msgs = rec["messages"]
                u_txt = msgs[1]["content"] if len(msgs) == 3 else " || ".join(
                    m["content"] for m in msgs if m["role"] == "user"
                )
                uh = int.from_bytes(hashlib.sha256(normalize_text(u_txt).encode("utf-8")).digest()[:8], "little")
                u_arr.append(uh)

    bin_path = tmp_dir / f"{fpath.name}.uhash"
    with bin_path.open("wb") as out_b:
        u_arr.tofile(out_b)
    del u_arr

    return {
        "file": fpath.name,
        "sha256": actual_sha,
        "records": count,
        "errors": errors,
        "hash_file": str(bin_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit .jsonl.xz / .jsonl.gz shards for integrity and uniqueness.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data", help="Directory containing shards + dataset_stats.json.")
    parser.add_argument("--workers", type=int, default=2, help="Parallel audit workers.")
    args = parser.parse_args()

    t0 = time.time()
    stats_path = args.data_dir / "dataset_stats.json"
    if not stats_path.exists():
        raise SystemExit(f"Missing stats file: {stats_path}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    shard_files = sorted(
        list(args.data_dir.glob("*.jsonl.xz"))
        + list(args.data_dir.glob("*.jsonl.gz"))
    )
    if not shard_files:
        raise SystemExit(f"No compressed shard files found in {args.data_dir}")

    expected_sha_map = {entry["file"]: entry["sha256"] for entry in stats.get("files", [])}
    expected_cnt_map = {entry["file"]: entry["records"] for entry in stats.get("files", [])}

    with tempfile.TemporaryDirectory(prefix="audit_10m_") as tmp_s:
        tasks = [(str(fp), tmp_s) for fp in shard_files]
        total_checked = 0
        all_errors: list[str] = []
        all_hashes = array("Q")

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for res in pool.map(_audit_shard_worker, tasks):
                fname = str(res["file"])
                cnt = int(res["records"])  # type: ignore[arg-type]
                sha = str(res["sha256"])
                errs = list(res["errors"])  # type: ignore[arg-type]
                total_checked += cnt
                all_errors.extend(errs)

                if fname in expected_sha_map and sha != expected_sha_map[fname]:
                    all_errors.append(f"SHA256 mismatch on {fname}: {sha} != {expected_sha_map[fname]}")
                if fname in expected_cnt_map and cnt != expected_cnt_map[fname]:
                    all_errors.append(f"Record count mismatch on {fname}: {cnt} != {expected_cnt_map[fname]}")

                hp = Path(str(res["hash_file"]))
                part = array("Q")
                part.frombytes(hp.read_bytes())
                hp.unlink(missing_ok=True)
                all_hashes.extend(part)
                del part

        duplicate_prompts = 0
        for nib in range(16):
            sub = sorted(h for h in all_hashes if (h >> 60) == nib)
            for i in range(1, len(sub)):
                if sub[i] == sub[i - 1]:
                    duplicate_prompts += 1
            del sub
        del all_hashes

    if total_checked != stats["total_records"]:
        all_errors.append(f"Total record mismatch: {total_checked} != {stats['total_records']}")

    elapsed = round(time.time() - t0, 2)
    report = {
        "audit_status": "PASSED" if (not all_errors and duplicate_prompts == 0) else "FAILED",
        "data_dir": str(args.data_dir),
        "total_records_verified": total_checked,
        "shards_verified": len(shard_files),
        "duplicate_user_prompts": duplicate_prompts,
        "schema_or_code_errors": len(all_errors),
        "audit_seconds": elapsed,
        "sample_errors": all_errors[:10],
    }
    print(json.dumps(report, indent=2))
    if report["audit_status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
