#!/usr/bin/env python3
"""Zero-Trust Auditor for Agent Ultra SFT & DPO Packs (`data/agent_ultra/` and `out/agent-ultra-pack/`)."""

from __future__ import annotations

import argparse
from array import array
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training_data.agent_dpo import validate_dpo_pair  # noqa: E402
from training_data.validator import normalize_text, validate_record  # noqa: E402


def _audit_one_file(fpath_s: str) -> dict[str, object]:
    fpath = Path(fpath_s)
    is_dpo = fpath.name.startswith("dpo_")

    h_sha = hashlib.sha256()
    with fpath.open("rb") as bfh:
        while chunk := bfh.read(1 << 20):
            h_sha.update(chunk)

    u_hashes = array("Q")
    count = 0
    errors: list[str] = []

    proc = subprocess.Popen(["xz", "-dc", str(fpath)], stdout=subprocess.PIPE)
    assert proc.stdout is not None
    for line_no, raw_line in enumerate(proc.stdout, start=1):
        rec = json.loads(raw_line)
        count += 1
        if is_dpo:
            ok, reason = validate_dpo_pair(rec)
            key_txt = " || ".join(m["content"] for m in rec["prompt"] if m["role"] == "user")
        else:
            ok, reason = validate_record(rec)
            key_txt = " || ".join(m["content"] for m in rec["messages"] if m["role"] == "user")
        if not ok:
            errors.append(f"{fpath.name}:{line_no}: {reason}")
            if len(errors) >= 10:
                break
        uh = int.from_bytes(hashlib.sha256(normalize_text(key_txt).encode("utf-8")).digest()[:8], "little")
        u_hashes.append(uh)
    proc.stdout.close()
    ret = proc.wait()
    if ret != 0 and not errors:
        errors.append(f"{fpath.name}: xz exited {ret}")

    return {
        "file": fpath.name,
        "kind": "dpo" if is_dpo else "sft",
        "sha256": h_sha.hexdigest(),
        "records": count,
        "errors": errors,
        "u_hashes": u_hashes.tobytes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Agent Ultra SFT + DPO .jsonl.xz shards.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "agent_ultra")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    t0 = time.time()
    stats = json.loads((args.data_dir / "dataset_stats.json").read_text(encoding="utf-8"))
    exp_sha = {f["file"]: f["sha256"] for f in stats["files"]}
    exp_cnt = {f["file"]: f["records"] for f in stats["files"]}

    shards = sorted(args.data_dir.glob("*.jsonl.xz"))
    errors: list[str] = []
    sft_hashes = array("Q")
    dpo_hashes = array("Q")
    sft_count = 0
    dpo_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(_audit_one_file, [str(p) for p in shards]):
            fname = str(res["file"])
            cnt = int(res["records"])  # type: ignore[arg-type]
            sha = str(res["sha256"])
            errs = list(res["errors"])  # type: ignore[arg-type]
            errors.extend(errs)
            if sha != exp_sha.get(fname):
                errors.append(f"SHA256 mismatch: {fname}")
            if cnt != exp_cnt.get(fname):
                errors.append(f"Count mismatch: {fname}")
            part = array("Q")
            part.frombytes(res["u_hashes"])  # type: ignore[arg-type]
            if res["kind"] == "dpo":
                dpo_count += cnt
                dpo_hashes.extend(part)
            else:
                sft_count += cnt
                sft_hashes.extend(part)

    sft_dups = 0
    dpo_dups = 0
    for arr_ref, is_d in ((sft_hashes, False), (dpo_hashes, True)):
        for nib in range(16):
            sub = sorted(h for h in arr_ref if (h >> 60) == nib)
            for i in range(1, len(sub)):
                if sub[i] == sub[i - 1]:
                    if is_d:
                        dpo_dups += 1
                    else:
                        sft_dups += 1
            del sub

    passed = (not errors) and sft_dups == 0 and dpo_dups == 0
    report = {
        "audit_status": "PASSED" if passed else "FAILED",
        "data_dir": str(args.data_dir),
        "sft_records_verified": sft_count,
        "dpo_pairs_verified": dpo_count,
        "total_verified": sft_count + dpo_count,
        "sft_duplicate_prompts": sft_dups,
        "dpo_duplicate_prompts": dpo_dups,
        "errors": len(errors),
        "audit_seconds": round(time.time() - t0, 2),
        "sample_errors": errors[:10],
    }
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
