#!/usr/bin/env python3
"""Export/convert ChatML .jsonl.xz (or .jsonl.gz) shards into plain .jsonl, OpenAI fine-tune JSONL, Alpaca, or ShareGPT."""

from __future__ import annotations

import argparse
import gzip
import json
import lzma
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

ROLE_TO_SHAREGPT = {
    "system": "system",
    "user": "human",
    "assistant": "gpt",
    "tool": "observation",
}


def convert_record(rec: dict[str, Any], fmt: str) -> dict[str, Any]:
    if fmt == "plain_jsonl":
        return rec
    if fmt == "openai_ft":
        msgs = []
        for m in rec["messages"]:
            if m["role"] == "tool":
                msgs.append({"role": "user", "content": f"<tool_response>\n{m['content']}\n</tool_response>"})
            else:
                msgs.append(m)
        return {"messages": msgs}
    if fmt == "sharegpt":
        return {
            "id": rec["id"],
            "domain": rec["domain"],
            "category": rec["category"],
            "difficulty": rec["difficulty"],
            "conversations": [{"from": ROLE_TO_SHAREGPT[m["role"]], "value": m["content"]} for m in rec["messages"]],
        }
    if fmt == "alpaca":
        sys_txt = rec["messages"][0]["content"]
        user_txt = rec["messages"][1]["content"]
        out_txt = rec["messages"][-1]["content"]
        middle = rec["messages"][2:-1]
        ctx = "\n".join(f"[{m['role']}] {m['content']}" for m in middle)
        return {
            "instruction": user_txt,
            "input": ctx if ctx else sys_txt,
            "output": out_txt,
            "domain": rec["domain"],
            "category": rec["category"],
            "difficulty": rec["difficulty"],
        }
    raise ValueError(f"Unsupported format: {fmt}")


def _open_compressed(path: Path):
    if path.name.endswith(".xz"):
        return lzma.open(path, "rt", encoding="utf-8")
    return gzip.open(path, "rt", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert dataset shards to plain JSONL, OpenAI FT, Alpaca, or ShareGPT.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--format", choices=["plain_jsonl", "openai_ft", "alpaca", "sharegpt"], default="plain_jsonl")
    parser.add_argument("--limit-per-shard", type=int, default=0, help="Optional cap per shard (0 = all).")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(list(args.data_dir.glob("*.jsonl.xz")) + list(args.data_dir.glob("*.jsonl.gz")))
    total = 0
    for shard_path in shards:
        out_name = shard_path.name[:-3]  # strip .xz or .gz -> .jsonl
        out_path = args.out_dir / out_name
        written = 0
        with _open_compressed(shard_path) as src, out_path.open("w", encoding="utf-8") as dst:
            for line in src:
                rec = json.loads(line)
                converted = convert_record(rec, args.format)
                dst.write(json.dumps(converted, separators=(",", ":"), ensure_ascii=False) + "\n")
                total += 1
                written += 1
                if args.limit_per_shard and written >= args.limit_per_shard:
                    break
    print(f"Exported {total} records in '{args.format}' format to {args.out_dir}")


if __name__ == "__main__":
    main()
