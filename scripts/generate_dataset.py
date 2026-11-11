#!/usr/bin/env python3
"""CLI entrypoint to synthesize 1.25M (in-repo) or 10,000,000 (10M full scale) solver-verified SFT examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training_data.pipeline import build_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize solver-verified Coding, Security & Agent SFT dataset.")
    parser.add_argument(
        "--scale",
        choices=["10m", "1.25m", "custom"],
        default="10m",
        help="Preset scale: '10m' (10,000,000 records in out/dataset-10m) or '1.25m' (1,250,000 in data/).",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory for .jsonl.xz shards.")
    parser.add_argument("--samples-dir", type=Path, default=ROOT / "samples", help="Output directory for plain-text previews.")
    parser.add_argument("--per-category", type=int, default=200_000, help="Unique examples per category (50 categories).")
    parser.add_argument("--val-per-category", type=int, default=5_000, help="Validation examples per category.")
    parser.add_argument("--test-per-category", type=int, default=5_000, help="Test examples per category.")
    parser.add_argument("--train-shards", type=int, default=20, help="Number of .jsonl.xz shards for train split.")
    parser.add_argument("--workers", type=int, default=2, help="Parallel worker processes.")
    parser.add_argument("--xz-preset", type=int, default=2, help="XZ LZMA2 preset (1..6).")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")
    args = parser.parse_args()

    if args.scale == "10m":
        out_dir = args.output_dir or (ROOT / "out" / "dataset-10m")
        per_cat, val_cat, test_cat, n_shards, preset = 200_000, 5_000, 5_000, 20, 2
        mirror = ROOT / "data" / "manifest_10m.json"
    elif args.scale == "1.25m":
        out_dir = args.output_dir or (ROOT / "data")
        per_cat, val_cat, test_cat, n_shards, preset = 25_000, 1_000, 1_000, 10, 3
        mirror = None
    else:
        out_dir = args.output_dir or (ROOT / "out" / "custom")
        per_cat, val_cat, test_cat, n_shards, preset = (
            args.per_category,
            args.val_per_category,
            args.test_per_category,
            args.train_shards,
            args.xz_preset,
        )
        mirror = None

    stats = build_dataset(
        output_dir=out_dir,
        samples_dir=args.samples_dir,
        per_category=per_cat,
        val_per_category=val_cat,
        test_per_category=test_cat,
        train_shards=n_shards,
        seed=args.seed,
        workers=args.workers,
        xz_preset=preset,
        stats_mirror_path=mirror,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(out_dir),
                "total_records": stats["total_records"],
                "num_categories": stats["num_categories"],
                "splits": stats["splits"],
                "storage": stats["storage"],
                "generation_seconds": stats["quality_assurance"]["generation_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
