"""Zero-disk deterministic streaming generator for 10,000,000+ verified ChatML training examples."""

from __future__ import annotations

from collections.abc import Iterator
import hashlib
import random
from typing import Any

from training_data.generators import ALL_GENERATORS
from training_data.validator import normalize_text, validate_record


def iter_streaming_dataset(
    *,
    per_category: int = 200_000,
    split: str = "train",
    val_per_category: int = 5_000,
    test_per_category: int = 5_000,
    seed: int = 42,
    verify: bool = True,
) -> Iterator[dict[str, Any]]:
    """
    Deterministically stream up to 10,000,000+ solver-verified ChatML records in O(category) RAM
    without requiring 14+ GB of uncompressed disk space. Compatible with Hugging Face
    `Dataset.from_generator` and PyTorch `IterableDataset`.
    """
    if split not in ("train", "validation", "test", "all"):
        raise ValueError(f"Invalid split: {split}")

    train_per_cat = per_category - val_per_category - test_per_category
    if split == "train":
        lo, hi = 0, train_per_cat
    elif split == "validation":
        lo, hi = train_per_cat, train_per_cat + val_per_category
    elif split == "test":
        lo, hi = train_per_cat + val_per_category, per_category
    else:
        lo, hi = 0, per_category

    num_cats = len(ALL_GENERATORS)
    rngs = [random.Random(seed * 100_003 + c) for c in range(num_cats)]
    seen_per_cat: list[set[int]] = [set() for _ in range(num_cats)]
    attempts = [0] * num_cats
    produced = [0] * num_cats

    # Interleave across all 50 categories in round-robin order for balanced training batches
    for target_idx in range(hi):
        for c_idx, (_, _, gen_fn) in enumerate(ALL_GENERATORS):
            while True:
                rec = gen_fn(rngs[c_idx], attempts[c_idx])
                attempts[c_idx] += 1
                if verify:
                    ok, reason = validate_record(rec)
                    if not ok:
                        raise ValueError(f"Validation failed in category {c_idx}: {reason}")
                user_concat = " || ".join(m["content"] for m in rec["messages"] if m["role"] == "user")
                h64 = int.from_bytes(hashlib.sha256(normalize_text(user_concat).encode("utf-8")).digest()[:8], "little")
                if h64 in seen_per_cat[c_idx]:
                    continue
                seen_per_cat[c_idx].add(h64)
                curr_pos = produced[c_idx]
                produced[c_idx] += 1
                if curr_pos >= lo:
                    yield rec
                break
