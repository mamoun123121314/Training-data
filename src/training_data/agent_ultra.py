"""
Pure Autonomous Agent Training & Fine-Tuning Engine (`AgentUltraDataset` & `AgentDPODataset`).

Covers all 42 specialized Autonomous Agent categories (`1,050,000,000` O(1)-addressable Agent SFT trajectories)
plus 8 Agent DPO / ORPO failure-mode taxonomies (`100,000,000` O(1)-addressable preference pairs).
"""

from __future__ import annotations

from collections.abc import Iterator
import hashlib
import random
import struct
from typing import Any

from training_data.agent_dpo import DPO_TAXONOMIES, synthesize_agent_dpo_pair
from training_data.generators import AGENT_GENERATORS
from training_data.ultra_billion import (
    AGENT_PROFILES,
    NATURAL_CONTEXT_OPENERS,
    _build_agent_thought,
    feistel_permute,
)

NUM_AGENT_CATEGORIES = len(AGENT_GENERATORS)  # 42 dedicated Agent categories
AGENT_PER_CAT_TOTAL = 25_000_000              # 25M x 42 = 1,050,000,000 pure Agent SFT examples!
AGENT_TRAIN_PER_CAT = 24_800_000
AGENT_VAL_PER_CAT = 100_000
AGENT_TEST_PER_CAT = 100_000

AGENT_SFT_TOTAL = NUM_AGENT_CATEGORIES * AGENT_PER_CAT_TOTAL       # 1,050,000,000
AGENT_SFT_TRAIN = NUM_AGENT_CATEGORIES * AGENT_TRAIN_PER_CAT       # 1,041,600,000
AGENT_SFT_VAL = NUM_AGENT_CATEGORIES * AGENT_VAL_PER_CAT           #     4,200,000
AGENT_SFT_TEST = NUM_AGENT_CATEGORIES * AGENT_TEST_PER_CAT         #     4,200,000

AGENT_DPO_PER_TAX = 12_500_000
AGENT_DPO_TOTAL = len(DPO_TAXONOMIES) * AGENT_DPO_PER_TAX          #   100,000,000


def synthesize_agent_ultra_sft(agent_cat_idx: int, local_coord: int, seed: int = 42) -> dict[str, Any]:
    """
    Deterministically synthesize an ultra-clear, schema-verified Agent SFT trajectory
    for `agent_cat_idx in [0..41]` and `local_coord in [0..24_999_999]`.
    """
    s64 = int.from_bytes(
        hashlib.blake2b(
            struct.pack("<QQQ", seed & 0xFFFFFFFFFFFFFFFF, agent_cat_idx + 500, local_coord),
            digest_size=8,
        ).digest(),
        "little",
    )
    rng = random.Random(s64)
    _, category, gen_fn = AGENT_GENERATORS[agent_cat_idx]
    rec = gen_fn(rng, local_coord)

    # Bijective natural context decomposition: 50 * 50 * 10_000 = 25,000,000
    a_idx = local_coord % 40
    b_idx = (local_coord // 40) % 50
    c_num = 1000 + (local_coord // 2000)

    opener_ctx = NATURAL_CONTEXT_OPENERS[a_idx]
    prof = AGENT_PROFILES[b_idx]

    rec["id"] = f"agtu-{agent_cat_idx:02d}-{local_coord:08d}"
    tags = list(rec["tags"])[:5]
    if "agent-ultra" not in tags:
        tags.append("agent-ultra")
    rec["tags"] = tags

    msgs = rec["messages"]
    msgs[1]["content"] = f"{msgs[1]['content']}\n\n*({opener_ctx} #{c_num} · {prof})*"

    if rec["style"] == "agent_trajectory" and len(msgs) >= 3 and "<tool_call>" in msgs[2]["content"]:
        first_asst = msgs[2]["content"]
        if not first_asst.lstrip().startswith("<thought>"):
            idx_tc = first_asst.index("<tool_call>")
            tc_part = first_asst[idx_tc:]
            grounded_thought = _build_agent_thought(msgs[0]["content"], first_asst, msgs[1]["content"])
            msgs[2]["content"] = f"{grounded_thought}\n{tc_part}"

    return rec


def resolve_agent_sft_coord(split: str, index: int, seed: int = 42) -> tuple[int, int]:
    if split == "all":
        if not (0 <= index < AGENT_SFT_TOTAL):
            raise IndexError(index)
        if index < AGENT_SFT_TRAIN:
            return resolve_agent_sft_coord("train", index, seed)
        if index < AGENT_SFT_TRAIN + AGENT_SFT_VAL:
            return resolve_agent_sft_coord("validation", index - AGENT_SFT_TRAIN, seed)
        return resolve_agent_sft_coord("test", index - AGENT_SFT_TRAIN - AGENT_SFT_VAL, seed)

    if split == "train":
        p = feistel_permute(index, AGENT_SFT_TRAIN, seed * 1_000_003 + 101)
        return p % NUM_AGENT_CATEGORIES, p // NUM_AGENT_CATEGORIES
    if split == "validation":
        p = feistel_permute(index, AGENT_SFT_VAL, seed * 1_000_003 + 103)
        return p % NUM_AGENT_CATEGORIES, AGENT_TRAIN_PER_CAT + (p // NUM_AGENT_CATEGORIES)
    if split == "test":
        p = feistel_permute(index, AGENT_SFT_TEST, seed * 1_000_003 + 107)
        return p % NUM_AGENT_CATEGORIES, AGENT_TRAIN_PER_CAT + AGENT_VAL_PER_CAT + (p // NUM_AGENT_CATEGORIES)
    raise ValueError(f"Unknown split: {split}")


class AgentUltraDataset:
    """O(1) Random-Access 1,050,000,000-Example Pure Autonomous Agent SFT Dataset across 42 categories."""

    def __init__(self, split: str = "train", seed: int = 42) -> None:
        self.split = split
        self.seed = seed
        self._length = {
            "train": AGENT_SFT_TRAIN,
            "validation": AGENT_SFT_VAL,
            "test": AGENT_SFT_TEST,
            "all": AGENT_SFT_TOTAL,
        }[split]

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += self._length
        if not (0 <= index < self._length):
            raise IndexError(index)
        c_idx, l_coord = resolve_agent_sft_coord(self.split, index, self.seed)
        return synthesize_agent_ultra_sft(c_idx, l_coord, self.seed)


class AgentDPODataset:
    """O(1) Random-Access 100,000,000-Pair Pure Agent DPO / ORPO Preference Alignment Dataset."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._length = AGENT_DPO_TOTAL

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += self._length
        if not (0 <= index < self._length):
            raise IndexError(index)
        p = feistel_permute(index, self._length, self.seed * 1_000_003 + 211)
        tax_idx = p % len(DPO_TAXONOMIES)
        local_coord = p // len(DPO_TAXONOMIES)
        return synthesize_agent_dpo_pair(tax_idx, local_coord, self.seed)


def iter_agent_ultra_sft(
    *, split: str = "train", start: int = 0, count: int | None = None, rank: int = 0, world_size: int = 1, seed: int = 42
) -> Iterator[dict[str, Any]]:
    ds = AgentUltraDataset(split=split, seed=seed)
    end = len(ds) if count is None else min(len(ds), start + count)
    for idx in range(start + rank, end, world_size):
        yield ds[idx]
