"""Automated unit & integration tests for v5.0.0 Agent Ultra SFT/DPO, 75 generators, Feistel 1B engines, and persisted shards."""

from __future__ import annotations

import json
import lzma
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training_data import (  # noqa: E402
    AGENT_DPO_TOTAL,
    AGENT_SFT_TOTAL,
    TOTAL_BILLION_RECORDS,
    AgentDPODataset,
    AgentUltraDataset,
    BillionUltraDataset,
    feistel_inverse,
    feistel_permute,
    iter_agent_ultra_sft,
    iter_billion_ultra,
    iter_streaming_dataset,
    validate_dpo_pair,
)
from training_data.generators import AGENT_GENERATORS, ALL_GENERATORS  # noqa: E402
from training_data.solvers import (  # noqa: E402
    compute_cvss31,
    evaluate_firewall_rules,
    solve_dijkstra,
    solve_min_ship_capacity,
    solve_subarray_at_most_budget,
)
from training_data.validator import validate_record  # noqa: E402


class TestDatasetQuality(unittest.TestCase):
    def test_solvers_exactness(self) -> None:
        res1 = compute_cvss31(av="N", ac="L", pr="N", ui="N", s="C", c="H", i="H", a="H")
        self.assertEqual((res1["score"], res1["severity"]), (10.0, "Critical"))

        res2 = compute_cvss31(av="N", ac="L", pr="L", ui="N", s="U", c="H", i="H", a="N")
        self.assertEqual((res2["score"], res2["severity"]), (8.1, "High"))

        self.assertEqual(solve_subarray_at_most_budget([2, 1, 3, 2, 1, 1], 5), 3)
        self.assertEqual(solve_min_ship_capacity([10, 20, 30, 40], 2), 60)
        self.assertEqual(solve_dijkstra(4, [(0, 1, 2), (0, 2, 6), (1, 2, 1), (1, 3, 7), (2, 3, 2)], 0, 3), 5)

        verdict, matched_id, shadowed = evaluate_firewall_rules(
            [
                {"id": 10, "action": "ALLOW", "src": "10.20.0.0/16", "port": 5432, "proto": "TCP"},
                {"id": 20, "action": "DENY", "src": "10.20.1.0/24", "port": 5432, "proto": "TCP"},
            ],
            "10.20.1.15",
            5432,
            "TCP",
        )
        self.assertEqual((verdict, matched_id), ("ALLOW", 10))
        self.assertIn((20, 10), shadowed)

    def test_all_75_generators_and_42_agent_tracks(self) -> None:
        self.assertEqual(len(ALL_GENERATORS), 75)
        self.assertEqual(len(AGENT_GENERATORS), 42)
        rng = random.Random(12345)
        for domain, category, gen_fn in ALL_GENERATORS:
            for idx in range(5):
                rec = gen_fn(rng, idx)
                ok, reason = validate_record(rec)
                self.assertTrue(ok, f"Generator {category} failed at idx={idx}: {reason}")
                self.assertEqual(rec["domain"], domain)
                self.assertEqual(rec["category"], category)

    def test_agent_ultra_1b_sft_and_100m_dpo_engines(self) -> None:
        ds_agent = AgentUltraDataset("all", seed=42)
        self.assertEqual(len(ds_agent), AGENT_SFT_TOTAL)
        self.assertEqual(len(ds_agent), 1_050_000_000)

        for g_idx in (0, 1, 42, 100_000_000, 525_000_000, 1_049_999_999):
            rec = ds_agent[g_idx]
            ok, reason = validate_record(rec)
            self.assertTrue(ok, f"AgentUltra[{g_idx}] failed: {reason}")
            self.assertEqual(rec["domain"], "agent_use")
            self.assertIn("agent-ultra", rec["tags"])

        ds_dpo = AgentDPODataset(seed=42)
        self.assertEqual(len(ds_dpo), AGENT_DPO_TOTAL)
        self.assertEqual(len(ds_dpo), 100_000_000)
        for d_idx in (0, 7, 50_000_000, 99_999_999):
            pair = ds_dpo[d_idx]
            ok, reason = validate_dpo_pair(pair)
            self.assertTrue(ok, f"AgentDPO[{d_idx}] failed: {reason}")

        r0 = [r["id"] for r in iter_agent_ultra_sft(split="train", start=0, count=10, rank=0, world_size=2, seed=42)]
        r1 = [r["id"] for r in iter_agent_ultra_sft(split="train", start=0, count=10, rank=1, world_size=2, seed=42)]
        self.assertEqual(len(set(r0) & set(r1)), 0)

    def test_persisted_agent_ultra_pack_in_data(self) -> None:
        ag_dir = ROOT / "data" / "agent_ultra"
        self.assertTrue((ag_dir / "dataset_stats.json").exists())
        st = json.loads((ag_dir / "dataset_stats.json").read_text(encoding="utf-8"))
        self.assertEqual(st["materialized_summary"]["agent_sft_records"], 450_000)
        self.assertEqual(st["materialized_summary"]["agent_dpo_pairs"], 200_000)

        with lzma.open(ag_dir / "sft_validation.jsonl.xz", "rt", encoding="utf-8") as fh:
            for _ in range(20):
                rec = json.loads(fh.readline())
                ok, reason = validate_record(rec)
                self.assertTrue(ok, reason)

        with lzma.open(ag_dir / "dpo_validation.jsonl.xz", "rt", encoding="utf-8") as fh:
            for _ in range(20):
                pair = json.loads(fh.readline())
                ok, reason = validate_dpo_pair(pair)
                self.assertTrue(ok, reason)

    def test_feistel_bijection_and_multi_domain_1b(self) -> None:
        for dom in (1, 17, 257, 2049):
            mapped = {feistel_permute(i, dom, 42) for i in range(dom)}
            self.assertEqual(len(mapped), dom)
            for i in range(dom):
                self.assertEqual(feistel_inverse(feistel_permute(i, dom, 42), dom, 42), i)

        ds_all = BillionUltraDataset("all", seed=42)
        self.assertEqual(len(ds_all), TOTAL_BILLION_RECORDS)
        for g_idx in (0, 181, 500_000_000, 999_999_999):
            ok, reason = validate_record(ds_all[g_idx])
            self.assertTrue(ok, reason)

        r0 = [r["id"] for r in iter_billion_ultra(split="train", start=0, count=10, rank=0, world_size=2, seed=42)]
        r1 = [r["id"] for r in iter_billion_ultra(split="train", start=0, count=10, rank=1, world_size=2, seed=42)]
        self.assertEqual(len(set(r0) & set(r1)), 0)

    def test_validator_rejects_bad_python(self) -> None:
        bad_rec = {
            "id": "td-bad-0001",
            "domain": "coding",
            "category": "bug_diagnosis_and_fix",
            "difficulty": "easy",
            "style": "direct",
            "tags": ["bugfix"],
            "quality_check": "ast_verified",
            "messages": [
                {"role": "system", "content": "You are a helpful software engineer."},
                {"role": "user", "content": "Write a Python helper function to add two integers cleanly."},
                {"role": "assistant", "content": "```python\ndef broken(:\n    pass\n```"},
            ],
        }
        ok, reason = validate_record(bad_rec)
        self.assertFalse(ok)
        self.assertIn("python_syntax_error", reason)

    def test_streaming_generator_determinism(self) -> None:
        batch_a = [
            rec["id"]
            for _, rec in zip(
                range(30),
                iter_streaming_dataset(per_category=10, val_per_category=1, test_per_category=1, split="train", seed=42),
            )
        ]
        batch_b = [
            rec["id"]
            for _, rec in zip(
                range(30),
                iter_streaming_dataset(per_category=10, val_per_category=1, test_per_category=1, split="train", seed=42),
            )
        ]
        self.assertEqual(batch_a, batch_b)

    def test_apex_intelligence_and_trainer_curriculum(self) -> None:
        import ast
        import re
        from training_data.apex_intelligence import synthesize_apex_dpo_pair, synthesize_apex_sft_trajectory
        from training_data.appsec_audit_patch import synthesize_appsec_audit_dpo_pair, synthesize_appsec_audit_sft

        for i in range(36):
            rec = synthesize_apex_sft_trajectory(i, seed=42)
            self.assertGreaterEqual(len(rec["messages"]), 3)
            for msg in rec["messages"]:
                if msg["role"] == "assistant":
                    self.assertNotIn("From my POV", msg["content"])
                    self.assertNotIn("# TODO", msg["content"])
                    for blk in re.findall(r"```python\s*(.*?)```", msg["content"], flags=re.DOTALL):
                        ast.parse(blk)

        for i in range(24):
            rec = synthesize_appsec_audit_sft(i, seed=42)
            self.assertGreaterEqual(len(rec["messages"]), 3)
            for msg in rec["messages"]:
                if msg["role"] == "assistant":
                    self.assertNotIn("I cannot", msg["content"])
                    self.assertNotIn("I'm sorry", msg["content"])
                    for blk in re.findall(r"```python\s*(.*?)```", msg["content"], flags=re.DOTALL):
                        ast.parse(blk)

        for j in range(16):
            for pair in (synthesize_apex_dpo_pair(j, seed=42), synthesize_appsec_audit_dpo_pair(j, seed=42)):
                chosen_txt = pair["chosen"][0]["content"]
                self.assertIn("<thought>", chosen_txt)
                self.assertNotIn("# TODO", chosen_txt)
                self.assertNotIn("I'm sorry, I cannot", chosen_txt)
                for blk in re.findall(r"```python\s*(.*?)```", chosen_txt, flags=re.DOTALL):
                    ast.parse(blk)

        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root / "scripts") not in sys.path:
            sys.path.insert(0, str(repo_root / "scripts"))
        import train_gpu_qlora_sft_dpo as trainer_mod

        sft_rows = trainer_mod.build_sft_examples(100, seed=42)
        self.assertEqual(len(sft_rows), 100)
        self.assertTrue(all("<|im_start|>assistant\n" in r["text"] for r in sft_rows))

        dpo_rows = trainer_mod.build_dpo_examples(80, seed=42)
        self.assertEqual(len(dpo_rows), 80)
        chosen_tc = sum("<tool_call>" in r["chosen"] for r in dpo_rows)
        rejected_tc = sum("<tool_call>" in r["rejected"] for r in dpo_rows)
        self.assertGreaterEqual(chosen_tc, rejected_tc)


if __name__ == "__main__":
    unittest.main()
