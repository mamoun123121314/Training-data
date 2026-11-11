#!/usr/bin/env python3
"""
Interactive CLI / Colab tester for your fine-tuned Qwen2.5-Coder-3B Agent POV LoRA Adapter.

Usage in Colab:
  !python3 Training-data/scripts/test_agent_interactive.py --prompt "Update batch limit in src/api/server.py to 250"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Test your fine-tuned Agent POV LoRA model on any prompt")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--adapter-dir", type=str, default="out/qwen25-agent-pov-lora/final_lora_adapter")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Check cluster health for `payments-core` in `production`, then read `src/payments/retry.py` before editing.",
    )
    args = parser.parse_args()

    from peft import PeftModel
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from training_data.schema import build_agent_system

    adapter_path = Path(args.adapter_dir)
    if not adapter_path.exists() and (REPO_ROOT / args.adapter_dir).exists():
        adapter_path = REPO_ROOT / args.adapter_dir

    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    tools = [
        {"name": "list_clusters", "description": "List cluster IDs in an environment.", "parameters": {"type": "object", "properties": {"env": {"type": "string"}}, "required": ["env"]}},
        {"name": "get_cluster_health", "description": "Fetch cluster health by exact cluster_id.", "parameters": {"type": "object", "properties": {"cluster_id": {"type": "string"}}, "required": ["cluster_id"]}},
        {"name": "read_file", "description": "Read file contents.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
        {"name": "edit_file", "description": "Replace exact old_text with new_text.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    ]
    sys_prompt = build_agent_system(random.Random(42), tools)
    chatml = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{args.prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(chatml, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=240,
            do_sample=False,
            repetition_penalty=1.12,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False).replace("<|im_end|>", "").strip()
    print("=" * 72)
    print("PROMPT:", args.prompt)
    print("-" * 72)
    print(reply)
    print("=" * 72)


if __name__ == "__main__":
    main()
