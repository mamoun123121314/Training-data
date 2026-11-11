#!/usr/bin/env python3
"""
Interactive Multi-Turn Agent Sandbox Tester for Qwen2.5-Coder-3B Agent POV LoRA.

Actually executes `<tool_call>` requests against a safe in-memory mock workspace (`read_file`,
`edit_file`, `list_clusters`, `get_cluster_health`) across up to 3 turns so you can watch your
fine-tuned model think, call tools, read tool outputs, and finish the job!
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

MOCK_FILES: dict[str, str] = {
    "src/api/server.py": (
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n"
        "BATCH_LIMIT = 100\n"
        "TIMEOUT_SECONDS = 30\n"
    ),
    "src/payments/handler.py": (
        " BATCH_LIMIT = 128\n"
        "MAX_RETRIES = 3\n"
    ),
    "src/payments/retry.py": (
        "BACKOFF_BASE_MS = 250\n"
        "MAX_ATTEMPTS = 4\n"
    ),
}

MOCK_CLUSTERS: dict[str, list[dict[str, str]]] = {
    "staging": [{"cluster_id": "cls-stg-checkout-01", "service": "checkout-api", "status": "healthy"}],
    "production": [{"cluster_id": "cls-prd-pay-09", "service": "payments-core", "status": "healthy (CPU 41%)"}],
}


def execute_mock_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "read_file":
        p = arguments.get("path", "")
        if p in MOCK_FILES:
            return {"status": "ok", "path": p, "content": MOCK_FILES[p]}
        return {"status": "ok", "path": p, "content": "BATCH_LIMIT = 100\nTIMEOUT_SECONDS = 30\n"}
    if name == "edit_file":
        p = arguments.get("path", "")
        old_t = arguments.get("old_text", "")
        new_t = arguments.get("new_text", "")
        cur = MOCK_FILES.get(p, "BATCH_LIMIT = 100\nTIMEOUT_SECONDS = 30\n")
        if old_t and old_t in cur:
            MOCK_FILES[p] = cur.replace(old_t, new_t, 1)
            return {"status": "edited", "path": p, "updated_content": MOCK_FILES[p]}
        return {"status": "error", "message": f"old_text not found in {p}; read_file first."}
    if name == "list_clusters":
        env = arguments.get("env", "staging")
        return {"env": env, "clusters": MOCK_CLUSTERS.get(env, [{"cluster_id": f"cls-{env}-01", "status": "healthy"}])}
    if name == "get_cluster_health":
        cid = arguments.get("cluster_id", "")
        return {"cluster_id": cid, "status": "healthy", "p99_latency_ms": 18.4, "error_rate": 0.0002}
    return {"status": "ok", "tool": name, "echo": arguments}


def format_chatml(messages: list[dict[str, str]]) -> str:
    out: list[str] = []
    for m in messages:
        role = m["role"]
        c = m["content"].strip()
        if role == "tool":
            if not c.startswith("<tool_response>"):
                c = f"<tool_response>\n{c}\n</tool_response>"
            out.append(f"<|im_start|>user\n{c}<|im_end|>")
        else:
            out.append(f"<|im_start|>{role}\n{c}<|im_end|>")
    out.append("<|im_start|>assistant\n")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-turn live agent sandbox for fine-tuned Qwen2.5 LoRA")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--adapter-dir", type=str, default="out/qwen25-agent-pov-lora/stage1_sft_adapter")
    parser.add_argument("--prompt", type=str, default="Update the batch limit in `src/api/server.py` to `250`.")
    parser.add_argument("--max-turns", type=int, default=3)
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
    stop_ids = []
    if isinstance(tokenizer.eos_token_id, int):
        stop_ids.append(tokenizer.eos_token_id)
    for s in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(s)
        if isinstance(tid, int) and tid >= 0 and tid not in stop_ids:
            stop_ids.append(tid)

    try:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, quantization_config=bnb_config, dtype=compute_dtype, device_map="auto", trust_remote_code=True
        )
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, quantization_config=bnb_config, torch_dtype=compute_dtype, device_map="auto", trust_remote_code=True
        )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    tools = [
        {"name": "read_file", "description": "Read file contents.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
        {"name": "edit_file", "description": "Replace exact old_text with new_text.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
        {"name": "list_clusters", "description": "List cluster IDs in an environment.", "parameters": {"type": "object", "properties": {"env": {"type": "string"}}, "required": ["env"]}},
        {"name": "get_cluster_health", "description": "Fetch cluster health by exact cluster_id.", "parameters": {"type": "object", "properties": {"cluster_id": {"type": "string"}}, "required": ["cluster_id"]}},
    ]
    messages: list[dict[str, str]] = [
        {"role": "system", "content": build_agent_system(random.Random(42), tools)},
        {"role": "user", "content": args.prompt},
    ]

    tc_re = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

    print("=" * 74)
    print("USER PROMPT:", args.prompt)
    print("=" * 74)

    for turn in range(1, args.max_turns + 1):
        prompt_str = format_chatml(messages)
        inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=False,
                eos_token_id=stop_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)
        for st in ("<|im_end|>", "<|endoftext|>"):
            if st in reply:
                reply = reply.split(st, 1)[0]
        reply = reply.strip()
        print(f"\n[Agent Turn {turn}]\n{reply}")
        messages.append({"role": "assistant", "content": reply})

        match = tc_re.search(reply)
        if not match:
            break
        try:
            call_obj = json.loads(match.group(1))
            obs = execute_mock_tool(call_obj.get("name", ""), call_obj.get("arguments", {}))
        except Exception as exc:
            obs = {"status": "error", "error": str(exc)}
        obs_json = json.dumps(obs)
        print(f"\n[Sandbox Tool Output -> {obs_json}]")
        messages.append({"role": "tool", "content": obs_json})

    print("\n" + "=" * 74)


if __name__ == "__main__":
    main()
