#!/usr/bin/env python3
"""
Apex Closed-Loop Relentless Agent & Self-Verifying Coding Harness.

Features:
  - Zero-Config Backbone Detection: Automatically reads `base_model_name_or_path` from
    `adapter_config.json` so both `Qwen2.5-Coder-7B-Instruct` and `3B` adapters load seamlessly.
  - 6-Turn Live Sandbox (`--mode agent`): Executes `<tool_call>` requests against an interactive
    workspace (`read_file`, `search_code`, `list_files`, `edit_file`, `run_tests`, `list_clusters`,
    `get_cluster_health`) including real Python test verification and obstacle recovery!
  - AST & Anti-Laziness Self-Verifier (`--mode code`): Verifies generated Python code via `ast.parse`
    and rejects `# TODO` stubs, automatically running a self-repair pass if needed.
"""

from __future__ import annotations

import argparse
import ast
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
        "BATCH_LIMIT = 128\n"
        "MAX_RETRIES = 3\n"
    ),
    "src/billing/middleware/rate_gate.py": (
        "class RateGate:\n"
        "    MAX_BURST_TOKENS = 75\n"
        "    REFILL_PER_SEC = 25.0\n"
    ),
    "src/billing/settlement.py": (
        "def split_settlement_cents(total_cents: int, recipients: int) -> list[int]:\n"
        "    if recipients <= 0:\n"
        "        raise ValueError('recipients must be positive')\n"
        "    share = total_cents // recipients\n"
        "    return [share for _ in range(recipients)]\n"
    ),
}

MOCK_CLUSTERS: dict[str, list[dict[str, str]]] = {
    "staging": [{"cluster_id": "cls-stg-checkout-01", "service": "checkout-api", "status": "healthy"}],
    "production": [{"cluster_id": "cls-prd-pay-09", "service": "payments-core", "status": "healthy (CPU 41%)"}],
}


def run_sandbox_tests(target: str) -> dict[str, Any]:
    """Execute deterministic assertions against in-memory MOCK_FILES."""
    if "settlement" in target:
        ns: dict[str, Any] = {}
        try:
            exec(compile(MOCK_FILES["src/billing/settlement.py"], "settlement.py", "exec"), ns)  # noqa: S102
            fn = ns["split_settlement_cents"]
            res = fn(100, 3)
            if sum(res) != 100 or len(res) != 3:
                return {
                    "status": "failed",
                    "passed": 2,
                    "failed": 1,
                    "failure": f"AssertionError: sum(split_settlement_cents(100, 3)) == {sum(res)} != 100 (got {res})",
                }
            return {"status": "passed", "passed": 3, "failed": 0}
        except Exception as exc:
            return {"status": "failed", "passed": 0, "failed": 1, "failure": repr(exc)}
    return {"status": "passed", "target": target, "passed": 8, "failed": 0}


def execute_mock_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "list_files":
        prefix = arguments.get("directory", "")
        matched = sorted(p for p in MOCK_FILES if p.startswith(prefix))
        return {"status": "ok", "files": matched}
    if name == "search_code":
        q = str(arguments.get("query", ""))
        hits = []
        for path, content in sorted(MOCK_FILES.items()):
            for line_no, line in enumerate(content.splitlines(), start=1):
                if q in line or q in path:
                    hits.append({"path": path, "line": line_no, "snippet": line})
        return {"status": "ok", "matches": hits}
    if name == "read_file":
        p = arguments.get("path", "")
        if p in MOCK_FILES:
            return {"status": "ok", "path": p, "content": MOCK_FILES[p]}
        return {"status": "error", "error": f"FileNotFoundError: '{p}' does not exist"}
    if name == "edit_file":
        p = arguments.get("path", "")
        old_t = arguments.get("old_text", "")
        new_t = arguments.get("new_text", "")
        if p not in MOCK_FILES:
            return {"status": "error", "error": f"FileNotFoundError: '{p}' does not exist"}
        cur = MOCK_FILES[p]
        if old_t and old_t in cur:
            MOCK_FILES[p] = cur.replace(old_t, new_t, 1)
            return {"status": "edited", "path": p, "updated_content": MOCK_FILES[p]}
        return {"status": "error", "error": f"old_text not found in {p}; call read_file first."}
    if name == "run_tests":
        return run_sandbox_tests(str(arguments.get("target", "tests")))
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


def resolve_adapter_and_base(adapter_arg: str, base_arg: str | None) -> tuple[Path, str]:
    candidates = [
        Path(adapter_arg),
        REPO_ROOT / adapter_arg,
        Path("out/qwen25-agent-pov-lora/final_lora_adapter"),
        REPO_ROOT / "out/qwen25-agent-pov-lora/final_lora_adapter",
        Path("out/qwen25-agent-pov-lora/stage1_sft_adapter"),
        REPO_ROOT / "out/qwen25-agent-pov-lora/stage1_sft_adapter",
    ]
    adapter_path = candidates[0]
    for cand in candidates:
        if (cand / "adapter_config.json").exists():
            adapter_path = cand
            break

    detected_base = base_arg
    cfg_file = adapter_path / "adapter_config.json"
    if not detected_base and cfg_file.exists():
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        detected_base = cfg.get("base_model_name_or_path")
    if not detected_base:
        detected_base = "Qwen/Qwen2.5-Coder-7B-Instruct"
    return adapter_path, detected_base


def main() -> None:
    parser = argparse.ArgumentParser(description="Apex 6-Turn Relentless Agent & Self-Verifying Coding Harness")
    parser.add_argument("--base-model", type=str, default=None, help="Auto-detected from adapter_config.json if omitted")
    parser.add_argument("--adapter-dir", type=str, default="out/qwen25-agent-pov-lora/final_lora_adapter")
    parser.add_argument("--mode", choices=("agent", "code"), default="agent")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Raise `MAX_BURST_TOKENS` in `src/billing/limiter.py` to `300` and run `tests/billing` to verify.",
    )
    parser.add_argument("--max-turns", type=int, default=6)
    args = parser.parse_args()

    from peft import PeftModel
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from training_data.apex_intelligence import APEX_SYSTEM_DIRECT
    from training_data.schema import build_agent_system

    adapter_path, base_model_id = resolve_adapter_and_base(args.adapter_dir, args.base_model)
    compute_dtype = (
        torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print("=" * 76)
    print(f" Base Model  : {base_model_id}")
    print(f" Adapter Dir : {adapter_path}")
    print(f" Mode        : {args.mode.upper()} (Closed-Loop Self-Verification Active)")
    print("=" * 76)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    stop_ids: list[int] = []
    if isinstance(tokenizer.eos_token_id, int):
        stop_ids.append(tokenizer.eos_token_id)
    for s in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(s)
        if isinstance(tid, int) and tid >= 0 and tid not in stop_ids:
            stop_ids.append(tid)

    try:
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id, quantization_config=bnb_config, dtype=compute_dtype, device_map="auto", trust_remote_code=True
        )
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id, quantization_config=bnb_config, torch_dtype=compute_dtype, device_map="auto", trust_remote_code=True
        )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    def generate_turn(msgs: list[dict[str, str]], max_new_tokens: int = 384) -> str:
        prompt_str = format_chatml(msgs)
        inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=stop_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)
        for st in ("<|im_end|>", "<|endoftext|>"):
            if st in reply:
                reply = reply.split(st, 1)[0]
        return reply.strip()

    print("\nUSER PROMPT:", args.prompt)
    print("-" * 76)

    if args.mode == "code":
        messages = [
            {"role": "system", "content": APEX_SYSTEM_DIRECT},
            {"role": "user", "content": args.prompt},
        ]
        for attempt in range(1, 3):
            reply = generate_turn(messages, max_new_tokens=640)
            print(f"\n[Apex Generation Pass {attempt}]\n{reply}")
            messages.append({"role": "assistant", "content": reply})

            py_blocks = re.findall(r"```python\s*(.*?)```", reply, flags=re.DOTALL)
            issues: list[str] = []
            for blk in py_blocks:
                try:
                    ast.parse(blk)
                except SyntaxError as syn:
                    issues.append(f"Python SyntaxError at line {syn.lineno}: {syn.msg}")
                if "# TODO" in blk or "NotImplementedError" in blk:
                    issues.append("Contains unfinished placeholder (`# TODO` or `NotImplementedError`).")
            if not issues:
                if py_blocks:
                    print("\n[Self-Verifier]: AST syntax clean & zero lazy placeholders.")
                break
            feedback = "Verifier caught issues: " + "; ".join(issues) + " Output the complete, fixed implementation."
            print(f"\n[Self-Verifier Retry -> {feedback}]")
            messages.append({"role": "user", "content": feedback})
        print("\n" + "=" * 76)
        return

    tools = [
        {"name": "read_file", "description": "Read exact file contents.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
        {"name": "search_code", "description": "Search repository files for a symbol or filename.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
        {"name": "list_files", "description": "List files under a directory prefix.", "parameters": {"type": "object", "properties": {"directory": {"type": "string"}}, "required": ["directory"]}},
        {"name": "edit_file", "description": "Replace exact old_text with new_text.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
        {"name": "run_tests", "description": "Run pytest suite and return pass/fail status.", "parameters": {"type": "object", "properties": {"target": {"type": "string"}}, "required": ["target"]}},
        {"name": "list_clusters", "description": "List cluster IDs in an environment.", "parameters": {"type": "object", "properties": {"env": {"type": "string"}}, "required": ["env"]}},
        {"name": "get_cluster_health", "description": "Fetch cluster health by exact cluster_id.", "parameters": {"type": "object", "properties": {"cluster_id": {"type": "string"}}, "required": ["cluster_id"]}},
    ]
    messages = [
        {"role": "system", "content": build_agent_system(random.Random(42), tools)},
        {"role": "user", "content": args.prompt},
    ]

    tc_re = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    last_tool_failed = False

    for turn in range(1, args.max_turns + 1):
        reply = generate_turn(messages, max_new_tokens=280)
        print(f"\n[Agent Turn {turn}]\n{reply}")
        messages.append({"role": "assistant", "content": reply})

        match = tc_re.search(reply)
        if not match:
            if last_tool_failed and turn < args.max_turns:
                nudge = {"status": "verifier_notice", "message": "Previous tool returned an error/failure. Use search_code, list_files, or read_file to resolve it and complete the task."}
                print(f"\n[Closed-Loop Verifier -> {json.dumps(nudge)}]")
                messages.append({"role": "tool", "content": json.dumps(nudge)})
                last_tool_failed = False
                continue
            break

        try:
            call_obj = json.loads(match.group(1))
            obs = execute_mock_tool(call_obj.get("name", ""), call_obj.get("arguments", {}))
        except Exception as exc:
            obs = {"status": "error", "error": f"Invalid tool_call JSON: {exc}"}

        last_tool_failed = obs.get("status") in {"error", "failed"}
        obs_json = json.dumps(obs)
        print(f"\n[Sandbox Tool Output -> {obs_json}]")
        messages.append({"role": "tool", "content": obs_json})

    print("\n" + "=" * 76)


if __name__ == "__main__":
    main()
