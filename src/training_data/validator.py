"""Strict automated quality, clarity, AST syntax, JSON/tool-schema, safety, and deduplication gates."""

from __future__ import annotations

import ast
import functools
import hashlib
import json
import re
from typing import Any

from training_data.schema import VALID_DIFFICULTIES, VALID_DOMAINS, VALID_ROLES, VALID_STYLES

FLUFF_PATTERNS = re.compile(
    r"(^\s*(certainly|sure|of course|absolutely|great question|i'd be happy to|i would be happy to)[!.,])"
    r"|(\bas an ai language model\b)"
    r"|(\bhope this helps\b)"
    r"|(\blet me know if you have any other questions\b)"
    r"|(\bTODO:\s*implement\b)",
    re.IGNORECASE,
)

UNSAFE_WEAPONIZATION_PATTERNS = re.compile(
    r"\b(msfvenom|meterpreter|cobalt\s*strike\s*beacon|wannacry\s*payload|build_ransomware|steal_browser_cookies)\b",
    re.IGNORECASE,
)

PY_CODE_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
JSON_CODE_BLOCK_RE = re.compile(r"```json\n(.*?)```", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
TOOLS_SYSTEM_RE = re.compile(r"<tools>\n(.*?)\n</tools>", re.DOTALL)
WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return WS_RE.sub(" ", text.strip().lower())


def sha256_norm(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def digest128_norm(text: str) -> bytes:
    """Compact 16-byte (128-bit) digest for memory-efficient million-scale deduplication."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).digest()[:16]


@functools.lru_cache(maxsize=8192)
def parse_declared_tools(system_content: str) -> dict[str, tuple[str, ...]]:
    """Parse <tools> JSON lines and cache mapping {tool_name: tuple(required_args)}."""
    match = TOOLS_SYSTEM_RE.search(system_content)
    if not match:
        return {}
    declared: dict[str, tuple[str, ...]] = {}
    for line in match.group(1).strip().splitlines():
        line = line.strip()
        if not line:
            continue
        spec = json.loads(line)
        fn = spec.get("function", spec)
        req = tuple(fn.get("parameters", {}).get("required", ()))
        declared[fn["name"]] = req
    return declared


@functools.lru_cache(maxsize=32768)
def _check_python_snippet(py_code: str) -> str | None:
    try:
        ast.parse(py_code)
        return None
    except SyntaxError as exc:
        return f"{exc.msg}@line{exc.lineno}"


def record_digests128(record: dict[str, Any]) -> tuple[bytes, bytes]:
    user_concat = " || ".join(m["content"] for m in record["messages"] if m["role"] == "user")
    full_concat = " || ".join(f"{m['role']}:{m['content']}" for m in record["messages"][1:])
    return digest128_norm(user_concat), digest128_norm(full_concat)


def validate_record(record: dict[str, Any]) -> tuple[bool, str]:
    """Validate a single ChatML record across schema, clarity, syntax, tool schemas, and safety."""
    required_keys = {"id", "domain", "category", "difficulty", "style", "tags", "quality_check", "messages"}
    if len(record) != 8 or required_keys - set(record.keys()):
        missing = required_keys - set(record.keys())
        return False, f"missing_keys:{sorted(missing)}"

    if record["domain"] not in VALID_DOMAINS:
        return False, f"invalid_domain:{record['domain']}"
    if record["difficulty"] not in VALID_DIFFICULTIES:
        return False, f"invalid_difficulty:{record['difficulty']}"
    if record["style"] not in VALID_STYLES:
        return False, f"invalid_style:{record['style']}"
    if not isinstance(record["tags"], list) or not (1 <= len(record["tags"]) <= 6):
        return False, "invalid_tags"

    messages = record["messages"]
    if not isinstance(messages, list) or len(messages) < 3:
        return False, "messages_too_short"

    if messages[0].get("role") != "system":
        return False, "first_role_must_be_system"
    if messages[1].get("role") != "user":
        return False, "second_role_must_be_user"
    if messages[-1].get("role") != "assistant":
        return False, "last_role_must_be_assistant"

    try:
        declared_tools = parse_declared_tools(messages[0]["content"])
    except Exception as exc:
        return False, f"invalid_tools_declaration:{exc}"

    prev_role = ""
    for idx, msg in enumerate(messages):
        if len(msg) != 2 or "role" not in msg or "content" not in msg:
            return False, f"msg_{idx}_invalid_keys"
        role = msg["role"]
        content = msg["content"]
        if role not in VALID_ROLES:
            return False, f"msg_{idx}_invalid_role:{role}"
        if not isinstance(content, str) or len(content.strip()) < 8:
            return False, f"msg_{idx}_empty_or_short_content"

        if idx > 0:
            if prev_role == "user" and role != "assistant":
                return False, f"msg_{idx}_user_must_be_followed_by_assistant"
            if prev_role == "tool" and role not in ("assistant", "tool"):
                return False, f"msg_{idx}_tool_must_be_followed_by_assistant"
            if prev_role == "assistant" and role == "assistant":
                return False, f"msg_{idx}_consecutive_assistant"
        prev_role = role

        if UNSAFE_WEAPONIZATION_PATTERNS.search(content):
            return False, f"msg_{idx}_unsafe_weaponization_pattern"

        if role == "assistant":
            if FLUFF_PATTERNS.search(content):
                return False, f"msg_{idx}_contains_fluff"

            if "```python\n" in content:
                for py_code in PY_CODE_BLOCK_RE.findall(content):
                    err = _check_python_snippet(py_code)
                    if err is not None:
                        return False, f"msg_{idx}_python_syntax_error:{err}"

            if "```json\n" in content:
                for json_block in JSON_CODE_BLOCK_RE.findall(content):
                    try:
                        json.loads(json_block)
                    except Exception as exc:
                        return False, f"msg_{idx}_json_block_error:{exc}"

            if "<tool_call>" in content:
                for raw_call in TOOL_CALL_RE.findall(content):
                    try:
                        call_obj = json.loads(raw_call)
                    except Exception as exc:
                        return False, f"msg_{idx}_tool_call_json_error:{exc}"
                    if not isinstance(call_obj, dict) or "name" not in call_obj or "arguments" not in call_obj:
                        return False, f"msg_{idx}_tool_call_missing_name_or_args"
                    fn_name = call_obj["name"]
                    fn_args = call_obj["arguments"]
                    if not isinstance(fn_name, str) or not isinstance(fn_args, dict):
                        return False, f"msg_{idx}_tool_call_invalid_types"
                    if declared_tools and fn_name not in declared_tools:
                        return False, f"msg_{idx}_undeclared_tool:{fn_name}"
                    if fn_name in declared_tools:
                        for req_key in declared_tools[fn_name]:
                            if req_key not in fn_args:
                                return False, f"msg_{idx}_missing_required_arg:{fn_name}.{req_key}"

        elif role == "tool":
            try:
                json.loads(content)
            except Exception as exc:
                return False, f"msg_{idx}_invalid_tool_response_json:{exc}"

    return True, "ok"


class DeduplicationRegistry:
    """Tracks 128-bit SHA-256 digests of normalized user prompts and full conversations."""

    def __init__(self) -> None:
        self.user_hashes: set[bytes] = set()
        self.conv_hashes: set[bytes] = set()

    def check_and_register(self, record: dict[str, Any]) -> bool:
        u_hash, c_hash = record_digests128(record)
        if u_hash in self.user_hashes or c_hash in self.conv_hashes:
            return False
        self.user_hashes.add(u_hash)
        self.conv_hashes.add(c_hash)
        return True
