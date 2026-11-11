"""Ethical Security Generators Part 3 (Categories 42–46): Deserialization/XXE, GraphQL/API Security, Linux Hardening, Secrets/OIDC, CSP Headers."""

from __future__ import annotations

import json
import random
from typing import Any

from training_data.schema import make_record, pick_security_system
from training_data.vocab import (
    ENVIRONMENTS,
    FILE_PATHS,
    SECURITY_OPENERS_DIRECT,
    SECURITY_OPENERS_STEP,
    SERVICES,
    TENANT_SLUGS,
)


# ---------------------------------------------------------------------------
# 42. ssrf_xxe_deserialization_deep_dive
# ---------------------------------------------------------------------------

def gen_ssrf_xxe_deserialization_deep_dive(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    fpath = rng.choice(FILE_PATHS)
    style = rng.choice(["direct", "step_by_step"])
    opener = rng.choice(SECURITY_OPENERS_DIRECT if style == "direct" else SECURITY_OPENERS_STEP)
    variant = idx % 2

    if variant == 0:
        difficulty = "hard"
        vuln = (
            "import pickle\n\n"
            f"def load_{domain_ctx}_session(raw_bytes: bytes) -> dict:\n"
            "    return pickle.loads(raw_bytes)"
        )
        patch = (
            "import hashlib\nimport hmac\nimport json\nfrom typing import Any\n\n"
            f"def load_{domain_ctx}_session(raw_bytes: bytes, sig_hex: str, secret: bytes) -> dict[str, Any]:\n"
            "    expected = hmac.new(secret, raw_bytes, hashlib.sha256).hexdigest()\n"
            "    if not hmac.compare_digest(expected, sig_hex):\n"
            "        raise PermissionError('invalid session signature')\n"
            "    data = json.loads(raw_bytes.decode('utf-8'))\n"
            "    if not isinstance(data, dict):\n"
            "        raise ValueError('expected JSON object')\n"
            "    return data"
        )
        desc = "**CWE-502 (Deserialization of Untrusted Data)**: `pickle.loads` executes arbitrary `__reduce__` callables during unpickling. Replace with HMAC-authenticated UTF-8 JSON."
    else:
        difficulty = "medium"
        vuln = (
            "from lxml import etree\n\n"
            f"def parse_{domain_ctx}_xml(xml_bytes: bytes):\n"
            "    parser = etree.XMLParser(resolve_entities=True)\n"
            "    return etree.fromstring(xml_bytes, parser=parser)"
        )
        patch = (
            "from defusedxml.ElementTree import fromstring\nfrom typing import Any\n\n"
            f"def parse_{domain_ctx}_xml(xml_bytes: bytes) -> Any:\n"
            "    return fromstring(xml_bytes, forbid_dtd=True, forbid_entities=True, forbid_external=True)"
        )
        desc = "**CWE-611 (XML External Entity / XXE)**: `resolve_entities=True` allows `<!ENTITY SYSTEM 'file:///etc/passwd'>` disclosure and Billion Laughs DoS. Use `defusedxml` with `forbid_dtd=True`."

    user_q = f"{opener}\nService `{svc_name}` (`{fpath}`, `{env}`, review `SEC-{idx + 100}`):\n```python\n{vuln}\n```"
    ans = f"### Deserialization & Parser Hardening\n- {desc}\n\n```python\n{patch}\n```" if style == "step_by_step" else f"{desc}\n```python\n{patch}\n```"

    return make_record(
        record_id=f"td-deser-{idx:07d}",
        domain="ethical_security",
        category="ssrf_xxe_deserialization_deep_dive",
        difficulty=difficulty,
        style=style,
        tags=["cwe-502", "xxe", "appsec"],
        quality_check="ast_and_cwe_verified",
        messages=[
            {"role": "system", "content": pick_security_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans},
        ],
    )


# ---------------------------------------------------------------------------
# 43. api_rate_limiting_and_graphql_security
# ---------------------------------------------------------------------------

def gen_api_rate_limiting_and_graphql_security(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    style = rng.choice(["direct", "step_by_step"])
    max_depth = rng.randint(4, 9)
    variant = idx % 2

    if variant == 0:
        difficulty = "medium"
        max_depth = rng.randint(3, 14)
        fpath = rng.choice(FILE_PATHS)
        user_q = (
            f"Write a Python AST/dict depth guard `enforce_query_depth(selection: dict, max_depth: int = {max_depth}) -> None` "
            f"for `{svc_name}` (`{fpath}`, `{env}`, rule `GQL-{idx + 100}`) GraphQL BFF to block deeply nested cyclical DoS queries (`OWASP API4:2023`)."
        )
        code = (
            "from typing import Any\n\n"
            f"def enforce_query_depth(selection: dict[str, Any], max_depth: int = {max_depth}, _depth: int = 1) -> None:\n"
            "    if _depth > max_depth:\n"
            "        raise ValueError(f'GraphQL query depth {_depth} exceeds max_depth={max_depth}')\n"
            "    for sub in selection.values():\n"
            "        if isinstance(sub, dict):\n"
            "            enforce_query_depth(sub, max_depth=max_depth, _depth=_depth + 1)"
        )
        note = f"Rejects nested selections exceeding `{max_depth}` levels before resolver execution."
    else:
        difficulty = "medium"
        user_q = (
            f"Remediate mass assignment (`CWE-915` / `OWASP API3:2023`) in `{svc_name}` (`{env}`, ref `{idx + 100}`), where `user_dict.update(request_json)` "
            f"lets callers overwrite `'is_admin'` or `'tenant_id'` alongside editable fields `('display_name', 'timezone')`."
        )
        code = (
            "from typing import Any\n\n"
            "EDITABLE_FIELDS = frozenset({'display_name', 'timezone'})\n\n"
            "def apply_profile_patch(profile: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:\n"
            "    unknown = set(body) - EDITABLE_FIELDS\n"
            "    if unknown:\n"
            "        raise PermissionError(f'disallowed fields: {sorted(unknown)}')\n"
            "    updated = dict(profile)\n"
            "    for k in EDITABLE_FIELDS & set(body):\n"
            "        updated[k] = str(body[k])\n"
            "    return updated"
        )
        note = "Explicit field allowlisting (`EDITABLE_FIELDS`) blocks privilege escalation via injected `is_admin`/`tenant_id` attributes."

    ans = f"### API Security Control\n- {note}\n\n```python\n{code}\n```" if style == "step_by_step" else f"```python\n{code}\n```\n- {note}"
    return make_record(
        record_id=f"td-apisec-{idx:07d}",
        domain="ethical_security",
        category="api_rate_limiting_and_graphql_security",
        difficulty=difficulty,
        style=style,
        tags=["api-security", "graphql", "owasp-api"],
        quality_check="ast_and_cwe_verified",
        messages=[
            {"role": "system", "content": pick_security_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans},
        ],
    )


# ---------------------------------------------------------------------------
# 44. linux_systemd_and_host_hardening
# ---------------------------------------------------------------------------

def gen_linux_systemd_and_host_hardening(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    style = rng.choice(["direct", "step_by_step"])
    port = rng.randint(8000, 9600)

    user_q = (
        f"Write a sandboxed `systemd` `[Service]` unit for `{svc_name}` (`{env}`, `/opt/{svc_name}/bin/server --port {port}`) "
        f"enforcing read-only OS mounts, private `/tmp`, zero Linux capabilities, and `AF_INET`/`AF_INET6` socket family restriction."
    )
    unit = (
        "[Service]\n"
        f"User={domain_ctx}\nGroup={domain_ctx}\n"
        f"ExecStart=/opt/{svc_name}/bin/server --port {port}\n"
        "NoNewPrivileges=yes\n"
        "ProtectSystem=strict\n"
        "ProtectHome=yes\n"
        "PrivateTmp=yes\n"
        "PrivateDevices=yes\n"
        "ProtectKernelTunables=yes\n"
        "CapabilityBoundingSet=\n"
        "RestrictAddressFamilies=AF_INET AF_INET6"
    )
    note = "`ProtectSystem=strict` mounts `/` read-only, `CapabilityBoundingSet=` drops all root capabilities, and `NoNewPrivileges=yes` blocks SUID escalation."
    ans = f"### Systemd Sandboxing\n- {note}\n\n```ini\n{unit}\n```" if style == "step_by_step" else f"```ini\n{unit}\n```\n- {note}"

    return make_record(
        record_id=f"td-hostsec-{idx:07d}",
        domain="ethical_security",
        category="linux_systemd_and_host_hardening",
        difficulty="medium",
        style=style,
        tags=["linux-hardening", "systemd", "sandboxing"],
        quality_check="config_structure_verified",
        messages=[
            {"role": "system", "content": pick_security_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans},
        ],
    )


# ---------------------------------------------------------------------------
# 45. secrets_management_and_ci_supply_chain
# ---------------------------------------------------------------------------

def gen_secrets_management_and_ci_supply_chain(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    org = rng.choice(TENANT_SLUGS)
    env = rng.choice(ENVIRONMENTS)
    style = rng.choice(["direct", "step_by_step"])
    acct = f"{100000000000 + idx}"

    user_q = (
        f"Audit an AWS IAM GitHub OIDC trust policy for `{org}/{svc_name}` (`{env}`, account `{acct}`): "
        f"why is checking only `token.actions.githubusercontent.com:aud == sts.amazonaws.com` critically insecure, and write the fixed trust policy locking `sub` to `refs/heads/main`."
    )
    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Federated": f"arn:aws:iam::{acct}:oidc-provider/token.actions.githubusercontent.com"},
                "Action": "sts:AssumeRoleWithWebIdentity",
                "Condition": {
                    "StringEquals": {
                        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
                        "token.actions.githubusercontent.com:sub": f"repo:{org}/{svc_name}:ref:refs/heads/main",
                    }
                },
            }
        ],
    }
    ans_json = json.dumps(trust, indent=2)
    note = "Without pinning `:sub`, **any GitHub repository on earth** can mint an OIDC token with `aud=sts.amazonaws.com` and assume the AWS role (`CWE-284`)."
    ans = f"### OIDC Claim Pinning\n- {note}\n\n```json\n{ans_json}\n```" if style == "step_by_step" else f"{note}\n```json\n{ans_json}\n```"

    return make_record(
        record_id=f"td-oidc-{idx:07d}",
        domain="ethical_security",
        category="secrets_management_and_ci_supply_chain",
        difficulty="hard",
        style=style,
        tags=["oidc", "supply-chain", "iam"],
        quality_check="json_and_policy_verified",
        messages=[
            {"role": "system", "content": pick_security_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans},
        ],
    )


# ---------------------------------------------------------------------------
# 46. web_security_headers_and_csp
# ---------------------------------------------------------------------------

def gen_web_security_headers_and_csp(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    org = rng.choice(TENANT_SLUGS)
    env = rng.choice(ENVIRONMENTS)
    style = rng.choice(["direct", "step_by_step"])
    hsts_age = rng.choice([31536000, 63072000])

    user_q = (
        f"Write a Python middleware helper `security_headers(nonce: str) -> dict[str, str]` for `{svc_name}` "
        f"(`{org}`, `{env}`, HSTS `max-age={hsts_age}`, req `{idx + 100}`) enforcing a strict nonce-based CSP (no `'unsafe-inline'`), clickjacking defense, and MIME-sniff blocking."
    )
    code = (
        f"def security_headers(nonce: str) -> dict[str, str]:\n"
        "    csp = (\n"
        "        \"default-src 'none'; \"\n"
        "        f\"script-src 'nonce-{nonce}' 'strict-dynamic'; \"\n"
        "        \"style-src 'self'; img-src 'self' data:; connect-src 'self'; \"\n"
        "        \"frame-ancestors 'none'; base-uri 'none'\"\n"
        "    )\n"
        "    return {\n"
        "        'Content-Security-Policy': csp,\n"
        f"        'Strict-Transport-Security': 'max-age={hsts_age}; includeSubDomains',\n"
        "        'X-Content-Type-Options': 'nosniff',\n"
        "        'Referrer-Policy': 'strict-origin-when-cross-origin',\n"
        "    }"
    )
    note = "Per-response cryptographic nonces + `frame-ancestors 'none'` + `base-uri 'none'` neutralize reflected/stored XSS and base-tag hijacking."
    ans = f"### Defense-in-Depth Headers\n- {note}\n\n```python\n{code}\n```" if style == "step_by_step" else f"```python\n{code}\n```\n- {note}"

    return make_record(
        record_id=f"td-csp-{idx:07d}",
        domain="ethical_security",
        category="web_security_headers_and_csp",
        difficulty="medium",
        style=style,
        tags=["csp", "web-security", "headers"],
        quality_check="ast_and_cwe_verified",
        messages=[
            {"role": "system", "content": pick_security_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans},
        ],
    )
