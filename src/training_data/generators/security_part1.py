"""Ethical Security Generators Part 1 (Categories 13–18): AppSec Remediation, Crypto/JWT, CVSS v3.1, Sigma/Semgrep, DFIR, Network/ACL."""

from __future__ import annotations

import random
from typing import Any

from training_data.schema import make_record, pick_security_system
from training_data.solvers import (
    analyze_ipv4_cidr,
    compute_cvss31,
    evaluate_firewall_rules,
)
from training_data.vocab import (
    DOCUMENTATION_TEST_IPS,
    ENVIRONMENTS,
    FILE_PATHS,
    SAFE_INTERNAL_IPS,
    SECURITY_OPENERS_DIRECT,
    SECURITY_OPENERS_STEP,
    SERVICES,
    TENANT_SLUGS,
    USERNAMES,
)


# ---------------------------------------------------------------------------
# 13. secure_code_remediation_web (AST-Verified Patched Code!)
# ---------------------------------------------------------------------------

def gen_secure_code_remediation_web(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    fpath = rng.choice(FILE_PATHS)
    style = rng.choice(["direct", "step_by_step", "step_by_step"])
    opener = rng.choice(SECURITY_OPENERS_DIRECT if style == "direct" else SECURITY_OPENERS_STEP)
    variant = idx % 6

    if variant == 0:
        difficulty = "easy"
        cwe = "CWE-89 (SQL Injection / OWASP A03:2021)"
        vuln = (
            f"def get_{domain_ctx}_row(conn, {id_col}: str, tenant: str):\n"
            f"    q = f\"SELECT * FROM {table_ctx} WHERE {id_col} = '{{{id_col}}}' AND tenant = '{{tenant}}'\"\n"
            f"    return conn.execute(q).fetchone()"
        )
        patch = (
            "from typing import Any\n\n"
            f"def get_{domain_ctx}_row(conn: Any, {id_col}: str, tenant: str) -> Any:\n"
            f"    query = 'SELECT * FROM {table_ctx} WHERE {id_col} = ? AND tenant = ?'\n"
            f"    return conn.execute(query, ({id_col}, tenant)).fetchone()"
        )
        root = f"Interpolating `{id_col}` and `tenant` into SQL via f-strings lets untrusted quotes break out of string literals and alter query structure."
    elif variant == 1:
        difficulty = "hard"
        cwe = "CWE-918 (Server-Side Request Forgery / OWASP API7:2023)"
        port = rng.choice([443, 8443])
        vuln = (
            f"import urllib.request\n\n"
            f"def probe_webhook(target_url: str) -> int:\n"
            f"    with urllib.request.urlopen(target_url, timeout=3) as resp:\n"
            f"        return resp.status"
        )
        patch = (
            "import ipaddress\nimport socket\nimport urllib.parse\nimport urllib.request\n\n"
            "def validate_public_https(url: str) -> str:\n"
            "    parsed = urllib.parse.urlsplit(url)\n"
            f"    if parsed.scheme != 'https' or parsed.port not in (None, {port}) or not parsed.hostname:\n"
            "        raise ValueError('disallowed scheme or port')\n"
            "    for info in socket.getaddrinfo(parsed.hostname, None):\n"
            "        ip = ipaddress.ip_address(info[4][0])\n"
            "        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:\n"
            "            raise ValueError('non-public destination blocked')\n"
            "    return url"
        )
        root = "Unvalidated `urlopen(target_url)` permits `file://`, loopback (`127.0.0.1`), or cloud link-local metadata (`169.254.169.254`) access."
    elif variant == 2:
        difficulty = "medium"
        cwe = "CWE-22 (Path Traversal)"
        base_dir = f"/var/lib/{svc_name}/exports"
        vuln = (
            f"from pathlib import Path\n\n"
            f"def read_export(filename: str) -> bytes:\n"
            f"    return Path('{base_dir}/' + filename).read_bytes()"
        )
        patch = (
            "from pathlib import Path\n\n"
            f"BASE_DIR = Path('{base_dir}').resolve()\n\n"
            "def read_export(filename: str) -> bytes:\n"
            "    candidate = (BASE_DIR / filename).resolve()\n"
            "    if not candidate.is_relative_to(BASE_DIR) or not candidate.is_file():\n"
            "        raise PermissionError('path outside export root')\n"
            "    return candidate.read_bytes()"
        )
        root = f"String concatenation allows `../` segments or symlinks to escape `{base_dir}` and read arbitrary host files."
    elif variant == 3:
        difficulty = "medium"
        cwe = "CWE-78 (OS Command Injection)"
        vuln = (
            f"import subprocess\n\n"
            f"def verify_archive(archive_name: str) -> int:\n"
            f"    return subprocess.call(f'gzip -t /srv/{domain_ctx}/{{archive_name}}', shell=True)"
        )
        patch = (
            "from pathlib import Path\nimport subprocess\n\n"
            f"ARCHIVE_DIR = Path('/srv/{domain_ctx}').resolve()\n\n"
            "def verify_archive(archive_name: str) -> int:\n"
            "    target = (ARCHIVE_DIR / Path(archive_name).name).resolve()\n"
            "    if not target.is_relative_to(ARCHIVE_DIR):\n"
            "        raise ValueError('invalid archive path')\n"
            "    proc = subprocess.run(['gzip', '-t', '--', str(target)], shell=False, check=False)\n"
            "    return proc.returncode"
        )
        root = "`shell=True` passes `archive_name` through `/bin/sh`, interpreting `;`, `|`, and `$()` metacharacters as arbitrary commands."
    elif variant == 4:
        difficulty = "medium"
        cwe = "CWE-639 (IDOR / Broken Object Level Authorization)"
        vuln = (
            f"def delete_{domain_ctx}_item(db, {id_col}: str, caller_tenant: str) -> bool:\n"
            f"    cur = db.execute('DELETE FROM {table_ctx} WHERE {id_col} = ?', ({id_col},))\n"
            f"    return cur.rowcount > 0"
        )
        patch = (
            "from typing import Any\n\n"
            f"def delete_{domain_ctx}_item(db: Any, {id_col}: str, caller_tenant: str) -> bool:\n"
            "    if not caller_tenant:\n"
            "        raise PermissionError('missing authenticated tenant')\n"
            f"    cur = db.execute(\n"
            f"        'DELETE FROM {table_ctx} WHERE {id_col} = ? AND tenant_id = ?',\n"
            f"        ({id_col}, caller_tenant),\n"
            "    )\n"
            "    return bool(cur.rowcount > 0)"
        )
        root = f"`caller_tenant` is accepted by the handler signature but omitted from the `DELETE` predicate, allowing cross-tenant deletion by `{id_col}`."
    else:
        difficulty = "easy"
        cwe = "CWE-79 (Reflected Cross-Site Scripting)"
        vuln = (
            f"def render_search_banner(query: str) -> str:\n"
            f"    return f'<div class=\"{domain_ctx}-banner\">Results for: {{query}}</div>'"
        )
        patch = (
            "import html\n\n"
            "def render_search_banner(query: str) -> str:\n"
            "    safe_q = html.escape(query, quote=True)\n"
            f"    return f'<div class=\"{domain_ctx}-banner\">Results for: {{safe_q}}</div>'"
        )
        root = "Unescaped HTML interpolation reflects `<script>` or attribute event handlers directly into the response DOM."

    line_ref = rng.randint(10, 850)
    user_q = f"{opener}\nService `{svc_name}` (`{fpath}:{line_ref}`, `{env}`):\n```python\n{vuln}\n```"
    ans_text = (
        f"### Vulnerability Analysis\n- **Classification**: {cwe}\n- **Root Cause**: {root}\n\n"
        f"### Remediated Code\n```python\n{patch}\n```"
        if style == "step_by_step"
        else f"**{cwe}** — {root}\n```python\n{patch}\n```"
    )

    return make_record(
        record_id=f"td-appsec-{idx:06d}",
        domain="ethical_security",
        category="secure_code_remediation_web",
        difficulty=difficulty,
        style=style,
        tags=["appsec", "remediation", "owasp"],
        quality_check="ast_and_cwe_verified",
        messages=[
            {"role": "system", "content": pick_security_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans_text},
        ],
    )


# ---------------------------------------------------------------------------
# 14. auth_crypto_jwt_security
# ---------------------------------------------------------------------------

def gen_auth_crypto_jwt_security(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    style = rng.choice(["direct", "step_by_step"])
    opener = rng.choice(SECURITY_OPENERS_DIRECT if style == "direct" else SECURITY_OPENERS_STEP)
    variant = idx % 5

    if variant == 0:
        difficulty = "easy"
        hdr = f"X-{domain_ctx.capitalize()}-Signature"
        vuln = (
            f"import hashlib, hmac\n\n"
            f"def check_sig(secret: bytes, body: bytes, sig_hex: str) -> bool:\n"
            f"    mac = hmac.new(secret, body, hashlib.sha256).hexdigest()\n"
            f"    return mac == sig_hex"
        )
        patch = (
            "import hashlib\nimport hmac\n\n"
            "def check_sig(secret: bytes, body: bytes, sig_hex: str) -> bool:\n"
            "    expected = hmac.new(secret, body, hashlib.sha256).hexdigest()\n"
            "    return hmac.compare_digest(expected, sig_hex.strip().lower())"
        )
        explanation = f"**CWE-208 (Observable Timing Discrepancy)**: Standard `==` short-circuits on the first mismatched byte of `{hdr}`, leaking prefix timing. `hmac.compare_digest` executes in constant time."
    elif variant == 1:
        difficulty = "medium"
        aud = f"https://api.{svc_name}.internal"
        iss = f"https://idp.{env}.internal"
        vuln = (
            "import jwt\n\n"
            "def parse_token(token: str, pub_key: str) -> dict:\n"
            "    return jwt.decode(token, options={'verify_signature': False})"
        )
        patch = (
            "from typing import Any\nimport jwt\n\n"
            "def parse_token(token: str, pub_key: str) -> dict[str, Any]:\n"
            "    return dict(\n"
            "        jwt.decode(\n"
            "            token,\n"
            "            key=pub_key,\n"
            "            algorithms=['RS256'],\n"
            f"            audience='{aud}',\n"
            f"            issuer='{iss}',\n"
            "            options={'require': ['exp', 'iat', 'sub', 'aud', 'iss']},\n"
            "        )\n"
            "    )"
        )
        explanation = "**CWE-347 (Improper Verification of Cryptographic Signature)**: Disabling signature checks accepts forged claims; pin `algorithms=['RS256']`, `audience`, `issuer`, and mandatory `exp`/`sub` claims."
    elif variant == 2:
        difficulty = "medium"
        n_bytes = rng.choice([32, 48])
        vuln = (
            "import random\n\n"
            f"def issue_reset_code() -> str:\n"
            f"    return ''.join(random.choices('0123456789abcdef', k={n_bytes}))"
        )
        patch = (
            "import secrets\n\n"
            "def issue_reset_code() -> str:\n"
            f"    return secrets.token_urlsafe({n_bytes})"
        )
        explanation = "**CWE-338 (Weak PRNG)**: `random` uses Mersenne Twister, whose internal state is recoverable from observed outputs. Use OS CSPRNG via `secrets.token_urlsafe`."
    elif variant == 3:
        difficulty = "hard"
        iters = rng.choice([600_000, 650_000, 700_000])
        vuln = (
            "import hashlib\n\n"
            "def hash_password(pw: str) -> str:\n"
            f"    return hashlib.sha256(('{svc_name}:' + pw).encode()).hexdigest()"
        )
        patch = (
            "import hashlib\nimport hmac\nimport secrets\n\n"
            f"PBKDF2_ITERS = {iters}\n\n"
            "def hash_password(pw: str) -> str:\n"
            "    salt = secrets.token_bytes(16)\n"
            "    dk = hashlib.pbkdf2_hmac('sha256', pw.encode('utf-8'), salt, PBKDF2_ITERS)\n"
            "    return f'pbkdf2_sha256${PBKDF2_ITERS}${salt.hex()}${dk.hex()}'\n\n"
            "def verify_password(pw: str, stored: str) -> bool:\n"
            "    _, iters_s, salt_hex, dk_hex = stored.split('$')\n"
            "    cand = hashlib.pbkdf2_hmac('sha256', pw.encode('utf-8'), bytes.fromhex(salt_hex), int(iters_s))\n"
            "    return hmac.compare_digest(cand.hex(), dk_hex)"
        )
        explanation = f"**CWE-916 (Insufficient Password Hash Effort)**: Static-salt single-round SHA-256 allows billions of GPU guesses/sec. Use per-user 128-bit CSPRNG salts + Argon2id or PBKDF2-HMAC-SHA256 (`{iters}` iterations, OWASP minimum)."
    else:
        difficulty = "hard"
        vuln = (
            "import base64, hashlib\n\n"
            "def verify_pkce(verifier: str, challenge: str, method: str) -> bool:\n"
            "    if method == 'plain':\n"
            "        return verifier == challenge\n"
            "    return False"
        )
        patch = (
            "import base64\nimport hashlib\nimport hmac\n\n"
            "def verify_pkce_s256(verifier: str, challenge: str) -> bool:\n"
            "    if not (43 <= len(verifier) <= 128):\n"
            "        return False\n"
            "    digest = hashlib.sha256(verifier.encode('ascii')).digest()\n"
            "    computed = base64.urlsafe_b64encode(digest).decode('ascii').rstrip('=')\n"
            "    return hmac.compare_digest(computed, challenge)"
        )
        explanation = "**CWE-345 / RFC 7636 Security**: Allowing `plain` PKCE exposes the verifier if the authorization code is intercepted. Enforce `S256` only, RFC 7636 length bounds (`43..128`), and constant-time comparison."

    fpath = rng.choice(FILE_PATHS)
    line_ref = rng.randint(10, 850)
    user_q = f"{opener}\nAudit this `{svc_name}` (`{fpath}:{line_ref}`, `{env}`) authentication/crypto helper:\n```python\n{vuln}\n```"
    ans_text = (
        f"### Cryptographic Review\n- {explanation}\n\n```python\n{patch}\n```"
        if style == "step_by_step"
        else f"{explanation}\n```python\n{patch}\n```"
    )

    return make_record(
        record_id=f"td-crypto-{idx:06d}",
        domain="ethical_security",
        category="auth_crypto_jwt_security",
        difficulty=difficulty,
        style=style,
        tags=["cryptography", "authentication", "appsec"],
        quality_check="ast_and_cwe_verified",
        messages=[
            {"role": "system", "content": pick_security_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans_text},
        ],
    )


# ---------------------------------------------------------------------------
# 15. cvss_triage_and_vuln_reporting (100% Exact Official CVSS v3.1 Math!)
# ---------------------------------------------------------------------------

def gen_cvss_triage_and_vuln_reporting(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    style = rng.choice(["direct", "step_by_step"])

    av = rng.choice(["N", "A", "L", "P"])
    ac = rng.choice(["L", "H"])
    pr = rng.choice(["N", "L", "H"])
    ui = rng.choice(["N", "R"])
    s = rng.choice(["U", "C"])
    c = rng.choice(["N", "L", "H"])
    i = rng.choice(["N", "L", "H"])
    a = rng.choice(["N", "L", "H"])
    if c == "N" and i == "N" and a == "N":
        c = "H"

    res = compute_cvss31(av, ac, pr, ui, s, c, i, a)
    lbl = res["labels"]
    difficulty = "easy" if s == "U" and ac == "L" else ("hard" if s == "C" else "medium")
    finding_id = f"VULN-2026-{idx + 1000}"

    sla_map = {
        "Critical": "24–48 hours (emergency hotfix + compensating WAF/ACL rule)",
        "High": "7 days (prioritized sprint patch)",
        "Medium": "30 days (scheduled release hardening)",
        "Low": "90 days (backlog defense-in-depth)",
    }
    sla = sla_map.get(res["severity"], "90 days")

    user_q = (
        f"Compute the exact CVSS v3.1 vector, base score, severity rating, and remediation SLA for authorized assessment finding `{finding_id}` on `{svc_name}` (`{env}`):\n"
        f"- Attack Vector: **{lbl['AV']}** | Attack Complexity: **{lbl['AC']}**\n"
        f"- Privileges Required: **{lbl['PR']}** | User Interaction: **{lbl['UI']}** | Scope: **{lbl['S']}**\n"
        f"- Confidentiality: **{lbl['C']}** | Integrity: **{lbl['I']}** | Availability: **{lbl['A']}**"
    )

    if style == "step_by_step":
        ans_text = (
            f"### CVSS v3.1 Calculation (`{finding_id}`)\n"
            f"1. **Vector**: `{res['vector']}`\n"
            f"2. **Sub-scores**: `ISS = {res['iss']}`, `Impact = {res['impact']}`, `Exploitability = {res['exploitability']}` "
            f"({'Scope Changed 1.08x multiplier applied' if s == 'C' else 'Scope Unchanged'}).\n"
            f"3. **Base Score**: **{res['score']} ({res['severity']})**\n"
            f"4. **Recommended SLA**: {sla}."
        )
    else:
        ans_text = (
            f"- **Vector**: `{res['vector']}`\n"
            f"- **Base Score**: **{res['score']} ({res['severity']})** (`Impact={res['impact']}`, `Exploitability={res['exploitability']}`)\n"
            f"- **Remediation SLA**: {sla}."
        )

    return make_record(
        record_id=f"td-cvss-{idx:06d}",
        domain="ethical_security",
        category="cvss_triage_and_vuln_reporting",
        difficulty=difficulty,
        style=style,
        tags=["cvss-v3.1", "vulnerability-triage", "appsec"],
        quality_check="cvss31_exact_math",
        messages=[
            {"role": "system", "content": pick_security_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans_text},
        ],
    )


# ---------------------------------------------------------------------------
# 16. detection_engineering_sigma_semgrep
# ---------------------------------------------------------------------------

def gen_detection_engineering_sigma_semgrep(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    style = rng.choice(["direct", "step_by_step"])
    variant = idx % 4
    rule_num = idx + 1000

    if variant == 0:
        difficulty = "medium"
        user_q = (
            f"Write a Semgrep YAML rule (`{svc_name}-unsafe-yaml-{rule_num}`) to detect `yaml.load(...)` in Python "
            f"that lacks `Loader=yaml.SafeLoader` (or `CSafeLoader`), avoiding false positives on safe calls."
        )
        yaml_rule = (
            "rules:\n"
            f"  - id: {svc_name}-unsafe-yaml-{rule_num}\n"
            "    languages: [python]\n"
            "    severity: ERROR\n"
            "    message: \"Use yaml.safe_load() or Loader=yaml.SafeLoader to prevent CWE-502 deserialization.\"\n"
            "    patterns:\n"
            "      - pattern: yaml.load($DATA, ...)\n"
            "      - pattern-not: yaml.load($DATA, Loader=yaml.SafeLoader, ...)\n"
            "      - pattern-not: yaml.load($DATA, Loader=yaml.CSafeLoader, ...)"
        )
        note = "`pattern-not` clauses whitelist `SafeLoader`/`CSafeLoader` so CI pipelines only block arbitrary-object deserialization."
    elif variant == 1:
        difficulty = "medium"
        user_q = (
            f"Write a Semgrep rule (`{svc_name}-subproc-shell-{rule_num}`) flagging Python `subprocess` calls with `shell=True` "
            f"when the command argument is non-literal (CWE-78)."
        )
        yaml_rule = (
            "rules:\n"
            f"  - id: {svc_name}-subproc-shell-{rule_num}\n"
            "    languages: [python]\n"
            "    severity: ERROR\n"
            "    message: \"Avoid shell=True with dynamic input (CWE-78); pass an argument list with shell=False.\"\n"
            "    patterns:\n"
            "      - pattern-either:\n"
            "          - pattern: subprocess.run($CMD, ..., shell=True, ...)\n"
            "          - pattern: subprocess.Popen($CMD, ..., shell=True, ...)\n"
            '      - pattern-not: subprocess.run("...", ..., shell=True, ...)\n'
            '      - pattern-not: subprocess.Popen("...", ..., shell=True, ...)'
        )
        note = "Excludes constant string literals via `pattern-not: subprocess.run(\"...\")` to eliminate noise on static commands."
    elif variant == 2:
        difficulty = "hard"
        ci_user = rng.choice(USERNAMES)
        user_q = (
            f"Write a Sigma detection rule (`{svc_name}-{rule_num}`, `{env}`) alerting on AWS CloudTrail `AttachUserPolicy` or `PutUserPolicy` "
            f"granting `AdministratorAccess`, excluding automation role `{ci_user}` (MITRE ATT&CK `T1098`)."
        )
        yaml_rule = (
            f"title: AWS IAM Admin Policy Attachment ({svc_name} {rule_num})\n"
            "status: stable\n"
            "logsource:\n"
            "  product: aws\n"
            "  service: cloudtrail\n"
            "detection:\n"
            "  selection:\n"
            "    eventSource: iam.amazonaws.com\n"
            "    eventName:\n"
            "      - AttachUserPolicy\n"
            "      - PutUserPolicy\n"
            "    requestParameters.policyArn|contains: AdministratorAccess\n"
            "  filter_ci:\n"
            f"    userIdentity.arn|contains: '{ci_user}'\n"
            "  condition: selection and not filter_ci\n"
            "level: high\n"
            "tags:\n"
            "  - attack.persistence\n"
            "  - attack.t1098"
        )
        note = f"Correlates IAM policy attachment events while filtering approved provisioning actor `{ci_user}`."
    else:
        difficulty = "medium"
        user_q = (
            f"Write a Sigma rule (`{svc_name}-{rule_num}`) for Linux `process_creation` logs detecting suspicious reverse-shell "
            f"spawns (`bash -i`, `nc -e`, `/dev/tcp/`) from web worker user `www-data` (ATT&CK `T1059.004`)."
        )
        yaml_rule = (
            f"title: Web Worker Interactive Shell Spawn ({svc_name} {rule_num})\n"
            "status: experimental\n"
            "logsource:\n"
            "  category: process_creation\n"
            "  product: linux\n"
            "detection:\n"
            "  selection_user:\n"
            "    User: www-data\n"
            "  selection_cmd:\n"
            "    CommandLine|contains:\n"
            "      - 'bash -i'\n"
            "      - '/dev/tcp/'\n"
            "      - 'nc -e'\n"
            "  condition: selection_user and selection_cmd\n"
            "level: critical\n"
            "tags:\n"
            "  - attack.execution\n"
            "  - attack.t1059.004"
        )
        note = "Scoping `User: www-data` alongside interactive/socket flags yields near-zero false positives in production containers."

    ans_text = (
        f"### Detection Logic & Tuning\n- {note}\n\n```yaml\n{yaml_rule}\n```"
        if style == "step_by_step"
        else f"```yaml\n{yaml_rule}\n```\n- {note}"
    )

    return make_record(
        record_id=f"td-detect-{idx:06d}",
        domain="ethical_security",
        category="detection_engineering_sigma_semgrep",
        difficulty=difficulty,
        style=style,
        tags=["detection-engineering", "semgrep", "sigma"],
        quality_check="yaml_detection_verified",
        messages=[
            {"role": "system", "content": pick_security_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans_text},
        ],
    )


# ---------------------------------------------------------------------------
# 17. log_forensics_incident_response (Ground-Truth Simulated Telemetry!)
# ---------------------------------------------------------------------------

def gen_log_forensics_incident_response(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    style = rng.choice(["direct", "step_by_step"])
    bad_ip = rng.choice(DOCUMENTATION_TEST_IPS)
    ok_ip1, ok_ip2 = rng.sample(SAFE_INTERNAL_IPS, 2)
    actor = rng.choice(USERNAMES)
    hr = rng.randint(1, 22)
    mn = rng.randint(10, 49)
    variant = idx % 3

    if variant == 0:
        difficulty = "medium"
        ts_bad = f"2026-10-18T{hr:02d}:{mn:02d}:19Z"
        logs = (
            f"2026-10-18T{hr:02d}:{mn:02d}:04Z {ok_ip1} GET /api/v1/{table_ctx}?limit=20 200 18ms\n"
            f"2026-10-18T{hr:02d}:{mn:02d}:11Z {ok_ip2} POST /api/v1/{table_ctx} 201 34ms\n"
            f"{ts_bad} {bad_ip} GET /api/v1/export?file=../../../../etc/passwd 403 4ms\n"
            f"2026-10-18T{hr:02d}:{mn:02d}:20Z {bad_ip} GET /api/v1/export?file=..%2f..%2fproc%2fself%2fenviron 200 9ms"
        )
        user_q = f"Triage these `{svc_name}` (`{env}`) HTTP logs, identify the successful intrusion indicator, and list immediate containment steps:\n```text\n{logs}\n```"
        finding = (
            f"- **Compromise Indicator**: Source `{bad_ip}` probed path traversal (`CWE-22`) at `{ts_bad}` (`403`), then bypassed naive filtering using URL-encoded `..%2f..%2fproc%2fself%2fenviron` (`HTTP 200` at `2026-10-18T{hr:02d}:{mn:02d}:20Z`), exposing process environment secrets.\n"
            f"- **Containment**: (1) Rotate all credentials/API keys present in `{svc_name}` process env immediately; (2) Block `{bad_ip}` at edge WAF; (3) Patch `/api/v1/export` to URL-decode once and verify `Path.resolve().is_relative_to(BASE_DIR)`."
        )
    elif variant == 1:
        difficulty = "hard"
        ssh_port = rng.randint(41000, 61000)
        ts_succ = f"2026-10-19T{hr:02d}:{mn:02d}:44Z"
        logs = (
            f"2026-10-19T{hr:02d}:{mn:02d}:38Z sshd[4120]: Failed password for {actor} from {bad_ip} port {ssh_port} ssh2\n"
            f"2026-10-19T{hr:02d}:{mn:02d}:40Z sshd[4122]: Failed password for {actor} from {bad_ip} port {ssh_port + 1} ssh2\n"
            f"{ts_succ} sshd[4129]: Accepted password for {actor} from {bad_ip} port {ssh_port + 4} ssh2\n"
            f"2026-10-19T{hr:02d}:{mn:02d}:51Z sudo: {actor} : TTY=pts/1 ; PWD=/tmp ; USER=root ; COMMAND=/usr/bin/crontab -l"
        )
        user_q = f"Analyze this `{env}` bastion `auth.log` excerpt from `{svc_name}` infrastructure and specify DFIR actions:\n```text\n{logs}\n```"
        finding = (
            f"- **Timeline & Root Cause**: External IP `{bad_ip}` brute-forced SSH password auth (`T1110.001`), authenticated as `{actor}` at `{ts_succ}`, and escalated via `sudo` to inspect `root` crontabs (`T1053.003`).\n"
            f"- **Containment & Hardening**: (1) Terminate active `pts/1` session and lock `{actor}`; (2) Audit `/var/spool/cron` and `/etc/cron*` for persistence; (3) Disable SSH password login (`PasswordAuthentication no`, key + hardware MFA only) and restrict SSH ingress CIDR."
        )
    else:
        difficulty = "hard"
        key_id = f"AKIA{rng.randint(100000, 999999)}PROD"
        ts_evt = f"2026-10-20T{hr:02d}:{mn:02d}:09Z"
        logs = (
            f'{{"ts":"2026-10-20T{hr:02d}:{mn:02d}:01Z","actor":"{actor}","ip":"{ok_ip1}","event":"GetObject","bucket":"{svc_name}-cfg"}}\n'
            f'{{"ts":"{ts_evt}","actor":"{key_id}","ip":"{bad_ip}","event":"CreateAccessKey","targetUser":"{actor}"}}\n'
            f'{{"ts":"2026-10-20T{hr:02d}:{mn:02d}:14Z","actor":"{key_id}","ip":"{bad_ip}","event":"DeleteTrail","trail":"{env}-audit"}}'
        )
        user_q = f"Review these CloudTrail events for `{svc_name}` (`{env}`) and determine what occurred and how to respond:\n```json\n{logs}\n```"
        finding = (
            f"- **Incident Triage**: Compromised IAM credential `{key_id}` from external IP `{bad_ip}` minted a persistence access key for `{actor}` (`CreateAccessKey` at `{ts_evt}`, `T1098.001`) and attempted defense evasion via `DeleteTrail` (`{env}-audit`, `T1562.008`).\n"
            f"- **Response**: (1) Deactivate `{key_id}` and the newly created access key on `{actor}`; (2) Re-enable `{env}-audit` CloudTrail with Organization SCP `Deny` on `cloudtrail:DeleteTrail`/`StopLogging`; (3) Hunt all API calls from `{bad_ip}` across regions."
        )

    ans_text = f"### DFIR Analysis\n{finding}" if style == "step_by_step" else finding

    return make_record(
        record_id=f"td-dfir-{idx:06d}",
        domain="ethical_security",
        category="log_forensics_incident_response",
        difficulty=difficulty,
        style=style,
        tags=["dfir", "log-analysis", "incident-response"],
        quality_check="dfir_ground_truth_verified",
        messages=[
            {"role": "system", "content": pick_security_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans_text},
        ],
    )


# ---------------------------------------------------------------------------
# 18. network_acl_firewall_analysis (100% Verified via Python ipaddress!)
# ---------------------------------------------------------------------------

def gen_network_acl_firewall_analysis(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    style = rng.choice(["direct", "step_by_step"])
    variant = idx % 2

    if variant == 0:
        oct2 = rng.randint(16, 31)
        oct3 = rng.randint(0, 250)
        prefix = rng.choice([24, 25, 26, 27, 28, 29])
        cidr_in = f"172.{oct2}.{oct3}.0/{prefix}"
        info = analyze_ipv4_cidr(cidr_in)
        difficulty = "easy" if prefix in (24, 26) else "medium"
        user_q = (
            f"Calculate the exact network address, broadcast address, netmask, usable host count, and usable host range "
            f"for `{svc_name}` (`{env}`) subnet `{info['canonical']}`."
        )
        ans_text = (
            f"### Subnet Breakdown (`{info['canonical']}`)\n"
            f"- **Netmask**: `{info['netmask']}` (`/{info['prefixlen']}`, `{32 - info['prefixlen']}` host bits = `{info['total_addresses']}` total IPs)\n"
            f"- **Network / Broadcast**: `{info['network']}` / `{info['broadcast']}`\n"
            f"- **Usable Hosts**: **{info['usable_hosts']}** (`{info['first_usable']}` – `{info['last_usable']}`)"
            if style == "step_by_step"
            else (
                f"- **Network**: `{info['network']}` | **Broadcast**: `{info['broadcast']}` | **Netmask**: `{info['netmask']}`\n"
                f"- **Usable Hosts**: **{info['usable_hosts']}** (`{info['first_usable']}` – `{info['last_usable']}`)"
            )
        )
    else:
        difficulty = "hard"
        oct2 = rng.randint(10, 99)
        port = rng.choice([443, 5432, 6379, 8443, 9090])
        probe_host = rng.randint(5, 60)
        probe_ip = f"10.{oct2}.4.{probe_host}"
        rules = [
            {"id": 10, "action": rng.choice(["DENY", "ALLOW"]), "src": f"10.{oct2}.0.0/16", "port": port, "proto": "TCP"},
            {"id": 20, "action": "ALLOW", "src": f"10.{oct2}.4.0/24", "port": port, "proto": "TCP"},
            {"id": 30, "action": "DENY", "src": "0.0.0.0/0", "port": "ANY", "proto": "ANY"},
        ]
        verdict, matched_id, shadowed = evaluate_firewall_rules(rules, probe_ip, port, "TCP")
        rules_txt = "\n".join(
            f"Rule {r['id']}: {r['action']} {r['proto']} src={r['src']} dst_port={r['port']}" for r in rules
        )
        user_q = (
            f"Audit this ordered first-match ACL on `{svc_name}` (`{env}`):\n```text\n{rules_txt}\n```\n"
            f"1. What is the verdict for `TCP` packet from `{probe_ip}` to port `{port}`?\n"
            f"2. Identify any shadowed rule and how to fix rule ordering."
        )
        sh_pair = shadowed[0]
        ans_text = (
            f"### ACL Evaluation & Shadow Analysis\n"
            f"1. **Packet Verdict**: **{verdict}** (matches **Rule {matched_id}**, since `{probe_ip}` falls inside `10.{oct2}.0.0/16` evaluated first).\n"
            f"2. **Shadowed Rule**: **Rule {sh_pair[0]}** (`10.{oct2}.4.0/24`) is completely shadowed by broader **Rule {sh_pair[1]}** (`10.{oct2}.0.0/16`) on the same protocol/port.\n"
            f"3. **Fix**: Place specific `/24` exceptions (`Rule {sh_pair[0]}`) **before** broader `/16` summary rules (`Rule {sh_pair[1]}`)."
        )

    return make_record(
        record_id=f"td-netsec-{idx:06d}",
        domain="ethical_security",
        category="network_acl_firewall_analysis",
        difficulty=difficulty,
        style=style,
        tags=["network-security", "cidr", "firewall-audit"],
        quality_check="ipaddress_verified",
        messages=[
            {"role": "system", "content": pick_security_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans_text},
        ],
    )
