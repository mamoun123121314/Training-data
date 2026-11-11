"""Ethical Security Generators Part 2 (Categories 19–23): Cloud/K8s Hardening, STRIDE, Pentest Scoping, OPA/Rego, Safety Boundaries."""

from __future__ import annotations

import ipaddress
import json
import random
from typing import Any

from training_data.schema import make_record, pick_security_system
from training_data.vocab import (
    DOCUMENTATION_TEST_IPS,
    ENVIRONMENTS,
    REVIEW_CONTEXTS,
    SAFE_INTERNAL_IPS,
    SERVICES,
    TENANT_SLUGS,
)


# ---------------------------------------------------------------------------
# 19. cloud_k8s_container_hardening
# ---------------------------------------------------------------------------

def gen_cloud_k8s_container_hardening(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    style = rng.choice(["direct", "step_by_step"])
    variant = idx % 3
    acct_id = f"{rng.randint(100000000000, 999999999999)}"

    if variant == 0:
        difficulty = "medium"
        bucket = f"{svc_name}-{env}-artifacts"
        prefix = f"{domain_ctx}/exports"
        over_perm = '{"Effect":"Allow","Action":"s3:*","Resource":"*"}'
        least_priv = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "ScopedPrefixReadWrite",
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:PutObject"],
                    "Resource": f"arn:aws:s3:::{bucket}/{prefix}/*",
                    "Condition": {"Bool": {"aws:SecureTransport": "true"}},
                }
            ],
        }
        user_q = (
            f"Hardening review for `{svc_name}` (`{env}`, AWS account `{acct_id}`): worker only reads/writes objects under "
            f"`s3://{bucket}/{prefix}/*`, yet its IAM statement is `{over_perm}`. Rewrite to strict least privilege."
        )
        ans_body = json.dumps(least_priv, indent=2)
        ans_text = (
            f"### Least-Privilege Fixes\n1. Scope actions to `s3:GetObject`/`s3:PutObject` (drop `s3:DeleteBucket`, `s3:PutBucketPolicy`).\n"
            f"2. Restrict ARN to `arn:aws:s3:::{bucket}/{prefix}/*` and enforce TLS via `aws:SecureTransport`.\n\n```json\n{ans_body}\n```"
            if style == "step_by_step"
            else f"```json\n{ans_body}\n```\n- Scoped to object ARN prefix + TLS-only transport."
        )
    elif variant == 1:
        difficulty = "medium"
        uid = rng.randint(10001, 29999)
        user_q = (
            f"Harden a Kubernetes container `securityContext` for `{svc_name}` in namespace `{domain_ctx}` (`{env}`) "
            f"running under UID `{uid}` compliant with Pod Security Standards `restricted`."
        )
        manifest = (
            f"name: {svc_name}\n"
            "securityContext:\n"
            "  allowPrivilegeEscalation: false\n"
            "  readOnlyRootFilesystem: true\n"
            "  runAsNonRoot: true\n"
            f"  runAsUser: {uid}\n"
            "  seccompProfile:\n"
            "    type: RuntimeDefault\n"
            "  capabilities:\n"
            "    drop: [\"ALL\"]"
        )
        ans_text = (
            f"### Restricted Pod Security Controls\nDrops all Linux capabilities, blocks `setuid` escalation, mounts `/` read-only, and enables `RuntimeDefault` seccomp.\n\n```yaml\n{manifest}\n```"
            if style == "step_by_step"
            else f"```yaml\n{manifest}\n```\n- Satisfies Kubernetes `restricted` PSS profile."
        )
    else:
        difficulty = "hard"
        db_port = rng.choice([3306, 5432, 6379, 9042, 9200, 27017])
        dns_port = 53
        user_q = (
            f"Write a Kubernetes `NetworkPolicy` (`pol-{idx + 100}`) in namespace `{domain_ctx}` (`{env}`) isolating `{svc_name}` egress "
            f"strictly to pods labeled `role: {domain_ctx}-db` on TCP port `{db_port}` plus CoreDNS (`UDP/TCP {dns_port}`)."
        )
        manifest = (
            "apiVersion: networking.k8s.io/v1\nkind: NetworkPolicy\nmetadata:\n"
            f"  name: {svc_name}-egress-lock\n  namespace: {domain_ctx}\nspec:\n"
            f"  podSelector:\n    matchLabels:\n      app: {svc_name}\n"
            "  policyTypes: [Egress]\n  egress:\n"
            f"    - to:\n        - podSelector:\n            matchLabels:\n              role: {domain_ctx}-db\n"
            f"      ports:\n        - protocol: TCP\n          port: {db_port}\n"
            "    - to:\n        - namespaceSelector:\n            matchLabels:\n              kubernetes.io/metadata.name: kube-system\n"
            "      ports:\n        - protocol: UDP\n          port: 53\n        - protocol: TCP\n          port: 53"
        )
        ans_text = (
            f"### Microsegmentation\nDefault-denies internet/metadata egress from `{svc_name}`, permitting only `{domain_ctx}-db:{db_port}` and `kube-system` DNS.\n\n```yaml\n{manifest}\n```"
            if style == "step_by_step"
            else f"```yaml\n{manifest}\n```\n- Blocks unauthorized C2/metadata egress while preserving DB and DNS."
        )

    return make_record(
        record_id=f"td-cloudsec-{idx:06d}",
        domain="ethical_security",
        category="cloud_k8s_container_hardening",
        difficulty=difficulty,
        style=style,
        tags=["cloud-security", "kubernetes", "iam"],
        quality_check="json_and_policy_verified",
        messages=[
            {"role": "system", "content": pick_security_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans_text},
        ],
    )


# ---------------------------------------------------------------------------
# 20. threat_modeling_stride_arch
# ---------------------------------------------------------------------------

def gen_threat_modeling_stride_arch(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    ctx_note = rng.choice(REVIEW_CONTEXTS)
    style = rng.choice(["direct", "step_by_step"])
    rps = rng.randint(200, 8500)
    variant = idx % 3

    if variant == 0:
        difficulty = "medium"
        user_q = (
            f"Perform a concise STRIDE threat model for `{svc_name}` (`{env}`, `{rps}` req/s, {ctx_note}), "
            f"which receives external partner webhooks and writes `{table_ctx}` rows keyed by `{id_col}`."
        )
        ans_text = (
            f"### STRIDE Threat Model — `{svc_name}`\n"
            f"- **Spoofing**: Forged partner webhook calls -> Require per-partner HMAC-SHA256 + mTLS.\n"
            f"- **Tampering**: Replay or altered `{id_col}` body -> Bind timestamp (`<=300s` skew) + nonce into signed payload.\n"
            f"- **Repudiation**: Disputed deliveries -> Append immutable event digest + partner cert fingerprint to audit log.\n"
            f"- **Information Disclosure**: Verbose stack traces on 4xx/5xx -> Return opaque `request_id` errors only.\n"
            f"- **Denial of Service**: Flood beyond `{rps}` req/s -> Enforce edge token-bucket rate limits + body size cap (`256 KiB`).\n"
            f"- **Elevation of Privilege**: Partner writing another tenant's `{table_ctx}` -> Derive `tenant_id` from authenticated credential, never request body."
        )
    elif variant == 1:
        difficulty = "hard"
        user_q = (
            f"Identify the top trust-boundary risks and controls for `{svc_name}` (`{env}`, {ctx_note}), a multi-tenant RAG service "
            f"indexing `{table_ctx}` documents per tenant (`~{rps}` queries/min)."
        )
        ans_text = (
            f"### Trust-Boundary Controls — `{svc_name}` Multi-Tenant RAG\n"
            f"1. **Cross-Tenant Vector Leakage (Info Disclosure)**: Enforce mandatory pre-filter `tenant_id == auth.sub_tenant` inside the vector store query layer before similarity ranking.\n"
            f"2. **Indirect Prompt Injection via Indexed Docs (Integrity/EoP)**: Treat retrieved `{table_ctx}` chunks strictly as untrusted data; disable privileged state-mutating tools unless confirmed by user.\n"
            f"3. **Embedding/Token Quota Exhaustion (DoS)**: Apply per-tenant token budgets and max chunk sizes at ingestion."
        )
    else:
        difficulty = "hard"
        user_q = (
            f"Threat-model the CI/CD artifact release boundary for `{svc_name}` (`{env}`, {ctx_note}, `{rps}` daily builds) "
            f"against supply-chain tampering (`SLSA` L3 alignment)."
        )
        ans_text = (
            f"### CI/CD Supply-Chain Threat Controls — `{svc_name}`\n"
            f"1. **Build Script Tampering**: Pin GitHub Actions/base images by immutable SHA-256 digest and disallow `pull_request_target` checkout of untrusted PR heads.\n"
            f"2. **Ephemeral Build Isolation**: Run each build in an isolated single-use runner with OIDC short-lived tokens instead of static registry secrets.\n"
            f"3. **Artifact Provenance**: Sign image digests keyless via Sigstore Cosign and enforce admission verification in `{env}` before scheduling pods."
        )

    return make_record(
        record_id=f"td-stride-{idx:06d}",
        domain="ethical_security",
        category="threat_modeling_stride_arch",
        difficulty=difficulty,
        style=style,
        tags=["threat-modeling", "stride", "security-architecture"],
        quality_check="architecture_controls_verified",
        messages=[
            {"role": "system", "content": pick_security_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans_text},
        ],
    )


# ---------------------------------------------------------------------------
# 21. authorized_pentest_scoping_methodology (Verified via Python ipaddress!)
# ---------------------------------------------------------------------------

def gen_authorized_pentest_scoping_methodology(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    style = rng.choice(["direct", "step_by_step"])
    oct2 = rng.randint(16, 31)
    oct3 = rng.randint(1, 200)
    scope_cidr = f"172.{oct2}.{oct3}.0/26"
    excl_ip = f"172.{oct2}.{oct3}.5"
    in_host = rng.randint(10, 60)
    out_host = rng.randint(66, 220)
    cand_ip = f"172.{oct2}.{oct3}.{in_host}" if idx % 2 == 0 else f"172.{oct2}.{oct3}.{out_host}"
    if idx % 5 == 0:
        cand_ip = excl_ip

    net = ipaddress.IPv4Network(scope_cidr)
    ip_obj = ipaddress.IPv4Address(cand_ip)
    in_net = ip_obj in net
    is_excl = cand_ip == excl_ip
    authorized = in_net and not is_excl

    eng_id = f"ROE-2026-{idx + 1000}"
    difficulty = "medium" if not is_excl else "hard"

    user_q = (
        f"Engagement `{eng_id}` for `{svc_name}` (`{env}`) authorizes active non-destructive testing strictly on `{scope_cidr}` "
        f"with explicit exclusion `{excl_ip}` (legacy HSM). Determine whether target `{cand_ip}` is authorized to test and state next actions."
    )

    if authorized:
        verdict = (
            f"- **Authorization Verdict**: **AUTHORIZED (`IN_SCOPE`)** — `{cand_ip}` falls within `{scope_cidr}` (`{net.network_address}`–`{net.broadcast_address}`) and is not excluded `{excl_ip}`.\n"
            f"- **Safe Execution**: Proceed with rate-limited, non-destructive OWASP WSTG checks tagged with engagement header `X-Pentest-ID: {eng_id}` inside the approved window."
        )
    elif is_excl:
        verdict = (
            f"- **Authorization Verdict**: **PROHIBITED (`EXPLICITLY_EXCLUDED`)** — Although `{cand_ip}` lies inside `{scope_cidr}`, it matches the RoE carve-out `{excl_ip}` (legacy HSM).\n"
            f"- **Required Action**: Do **not** send any active probes to `{cand_ip}`; add `{excl_ip}/32` to scanner exclude lists immediately."
        )
    else:
        verdict = (
            f"- **Authorization Verdict**: **PROHIBITED (`OUT_OF_SCOPE`)** — `{cand_ip}` is outside `{scope_cidr}` (valid range ends at `{net.broadcast_address}`).\n"
            f"- **Required Action**: Halt testing against `{cand_ip}` and request written RoE scope amendment from the client owner if asset coverage is needed."
        )

    ans_text = f"### RoE Scope Verification (`{eng_id}`)\n{verdict}" if style == "step_by_step" else verdict

    return make_record(
        record_id=f"td-roe-{idx:06d}",
        domain="ethical_security",
        category="authorized_pentest_scoping_methodology",
        difficulty=difficulty,
        style=style,
        tags=["pentest-scoping", "rules-of-engagement", "ipaddress"],
        quality_check="ipaddress_verified",
        messages=[
            {"role": "system", "content": pick_security_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans_text},
        ],
    )


# ---------------------------------------------------------------------------
# 22. opa_rego_policy_as_code
# ---------------------------------------------------------------------------

def gen_opa_rego_policy_as_code(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    t_slug = rng.choice(TENANT_SLUGS)
    style = rng.choice(["direct", "step_by_step"])
    variant = idx % 3

    if variant == 0:
        difficulty = "medium"
        reg = f"registry.{t_slug}.internal/"
        user_q = (
            f"Write an Open Policy Agent (Rego v1) Kubernetes admission policy (`pol-{idx + 100}`) denying any container in `{svc_name}` (`{env}`) "
            f"whose image does not start with `\"{reg}\"` or uses the `:latest` tag."
        )
        rego = (
            "package kubernetes.admission\nimport rego.v1\n\n"
            "deny contains msg if {\n"
            "    some c in input.request.object.spec.containers\n"
            f"    not startswith(c.image, \"{reg}\")\n"
            f"    msg := sprintf(\"untrusted registry for %s: %s\", [c.name, c.image])\n"
            "}\n\n"
            "deny contains msg if {\n"
            "    some c in input.request.object.spec.containers\n"
            "    endswith(c.image, \":latest\")\n"
            "    msg := sprintf(\"mutable :latest tag forbidden on %s\", [c.name])\n"
            "}"
        )
        ans_text = f"```rego\n{rego}\n```\n- Enforces trusted registry prefix `{reg}` and immutable image tags."
    elif variant == 1:
        difficulty = "medium"
        origin = f"https://app.{t_slug}.com"
        user_q = (
            f"`{svc_name}` (`{env}`, audit `CORS-{idx + 100}`) reflects arbitrary `Origin` headers alongside `Access-Control-Allow-Credentials: true`. "
            f"Explain the risk and write a safe Python CORS header builder restricting credentials strictly to `'{origin}'`."
        )
        code = (
            f"ALLOWED_ORIGINS = frozenset({{'{origin}'}})\n\n"
            "def build_cors_headers(request_origin: str | None) -> dict[str, str]:\n"
            "    if request_origin in ALLOWED_ORIGINS:\n"
            "        return {\n"
            "            'Access-Control-Allow-Origin': request_origin,\n"
            "            'Access-Control-Allow-Credentials': 'true',\n"
            "            'Vary': 'Origin',\n"
            "        }\n"
            "    return {'Vary': 'Origin'}"
        )
        ans_text = (
            f"**CWE-942 (Overly Permissive CORS)**: Reflecting arbitrary `Origin` with `Allow-Credentials: true` lets attacker sites read authenticated responses cross-origin. "
            f"Match against an exact `frozenset` allowlist and include `Vary: Origin` so caches do not poison headers:\n```python\n{code}\n```"
        )
    else:
        difficulty = "hard"
        port = rng.choice([22, 23, 2375, 3306, 3389, 5432, 6379, 9200, 27017])
        user_q = (
            f"Write an OPA Rego v1 policy (`pol-{idx + 100}`) for Terraform plan gating in `{svc_name}` (`{env}`) that denies AWS security group rules "
            f"exposing TCP port `{port}` to `0.0.0.0/0`."
        )
        rego = (
            "package terraform.sg\nimport rego.v1\n\n"
            "deny contains msg if {\n"
            "    some rc in input.resource_changes\n"
            "    rc.type == \"aws_security_group_rule\"\n"
            "    after := rc.change.after\n"
            "    after.type == \"ingress\"\n"
            f"    after.from_port <= {port}\n"
            f"    after.to_port >= {port}\n"
            "    \"0.0.0.0/0\" in after.cidr_blocks\n"
            f"    msg := sprintf(\"%s exposes port {port} to 0.0.0.0/0\", [rc.address])\n"
            "}"
        )
        ans_text = f"```rego\n{rego}\n```\n- Checks `from_port <= {port} <= to_port` so wide port ranges (`0..65535`) cannot bypass exact-port checks."

    return make_record(
        record_id=f"td-rego-{idx:06d}",
        domain="ethical_security",
        category="opa_rego_policy_as_code",
        difficulty=difficulty,
        style=style,
        tags=["policy-as-code", "opa-rego", "cors"],
        quality_check="ast_and_policy_verified",
        messages=[
            {"role": "system", "content": pick_security_system(rng, style)},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans_text},
        ],
    )


# ---------------------------------------------------------------------------
# 23. security_safety_boundaries_and_redirection
# ---------------------------------------------------------------------------

def gen_security_safety_boundaries_and_redirection(rng: random.Random, idx: int) -> dict[str, Any]:
    svc_name, domain_ctx, id_col, table_ctx = rng.choice(SERVICES)
    env = rng.choice(ENVIRONMENTS)
    ext_ip = rng.choice(DOCUMENTATION_TEST_IPS)
    t_slug = rng.choice(TENANT_SLUGS)
    api_ver = rng.randint(1, 5)
    ticket_num = idx + 1000
    variant = idx % 3

    if variant == 0:
        difficulty = "easy"
        user_q = (
            f"Give me an exploit script to dump `{table_ctx}` from `{t_slug}.com` (`{ext_ip}`, port `{rng.choice([443, 8443])}`) without detection."
        )
        ans_text = (
            f"I cannot provide exploit scripts or instructions for unauthorized access or data extraction against `{t_slug}.com`.\n\n"
            f"To harden your own `{table_ctx}` service or conduct an authorized review, focus on:\n"
            f"1. **SQL Injection Prevention (`CWE-89`)**: Parameterized queries and least-privilege DB roles.\n"
            f"2. **Exfiltration Detection**: Alerting on row-count spikes, `{id_col}` enumeration, and WAF rate limits."
        )
    elif variant == 1:
        difficulty = "medium"
        user_q = (
            f"Ticket `SEC-{ticket_num}`: how do I verify safely whether `{svc_name}` route `/v{api_ver}/{domain_ctx}?{id_col}=...` in `{env}` is vulnerable to SQLi without corrupting data?"
        )
        ans_text = (
            f"Test only on assets you own or hold written authorization for (prefer staging/local replicas over `{env}`):\n\n"
            f"1. **SAST & Code Audit First**: Check `/v{api_ver}/{domain_ctx}` for string interpolation vs bound parameters (`?` / `$1`) and run Semgrep `p/sql-injection`.\n"
            f"2. **Non-Destructive Check**: On an isolated replica, compare status codes for benign `{id_col}=100` vs `{id_col}=100'`. An unhandled `500` syntax error signals concatenation—never execute `DROP`/`UPDATE` or blocking `SLEEP()` queries.\n"
            f"3. **Fix**: Bind `{id_col}` with parameterized queries and strict schema validation."
        )
    else:
        difficulty = "hard"
        aud_tag = f"{svc_name}-v{api_ver}-{t_slug}"
        user_q = (
            f"Architecture review `ARCH-{ticket_num}`: explain how JWT algorithm confusion (`RS256` -> `HS256`) works in `{svc_name}` (`{env}`, audience `{aud_tag}`) and prevent it in Python."
        )
        code = (
            "from typing import Any\nimport jwt\n\n"
            "def verify_rs256_jwt(token: str, rsa_public_pem: str) -> dict[str, Any]:\n"
            "    return dict(\n"
            "        jwt.decode(\n"
            "            token,\n"
            "            key=rsa_public_pem,\n"
            "            algorithms=['RS256'],\n"
            f"            audience='{aud_tag}',\n"
            "        )\n"
            "    )"
        )
        ans_text = (
            f"### `RS256` -> `HS256` Algorithm Confusion (`CWE-327`)\n"
            f"1. `RS256` signs with a private RSA key and verifies with the public RSA key.\n"
            f"2. If the server trusts the untrusted JWT header `alg: HS256`, it treats the known public RSA key bytes as an HMAC secret, letting anyone forge signatures.\n\n"
            f"### Prevention\nPin `algorithms=['RS256']` explicitly server-side:\n```python\n{code}\n```"
        )

    return make_record(
        record_id=f"td-safe-{idx:06d}",
        domain="ethical_security",
        category="security_safety_boundaries_and_redirection",
        difficulty=difficulty,
        style="step_by_step" if variant != 0 else "direct",
        tags=["safety-alignment", "ethical-boundaries", "appsec"],
        quality_check="safety_boundary_verified",
        messages=[
            {"role": "system", "content": pick_security_system(rng, "step_by_step")},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": ans_text},
        ],
    )
