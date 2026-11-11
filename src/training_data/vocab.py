"""Expanded domain vocabularies (100 services, 32 environments, 80 tenants) for million-scale synthesis."""

from __future__ import annotations

import random

SERVICES = [
    ("ledger-core", "payments", "txn_id", "settlement_batches"),
    ("auth-gateway", "identity", "session_id", "oauth_tokens"),
    ("fleet-telemetry", "logistics", "vehicle_id", "gps_pings"),
    ("fhir-ingest", "clinical", "encounter_id", "patient_records"),
    ("billing-meter", "subscriptions", "invoice_id", "usage_events"),
    ("artifact-registry", "ci_platform", "digest_sha", "package_blobs"),
    ("checkout-orchestrator", "commerce", "order_id", "cart_items"),
    ("siem-collector", "secops", "event_uuid", "audit_trails"),
    ("feature-store-api", "ml_infra", "entity_key", "feature_vectors"),
    ("webhook-dispatcher", "integrations", "delivery_id", "webhook_endpoints"),
    ("tenant-provisioner", "control_plane", "tenant_id", "workspace_quotas"),
    ("search-indexer", "discovery", "doc_id", "index_shards"),
    ("media-transcoder", "streaming", "job_id", "rendition_manifests"),
    ("dns-edge-router", "neteng", "zone_id", "rrset_records"),
    ("risk-scorer", "fraud_ops", "eval_id", "velocity_counters"),
    ("quota-enforcer", "api_gateway", "client_id", "rate_buckets"),
    ("secrets-vault-proxy", "platform_sec", "lease_id", "secret_versions"),
    ("iot-ota-broker", "embedded", "device_serial", "firmware_rollouts"),
    ("ticket-triage-svc", "support_ops", "ticket_id", "sla_timers"),
    ("inventory-allocator", "warehouse", "sku_code", "stock_reservations"),
    ("notification-hub", "messaging", "dispatch_id", "channel_templates"),
    ("analytics-rollup", "bi_engine", "metric_key", "hourly_aggregates"),
    ("scheduler-daemon", "batch_compute", "run_id", "cron_executions"),
    ("kyc-verifier", "compliance", "case_id", "verification_checks"),
    ("model-router", "inference", "request_id", "token_ledgers"),
    ("cdn-purger", "edge_cache", "purge_id", "cache_tags"),
    ("git-mirror-worker", "devtools", "repo_id", "sync_checkpoints"),
    ("policy-evaluator", "governance", "bundle_id", "decision_logs"),
    ("timeseries-compactor", "observability", "series_id", "chunk_segments"),
    ("session-recorder", "ux_telemetry", "replay_id", "dom_snapshots"),
    ("orderbook-matcher", "trading", "fill_id", "limit_orders"),
    ("fx-rate-oracle", "treasury", "quote_id", "currency_pairs"),
    ("ach-reconciler", "banking", "trace_no", "clearing_entries"),
    ("tax-engine", "fiscal", "calc_id", "jurisdiction_rules"),
    ("payout-scheduler", "disbursements", "payout_id", "merchant_balances"),
    ("chargeback-guard", "disputes", "dispute_id", "evidence_packets"),
    ("cert-rotator", "pki_ops", "serial_hex", "x509_leases"),
    ("waf-rule-sync", "edge_sec", "ruleset_id", "ip_reputation"),
    ("sbom-scanner", "supply_chain", "scan_id", "component_vulns"),
    ("honeypot-broker", "deception", "sensor_id", "canary_hits"),
    ("dlp-inspector", "data_gov", "finding_id", "redaction_spans"),
    ("pam-bastion", "zero_trust", "grant_id", "jit_sessions"),
    ("kms-envelope-svc", "crypto_ops", "dek_id", "key_rotations"),
    ("pod-autoscaler", "k8s_ops", "scale_id", "replica_metrics"),
    ("mesh-sidecar-mgr", "service_mesh", "proxy_id", "envoy_clusters"),
    ("blob-gc-worker", "object_store", "chunk_id", "orphan_blocks"),
    ("wal-archiver", "db_engine", "lsn_id", "wal_segments"),
    ("schema-registry", "event_bus", "schema_id", "avro_subjects"),
    ("kafka-lag-monitor", "streaming_ops", "group_id", "partition_offsets"),
    ("redis-shard-proxy", "cache_tier", "slot_id", "hotkey_stats"),
    ("graphql-federator", "bff_layer", "op_hash", "subgraph_traces"),
    ("grpc-ingress", "rpc_mesh", "stream_id", "method_descriptors"),
    ("email-dkim-signer", "mail_infra", "msg_id", "bounce_events"),
    ("sms-carrier-router", "telecom", "route_id", "dlr_receipts"),
    ("push-fanout-svc", "mobile_ops", "apns_id", "device_tokens"),
    ("feature-flag-eval", "experimentation", "flag_key", "rollout_rules"),
    ("ab-stats-engine", "growth_sci", "exp_id", "conversion_ cohorts".replace(" ", "")),
    ("vector-ann-index", "rag_core", "embed_id", "hnsw_segments"),
    ("prompt-guard-api", "ai_safety", "check_id", "policy_verdicts"),
    ("eval-harness-svc", "llm_ops", "suite_id", "benchmark_runs"),
    ("gpu-quota-scheduler", "cluster_sched", "alloc_id", "node_leases"),
    ("dataset-curator", "data_prep", "shard_id", "dedup_hashes"),
    ("ocr-layout-parser", "doc_ai", "page_id", "bounding_boxes"),
    ("asr-stream-gateway", "speech_ops", "utterance_id", "audio_frames"),
    ("dicom-anonymizer", "radiology", "study_uid", "pixel_series"),
    ("hl7-bridge", "hospital_it", "hl7_id", "adt_messages"),
    ("lab-result-router", "diagnostics", "specimen_id", "assay_panels"),
    ("pharmacy-formulary", "rx_ops", "rx_id", "drug_interactions"),
    ("parcel-sorter", "fulfillment", "tote_id", "conveyor_scans"),
    ("customs-manifest", "freight", "bol_id", "tariff_lines"),
    ("route-optimizer", "last_mile", "stop_id", "eta_windows"),
    ("coldchain-monitor", "reefer_iot", "probe_id", "temp_readings"),
    ("grid-load-balancer", "smart_grid", "meter_id", "kw_intervals"),
    ("ev-charger-ocpp", "mobility", "connector_id", "charging_sessions"),
    ("solar-inverter-ingest", "renewables", "array_id", "mppt_samples"),
    ("matchmaking-lobby", "game_backend", "lobby_id", "mmr_tickets"),
    ("leaderboard-ranker", "esports", "season_id", "player_scores"),
    ("anti-cheat-verifier", "game_sec", "heartbeat_id", "telemetry_proofs"),
    ("inventory-mint-svc", "virtual_econ", "item_uid", "trade_escrows"),
    ("drm-license-proxy", "media_sec", "key_kid", "playback_grants"),
    ("live-hls-packager", "broadcast", "segment_no", "manifest_variants"),
    ("ad-auction-bidder", "adtech", "bid_id", "impression_logs"),
    ("attribution-sink", "mkt_analytics", "click_id", "touchpoint_chains"),
    ("crm-contact-dedup", "revops", "lead_id", "merge_candidates"),
    ("payroll-withholding", "hr_tech", "payrun_id", "tax_stubs"),
    ("timecard-approver", "workforce", "shift_id", "overtime_entries"),
    ("procurement-po-svc", "erp_core", "po_number", "vendor_invoices"),
    ("contract-esign-api", "legal_tech", "envelope_id", "signer_events"),
    ("ediscovery-indexer", "litigation", "custodian_id", "legal_holds"),
    ("sanctions-screener", "aml_ops", "hit_id", "watchlist_entities"),
    ("travel-pnr-sync", "aviation", "pnr_locator", "flight_coupons"),
    ("hotel-ari-cache", "hospitality", "property_id", "room_allotments"),
    ("pos-offline-sync", "retail_edge", "terminal_id", "receipt_batches"),
    ("loyalty-points-ledger", "rewards", "member_id", "accrual_events"),
    ("catalog-price-engine", "merchandising", "offer_id", "markdown_rules"),
    ("returns-rma-portal", "reverse_log", "rma_id", "inspection_grades"),
    ("ci-cache-daemon", "build_infra", "cas_digest", "remote_artifacts"),
    ("terraform-state-lock", "iac_plane", "workspace_id", "state_versions"),
    ("canary-judge-svc", "progressive_cd", "rollout_id", "metric_verdicts"),
    ("chaos-fault-injector", "resilience", "experiment_id", "blast_targets"),
]

ENVIRONMENTS = [
    "prod-us-east-1", "prod-us-east-2", "prod-us-west-2", "prod-eu-west-1",
    "prod-eu-central-1", "prod-ap-south-1", "prod-ap-northeast-1", "prod-sa-east-1",
    "prod-ca-central-1", "prod-me-central-1", "prod-af-south-1", "prod-ap-southeast-2",
    "staging-us-east-1", "staging-us-west-2", "staging-eu-west-1", "staging-ap-northeast-1",
    "canary-us-east-1", "canary-eu-central-1", "canary-ap-south-1", "canary-us-west-2",
    "pci-cde-us-east", "pci-cde-eu-west", "hipaa-enclave-us", "fedramp-high-east",
    "corp-sec-vpc", "shared-data-vpc", "edge-pop-iad", "edge-pop-fra",
    "edge-pop-sin", "edge-pop-nrt", "dr-failover-west", "sandbox-perf-01",
]

TENANT_SLUGS = [
    "acme_corp", "globex_eu", "initech_us", "umbrella_bio", "stark_energy",
    "wayne_fin", "cyberdyne_ai", "soylent_ops", "hooli_cloud", "piedpiper_net",
    "massive_dyn", "aperture_lab", "tyrell_sys", "weyland_log", "oscorp_med",
    "nakatomi_trd", "rekall_mem", "omni_consumer", "virtucon_ind", "blue_sun",
    "choam_holdings", "spacing_guild", "shinra_power", "mishima_zaib", "abstergo_ent",
    "sarif_biotech", "faro_automata", "lumon_macro", "delos_parks", "ingen_genomics",
    "black_mesa", "tricor_pharma", "vought_media", "allstate_grid", "cobalt_freight",
    "meridian_bank", "veridian_dyn", "kestrel_aero", "solaris_orbital", "hyperion_mfg",
    "atlas_shipping", "vanguard_tel", "pinnacle_pay", "redwood_health", "summit_clearing",
    "harbor_insure", "ironclad_sec", "aurora_quant", "cascade_water", "granite_mining",
    "zephyr_wind", "boreal_timber", "pacific_rail", "continental_air", "metro_transit",
    "citadel_vault", "nexus_robotics", "prism_optics", "helix_labs", "quantum_ledger",
    "stratos_cdn", "vertex_compute", "pulse_cardio", "beacon_retail", "anchor_marine",
    "crestline_erp", "keystone_auth", "noble_logistics", "orbit_satcom", "titan_foundry",
    "velox_payments", "aegis_defense", "Lumen_fiber".lower(), "fjord_shipping", "alpine_re",
    "oasis_desal", "savanna_agri", "delta_cargo", "ionic_battery", "zenith_cloud",
]

FILE_PATHS = [
    "src/api/routes/orders.py", "src/auth/token_verifier.py", "internal/storage/repository.go",
    "services/webhooks/dispatcher.ts", "pkg/middleware/ratelimit.go", "app/workers/invoice_sync.py",
    "src/security/ssrf_guard.py", "src/db/query_builder.py", "deploy/k8s/api-deployment.yaml",
    "infra/iam/worker_policy.json", "src/ingest/stream_consumer.py", "src/billing/ledger_reconciler.py",
    "src/scheduler/dag_runner.py", "src/cache/tiered_store.py", "src/telemetry/span_exporter.py",
    "src/gateway/circuit_breaker.py", "src/crypto/envelope_kms.py", "src/auth/pkce_validator.py",
    "src/net/cidr_acl.py", "src/audit/cloudtrail_parser.py", "src/graphql/depth_limiter.py",
    "src/storage/s3_presigner.py", "src/queue/dlq_router.py", "src/metrics/histograms.py",
    "src/policy/opa_client.py", "src/tenant/quota_guard.py", "src/search/shard_router.py",
    "src/webhooks/hmac_verifier.py", "src/sync/checkpoint_store.py", "src/export/csv_streamer.py",
    "src/session/cookie_signer.py", "src/ratelimit/sliding_window.py", "src/rpc/retry_interceptor.py",
    "src/config/env_loader.py", "src/batch/chunk_writer.py", "src/pii/pseudonymizer.py",
]

REVIEW_CONTEXTS = [
    "during production readiness review", "before enabling multi-region traffic",
    "while triaging high tail latency", "during weekly architecture review",
    "ahead of SOC2 Type II audit", "while migrating off legacy workers",
    "during load-test bottleneck analysis", "in our core platform SDK",
    "for our zero-downtime release pipeline", "while hardening public API handlers",
    "in our high-throughput ingestion tier", "before cutting release candidate v3.2",
    "after post-incident action item review", "ahead of PCI-DSS scope assessment",
    "while isolating noisy-neighbor tenants", "during memory-footprint profiling",
    "for active-active failover readiness", "during annual penetration test remediation",
    "while upgrading Python 3.11 workers", "in our edge gateway middleware",
]

USER_OPENERS_DIRECT = [
    "Write a clean implementation for the following requirement:",
    "Give me production-ready code for this task (no filler):",
    "Implement this cleanly:",
    "Need a concise, bug-free solution for:",
    "Provide a direct solution:",
    "How should I implement this in production?",
    "Quick request—write clean, verified code for:",
    "Can you provide a minimal, idiomatic solution for:",
    "Supply a crisp, production-safe implementation:",
    "Write clean, well-typed code to solve this:",
]

USER_OPENERS_STEP = [
    "Walk through your reasoning step by step, then provide the implementation:",
    "Explain the root mechanism and edge cases first, then give the clean code:",
    "Break down the approach, prove complexity/safety, and write the full solution:",
    "Analyze this step by step before showing the final code:",
    "Help me reason through this carefully and then provide a verified implementation:",
    "Diagnose what's needed step by step and supply production-ready code:",
    "Detail the invariants step by step, then write the complete solution:",
    "Explain why naive approaches fail here and present the verified implementation:",
]

SECURITY_OPENERS_DIRECT = [
    "Audit this snippet, identify the CWE, and return the remediated version:",
    "Spot the security flaw here and provide the hardened fix concisely:",
    "Perform a defensive code review on this handler and patch the vulnerability:",
    "What vulnerability exists here and how do we fix it cleanly in production?",
    "Review this configuration/code for security risks and output the patched version:",
    "Identify the CWE/OWASP risk and supply hardened replacement code:",
]

SECURITY_OPENERS_STEP = [
    "Conduct a step-by-step AppSec review: identify CWE/OWASP category, explain the root cause, and provide patched code:",
    "Analyze the security posture here methodically—walk through trust boundaries, impact, and full defense-in-depth remediation:",
    "Break down why this is vulnerable, how an authorized auditor confirms it safely, and how to remediate it completely:",
    "Step through the vulnerability mechanics, severity reasoning, and secure code replacement:",
]

SAFE_INTERNAL_IPS = [
    f"10.{a}.{b}.{c}"
    for a, b, c in [
        (14, 22, 18), (28, 4, 105), (42, 90, 12), (8, 112, 77), (99, 3, 200),
        (60, 14, 33), (11, 45, 9), (73, 18, 144), (120, 6, 51), (35, 80, 211),
        (52, 19, 64), (88, 201, 17), (19, 7, 250), (64, 130, 29), (105, 31, 82),
    ]
] + [
    "172.16.8.44", "172.20.15.9", "172.31.19.55", "192.168.40.21", "192.168.120.85",
]

DOCUMENTATION_TEST_IPS = (
    [f"198.51.100.{h}" for h in (7, 14, 29, 45, 72, 96, 118, 154, 199, 241)]
    + [f"203.0.113.{h}" for h in (11, 25, 53, 88, 104, 146, 177, 202, 230, 249)]
    + [f"192.0.2.{h}" for h in (19, 31, 68, 109, 135, 162, 190, 214, 233, 251)]
)

USERNAMES = [
    "svc_deploy", "ci_runner_04", "analytics_ro", "app_worker", "ops_oncall",
    "billing_sync", "audit_exporter", "k8s_controller", "etl_ingestor", "backup_agent",
    "j.okafor", "m.hassan", "s.chen", "a.kovacs", "r.patel", "l.ramirez", "e.lindqvist",
    "d.mensah", "n.takahashi", "k.novak", "f.almansoor", "c.dubois", "t.nguyen",
    "h.bergstrom", "p.kowalski", "y.kim", "v.popov", "g.silva", "b.adeyemi", "z.ibrahim",
]


def gen_short_sha(rng: random.Random) -> str:
    return "".join(rng.choices("0123456789abcdef", k=8))


def gen_uuid_short(rng: random.Random, prefix: str = "req") -> str:
    part1 = "".join(rng.choices("0123456789abcdef", k=6))
    part2 = "".join(rng.choices("0123456789abcdef", k=6))
    return f"{prefix}_{part1}{part2}"
