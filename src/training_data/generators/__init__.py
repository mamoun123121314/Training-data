"""Registry of all 75 specialized generators (42 Autonomous Agent + 17 Coding + 16 Ethical Security)."""

from __future__ import annotations

from collections.abc import Callable
import random
from typing import Any

from training_data.generators.coding_part1 import (
    gen_algo_data_structures,
    gen_api_backend_engineering,
    gen_bug_diagnosis_and_fix,
    gen_refactoring_clean_code,
    gen_sql_query_and_optimization,
    gen_testing_property_edge_cases,
)
from training_data.generators.coding_part2 import (
    gen_data_pipelines_parsing,
    gen_devops_ci_containers,
    gen_git_cli_shell_engineering,
    gen_multi_turn_code_review,
    gen_systems_concurrency_perf,
    gen_typescript_modern_js,
)
from training_data.generators.coding_part3 import (
    gen_go_concurrency_and_services,
    gen_observability_opentelemetry_metrics,
    gen_regex_and_input_validation,
    gen_rust_memory_and_error_handling,
    gen_serialization_schema_migrations,
)
from training_data.generators.security_part1 import (
    gen_auth_crypto_jwt_security,
    gen_cvss_triage_and_vuln_reporting,
    gen_detection_engineering_sigma_semgrep,
    gen_log_forensics_incident_response,
    gen_network_acl_firewall_analysis,
    gen_secure_code_remediation_web,
)
from training_data.generators.security_part2 import (
    gen_authorized_pentest_scoping_methodology,
    gen_cloud_k8s_container_hardening,
    gen_opa_rego_policy_as_code,
    gen_security_safety_boundaries_and_redirection,
    gen_threat_modeling_stride_arch,
)
from training_data.generators.security_part3 import (
    gen_api_rate_limiting_and_graphql_security,
    gen_linux_systemd_and_host_hardening,
    gen_secrets_management_and_ci_supply_chain,
    gen_ssrf_xxe_deserialization_deep_dive,
    gen_web_security_headers_and_csp,
)
from training_data.generators.agent_part1 import (
    gen_agent_error_recovery_and_retry,
    gen_code_repo_navigation_agent,
    gen_missing_params_clarification_agent,
    gen_parallel_and_chained_tool_use,
    gen_security_audit_agent_workflow,
    gen_single_tool_precision_calling,
)
from training_data.generators.agent_part2 import (
    gen_devops_incident_triage_agent,
    gen_multi_agent_handoff_and_state_tracking,
    gen_prompt_injection_resistant_agent,
    gen_rag_citation_and_conflict_resolution,
    gen_sandbox_python_interpreter_agent,
    gen_sql_database_exploration_agent,
    gen_structured_output_and_planning,
)
from training_data.generators.agent_part3 import (
    gen_context_budget_and_pagination_agent,
    gen_guardrail_policy_check_agent,
    gen_multi_turn_user_agent_collaboration,
    gen_parallel_multi_tool_fanout,
)
from training_data.generators.agent_part4 import (
    gen_agent_long_horizon_plan_and_replan,
    gen_agent_memory_and_preferences_retrieval,
    gen_agent_subagent_orchestration_and_merge,
    gen_mcp_server_discovery_and_invocation,
    gen_swe_git_conflict_and_pr_workflow,
    gen_swe_multi_file_bugfix_trajectory,
    gen_swe_shell_build_and_linter_loop,
    gen_swe_stacktrace_debugger_agent,
    gen_swe_test_driven_feature_agent,
)
from training_data.generators.agent_part5 import (
    gen_agent_ambiguous_tool_disambiguation,
    gen_agent_api_http_graphql_rest_client,
    gen_agent_browser_dom_navigation_workflow,
    gen_agent_hallucination_abstention_on_empty_results,
    gen_agent_idempotency_and_rollback_safety,
    gen_agent_no_tool_needed_restraint,
    gen_agent_pii_redaction_and_secret_hygiene,
    gen_agent_streaming_chunk_and_large_file_handling,
)
from training_data.generators.agent_part6 import (
    gen_agent_cost_and_latency_budget_routing,
    gen_agent_data_visualization_and_report_builder,
    gen_agent_human_in_the_loop_approval_gate,
    gen_agent_json_schema_strict_extraction,
    gen_agent_runbook_table_and_sla_synthesis,
    gen_agent_sql_analytics_self_correction,
    gen_agent_time_zone_and_temporal_reasoning,
    gen_agent_trajectory_judge_and_reward_critique,
)

GeneratorFn = Callable[[random.Random, int], dict[str, Any]]

ALL_GENERATORS: list[tuple[str, str, GeneratorFn]] = [
    # Pillar 1: Coding & Software Engineering (17 categories)
    ("coding", "algo_data_structures", gen_algo_data_structures),
    ("coding", "bug_diagnosis_and_fix", gen_bug_diagnosis_and_fix),
    ("coding", "sql_query_and_optimization", gen_sql_query_and_optimization),
    ("coding", "api_backend_engineering", gen_api_backend_engineering),
    ("coding", "refactoring_clean_code", gen_refactoring_clean_code),
    ("coding", "testing_property_edge_cases", gen_testing_property_edge_cases),
    ("coding", "typescript_modern_js", gen_typescript_modern_js),
    ("coding", "systems_concurrency_perf", gen_systems_concurrency_perf),
    ("coding", "data_pipelines_parsing", gen_data_pipelines_parsing),
    ("coding", "devops_ci_containers", gen_devops_ci_containers),
    ("coding", "git_cli_shell_engineering", gen_git_cli_shell_engineering),
    ("coding", "multi_turn_code_review", gen_multi_turn_code_review),
    ("coding", "go_concurrency_and_services", gen_go_concurrency_and_services),
    ("coding", "rust_memory_and_error_handling", gen_rust_memory_and_error_handling),
    ("coding", "regex_and_input_validation", gen_regex_and_input_validation),
    ("coding", "serialization_schema_migrations", gen_serialization_schema_migrations),
    ("coding", "observability_opentelemetry_metrics", gen_observability_opentelemetry_metrics),
    # Pillar 2: Ethical Security & Defensive AppSec (16 categories)
    ("ethical_security", "secure_code_remediation_web", gen_secure_code_remediation_web),
    ("ethical_security", "auth_crypto_jwt_security", gen_auth_crypto_jwt_security),
    ("ethical_security", "cvss_triage_and_vuln_reporting", gen_cvss_triage_and_vuln_reporting),
    ("ethical_security", "detection_engineering_sigma_semgrep", gen_detection_engineering_sigma_semgrep),
    ("ethical_security", "log_forensics_incident_response", gen_log_forensics_incident_response),
    ("ethical_security", "network_acl_firewall_analysis", gen_network_acl_firewall_analysis),
    ("ethical_security", "cloud_k8s_container_hardening", gen_cloud_k8s_container_hardening),
    ("ethical_security", "threat_modeling_stride_arch", gen_threat_modeling_stride_arch),
    ("ethical_security", "authorized_pentest_scoping_methodology", gen_authorized_pentest_scoping_methodology),
    ("ethical_security", "opa_rego_policy_as_code", gen_opa_rego_policy_as_code),
    ("ethical_security", "security_safety_boundaries_and_redirection", gen_security_safety_boundaries_and_redirection),
    ("ethical_security", "ssrf_xxe_deserialization_deep_dive", gen_ssrf_xxe_deserialization_deep_dive),
    ("ethical_security", "api_rate_limiting_and_graphql_security", gen_api_rate_limiting_and_graphql_security),
    ("ethical_security", "linux_systemd_and_host_hardening", gen_linux_systemd_and_host_hardening),
    ("ethical_security", "secrets_management_and_ci_supply_chain", gen_secrets_management_and_ci_supply_chain),
    ("ethical_security", "web_security_headers_and_csp", gen_web_security_headers_and_csp),
    # Pillar 3: Autonomous Agent Use & Agent Fine-Tuning (42 categories!)
    ("agent_use", "single_tool_precision_calling", gen_single_tool_precision_calling),
    ("agent_use", "parallel_and_chained_tool_use", gen_parallel_and_chained_tool_use),
    ("agent_use", "agent_error_recovery_and_retry", gen_agent_error_recovery_and_retry),
    ("agent_use", "code_repo_navigation_agent", gen_code_repo_navigation_agent),
    ("agent_use", "security_audit_agent_workflow", gen_security_audit_agent_workflow),
    ("agent_use", "missing_params_clarification_agent", gen_missing_params_clarification_agent),
    ("agent_use", "rag_citation_and_conflict_resolution", gen_rag_citation_and_conflict_resolution),
    ("agent_use", "structured_output_and_planning", gen_structured_output_and_planning),
    ("agent_use", "sql_database_exploration_agent", gen_sql_database_exploration_agent),
    ("agent_use", "devops_incident_triage_agent", gen_devops_incident_triage_agent),
    ("agent_use", "prompt_injection_resistant_agent", gen_prompt_injection_resistant_agent),
    ("agent_use", "sandbox_python_interpreter_agent", gen_sandbox_python_interpreter_agent),
    ("agent_use", "multi_agent_handoff_and_state_tracking", gen_multi_agent_handoff_and_state_tracking),
    ("agent_use", "parallel_multi_tool_fanout", gen_parallel_multi_tool_fanout),
    ("agent_use", "multi_turn_user_agent_collaboration", gen_multi_turn_user_agent_collaboration),
    ("agent_use", "context_budget_and_pagination_agent", gen_context_budget_and_pagination_agent),
    ("agent_use", "guardrail_policy_check_agent", gen_guardrail_policy_check_agent),
    ("agent_use", "swe_multi_file_bugfix_trajectory", gen_swe_multi_file_bugfix_trajectory),
    ("agent_use", "swe_git_conflict_and_pr_workflow", gen_swe_git_conflict_and_pr_workflow),
    ("agent_use", "swe_test_driven_feature_agent", gen_swe_test_driven_feature_agent),
    ("agent_use", "swe_stacktrace_debugger_agent", gen_swe_stacktrace_debugger_agent),
    ("agent_use", "swe_shell_build_and_linter_loop", gen_swe_shell_build_and_linter_loop),
    ("agent_use", "mcp_server_discovery_and_invocation", gen_mcp_server_discovery_and_invocation),
    ("agent_use", "agent_long_horizon_plan_and_replan", gen_agent_long_horizon_plan_and_replan),
    ("agent_use", "agent_memory_and_preferences_retrieval", gen_agent_memory_and_preferences_retrieval),
    ("agent_use", "agent_subagent_orchestration_and_merge", gen_agent_subagent_orchestration_and_merge),
    ("agent_use", "agent_no_tool_needed_restraint", gen_agent_no_tool_needed_restraint),
    ("agent_use", "agent_ambiguous_tool_disambiguation", gen_agent_ambiguous_tool_disambiguation),
    ("agent_use", "agent_api_http_graphql_rest_client", gen_agent_api_http_graphql_rest_client),
    ("agent_use", "agent_browser_dom_navigation_workflow", gen_agent_browser_dom_navigation_workflow),
    ("agent_use", "agent_streaming_chunk_and_large_file_handling", gen_agent_streaming_chunk_and_large_file_handling),
    ("agent_use", "agent_idempotency_and_rollback_safety", gen_agent_idempotency_and_rollback_safety),
    ("agent_use", "agent_pii_redaction_and_secret_hygiene", gen_agent_pii_redaction_and_secret_hygiene),
    ("agent_use", "agent_hallucination_abstention_on_empty_results", gen_agent_hallucination_abstention_on_empty_results),
    ("agent_use", "agent_json_schema_strict_extraction", gen_agent_json_schema_strict_extraction),
    ("agent_use", "agent_sql_analytics_self_correction", gen_agent_sql_analytics_self_correction),
    ("agent_use", "agent_cost_and_latency_budget_routing", gen_agent_cost_and_latency_budget_routing),
    ("agent_use", "agent_human_in_the_loop_approval_gate", gen_agent_human_in_the_loop_approval_gate),
    ("agent_use", "agent_time_zone_and_temporal_reasoning", gen_agent_time_zone_and_temporal_reasoning),
    ("agent_use", "agent_data_visualization_and_report_builder", gen_agent_data_visualization_and_report_builder),
    ("agent_use", "agent_trajectory_judge_and_reward_critique", gen_agent_trajectory_judge_and_reward_critique),
    ("agent_use", "agent_runbook_table_and_sla_synthesis", gen_agent_runbook_table_and_sla_synthesis),
]

AGENT_GENERATORS: list[tuple[str, str, GeneratorFn]] = [
    entry for entry in ALL_GENERATORS if entry[0] == "agent_use"
]
