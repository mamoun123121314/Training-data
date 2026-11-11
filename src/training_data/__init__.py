"""Training-Data Forge v5.0.0: Multi-Billion Solver-Verified Coding, Security & Autonomous Agent Fine-Tuning Suite."""

from training_data.agent_dpo import DPO_TAXONOMIES, synthesize_agent_dpo_pair, validate_dpo_pair
from training_data.agent_ultra import (
    AGENT_DPO_TOTAL,
    AGENT_SFT_TOTAL,
    AgentDPODataset,
    AgentUltraDataset,
    iter_agent_ultra_sft,
    synthesize_agent_ultra_sft,
)
from training_data.streaming import iter_streaming_dataset
from training_data.ultra_billion import (
    TOTAL_BILLION_RECORDS,
    BillionUltraDataset,
    feistel_inverse,
    feistel_permute,
    get_shard_bounds,
    iter_billion_ultra,
    resolve_coordinate,
    synthesize_ultra_record,
)

__version__ = "5.0.0"

__all__ = [
    "AGENT_DPO_TOTAL",
    "AGENT_SFT_TOTAL",
    "DPO_TAXONOMIES",
    "TOTAL_BILLION_RECORDS",
    "AgentDPODataset",
    "AgentUltraDataset",
    "BillionUltraDataset",
    "feistel_inverse",
    "feistel_permute",
    "get_shard_bounds",
    "iter_agent_ultra_sft",
    "iter_billion_ultra",
    "iter_streaming_dataset",
    "resolve_coordinate",
    "synthesize_agent_dpo_pair",
    "synthesize_agent_ultra_sft",
    "synthesize_ultra_record",
    "validate_dpo_pair",
]
