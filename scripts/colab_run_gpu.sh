#!/usr/bin/env bash
set -euo pipefail

echo "========================================================================"
echo " Training-data v6.0 — 1-Click Colab T4 GPU Agent POV Trainer (SFT + DPO)"
echo "========================================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

echo "[Step 1/2] Ensuring QLoRA dependencies (bitsandbytes, peft, trl)..."
python3 -c "import bitsandbytes, peft, trl" 2>/dev/null || pip install -q bitsandbytes peft trl

echo "[Step 2/2] Launching 4-bit QLoRA Stage 1 (SFT) + Stage 2 (DPO) on GPU..."
python3 scripts/train_gpu_qlora_sft_dpo.py "$@"
