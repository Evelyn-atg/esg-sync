#!/bin/bash
# ==========================================================================
# Qwen3.8-Max LLM Matching Runner (with reasoning mode)
# ==========================================================================
# Runs LLM variable matching for all 100 GT PDFs using qwen3.8-max with
# enable_thinking=true for complex indicator transformations.
#
# Previous outputs (quantitative_results_Qwen/) are preserved.
# New outputs go to quantitative_results_qwen38/.
#
# Usage:
#   bash run_qwen38_matching.sh --dry-run     # Preview + cost estimate
#   bash run_qwen38_matching.sh               # Full run (will ask confirmation)
#   bash run_qwen38_matching.sh --resume      # Skip already-processed PDFs
#   bash run_qwen38_matching.sh --no-thinking # qwen3.8-max without reasoning
#
# Can run on:
#   - Local Mac (with DashScope API key, numeric_extracts/ from git pull)
#   - Cluster login node (no GPU needed, API calls only)
# ==========================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Model configuration ---
export QWEN_MAX_MODEL="${QWEN_MAX_MODEL:-qwen3.8-max}"
export ENABLE_THINKING="${ENABLE_THINKING:-true}"
export THINKING_MAX_TOKENS="${THINKING_MAX_TOKENS:-16000}"
export QUANTITATIVE_RESULT_DIR="${QUANTITATIVE_RESULT_DIR:-quantitative_results_qwen38}"

# --- API Key (fallback to hardcoded if env not set) ---
if [ -z "${QWEN_MAX_API_KEY:-}" ]; then
    # Try .env file
    if [ -f .env ]; then
        export $(grep -v '^#' .env | xargs 2>/dev/null || true)
    fi
fi

if [ -z "${QWEN_MAX_API_KEY:-}" ]; then
    echo "[ERROR] QWEN_MAX_API_KEY not set. Put it in .env or export it."
    echo "        export QWEN_MAX_API_KEY=sk-xxxxx"
    exit 1
fi

# --- Find Python ---
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    if command -v python3 &>/dev/null; then
        PYTHON=python3
    elif command -v python &>/dev/null; then
        PYTHON=python
    else
        echo "[ERROR] No python3 found"
        exit 1
    fi
fi

# --- Check numeric_extracts exists ---
if [ ! -d numeric_extracts ]; then
    echo "[ERROR] numeric_extracts/ directory not found."
    echo "        On cluster: already in esg-pipeline/"
    echo "        On Mac: git pull from esg-pipeline repo first"
    exit 1
fi

# --- Run ---
echo "================================================================"
echo "  Qwen3.8-Max LLM Matching"
echo "  Model:    $QWEN_MAX_MODEL"
echo "  Thinking: $ENABLE_THINKING"
echo "  Output:   $QUANTITATIVE_RESULT_DIR"
echo "================================================================"
echo ""

exec "$PYTHON" run_qwen38_matching.py "$@"
