#!/bin/bash
#SBATCH -p A800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=1440
#SBATCH -o qwen38_pipeline_%j.out

# ==========================================================================
# Qwen3.8-Max Pipeline Runner (Cluster) — v3: text-LLM-only comparison
# ==========================================================================
# Re-runs ONLY the text-LLM steps (LLM matching + calculation) with:
#   - Text LLM:  qwen3.8-max + enable_thinking=true
#
# OCR and numeric_extracts are NOT re-run:
#   - OCR uses chandra-ocr-2 (local model, no LLM) — results unchanged
#   - numeric_extracts uses VL model (qwen-vl-plus) — not the text LLM
#   - Both reuse existing results from their original directories
#
# Only these outputs go to SEPARATE directories for qwen3.7 vs qwen3.8 comparison:
#
#   Old (qwen3.7-max):                         New (qwen3.8-max + thinking):
#   ─────────────────────────────────────────  ────────────────────────────────────────
#   quantitative_results_Qwen/                quantitative_results_qwen38/   (LLM matching)
#   calculation_results/                      calculation_results_qwen38/    (calculation)
#
# Usage:
#   sbatch run_qwen38_cluster.sh
# ==========================================================================

set -euo pipefail

module load apps/anaconda3/2021.05
source activate "$HOME/envs/chandra"
cd "$HOME/esg-pipeline" || { echo "ERROR: ~/esg-pipeline not found"; exit 1; }

# --- Pull latest code ---
echo "=== Git pull ==="
git pull || echo "WARNING: git pull failed, using existing code"

# ==========================================================================
# Model configuration — text LLM only, with thinking
# ==========================================================================
export QWEN_MAX_MODEL="qwen3.8-max"
export ENABLE_THINKING="true"
export THINKING_BUDGET="8192"           # reasoning token budget (CoT trace)
export THINKING_MAX_TOKENS="16384"      # total output = thinking_budget + answer (8192+8192)

# VL model NOT changed — keep using qwen-vl-plus (default)
# OCR model NOT changed — chandra-ocr-2 (local)

# ==========================================================================
# Output directory separation — only text-LLM outputs
# ==========================================================================
export QUANTITATIVE_RESULT_DIR="quantitative_results_qwen38"      # LLM matching output
export RESULT_OUTPUT_DIR="calculation_results_qwen38"             # Calculation output

# --- API Key ---
if [ -z "${QWEN_MAX_API_KEY:-}" ] && [ -f .env ]; then
    export $(grep -v '^#' .env | xargs 2>/dev/null || true)
fi
if [ -z "${QWEN_MAX_API_KEY:-}" ]; then
    echo "[ERROR] QWEN_MAX_API_KEY not set. Put it in ~/esg-pipeline/.env"
    exit 1
fi

mkdir -p logs
JOB_TAG="${SLURM_JOB_ID:-local}"

echo "=========================================================================="
echo "  Qwen3.8-Max Pipeline  (v3 — text-LLM-only comparison)"
echo "=========================================================================="
echo "  Text LLM:            $QWEN_MAX_MODEL (thinking=$ENABLE_THINKING)"
echo "  Thinking budget:     $THINKING_BUDGET tokens (reasoning_content)"
echo "  Max tokens:          $THINKING_MAX_TOKENS (reasoning + answer combined)"
echo "  VL model:            qwen-vl-plus (unchanged)"
echo "  OCR model:           chandra-ocr-2 (unchanged)"
echo ""
echo "  Reads from (unchanged, reused):"
echo "    numeric_extracts/                    (VL extraction results)"
echo "    quantitative_results_ocr/chandra_ocr_2/  (OCR results)"
echo ""
echo "  Writes to (NEW, separate from old):"
echo "    $QUANTITATIVE_RESULT_DIR/  (LLM matching)"
echo "    $RESULT_OUTPUT_DIR/        (calculation)"
echo ""
echo "  Old results preserved:"
echo "    quantitative_results_Qwen/  (old LLM matching, qwen3.7-max)"
echo "    calculation_results/        (old calculation)"
echo "=========================================================================="
echo ""

# --- Step 1/2: LLM matching ---
echo "=== Step 1/2: LLM matching ($QWEN_MAX_MODEL, thinking=$ENABLE_THINKING) ==="
echo "    thinking_budget=$THINKING_BUDGET, max_tokens=$THINKING_MAX_TOKENS"
python -m src.main --step llm_matching --force > "logs/matching_qwen38_${JOB_TAG}.log" 2>&1
echo "Matching done. See logs/matching_qwen38_${JOB_TAG}.log"

# --- Step 2/2: Calculation ---
echo "=== Step 2/2: Calculation ($QWEN_MAX_MODEL, thinking=$ENABLE_THINKING) ==="
echo "    thinking_budget=4096 (calculator uses smaller budget), max_tokens=8192"
python -m src.main --step calculation --force > "logs/calc_qwen38_${JOB_TAG}.log" 2>&1
echo "Calculation done. See logs/calc_qwen38_${JOB_TAG}.log"

echo ""
echo "=========================================================================="
echo "  Pipeline complete!"
echo ""
echo "  New results (qwen3.8-max + thinking):"
echo "    $QUANTITATIVE_RESULT_DIR/  (LLM matching)"
echo "    $RESULT_OUTPUT_DIR/        (calculation)"
echo ""
echo "  Old results (qwen3.7-max, preserved):"
echo "    quantitative_results_Qwen/  (LLM matching)"
echo "    calculation_results/        (calculation)"
echo ""
echo "  Unchanged (reused, not re-run):"
echo "    numeric_extracts/           (VL extraction)"
echo "    quantitative_results_ocr/chandra_ocr_2/  (OCR)"
echo "=========================================================================="
