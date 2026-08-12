#!/bin/bash
#SBATCH -p A800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=1440
#SBATCH -o qwen38_pipeline_%j.out

# ==========================================================================
# Qwen3.8-Max Full Pipeline Runner (Cluster)
# ==========================================================================
# Re-runs the ESG pipeline for 100 GT PDFs with:
#   - Text LLM:  qwen3.8-max (with enable_thinking=true for reasoning)
#   - VL model:  qwen3.8-max (multimodal, replaces qwen-vl-plus)
#   - OCR model: chandra-ocr-2 (unchanged — local GPU model)
#
# Previous outputs preserved:
#   - quantitative_results_ocr/chandra_ocr_2/  (OCR, unchanged)
#   - numeric_extracts/                         (extraction, unchanged unless --reextract)
#   - quantitative_results_Qwen/                (old qwen3.7-max results)
# New outputs:
#   - quantitative_results_qwen38/              (new qwen3.8-max results)
#
# Usage:
#   sbatch run_qwen38_cluster.sh                # Full run (matching only, reuses OCR)
#   sbatch run_qwen38_cluster.sh --reextract    # Also re-run extraction+vision fallback
#   sbatch run_qwen38_cluster.sh --full         # Full re-run including OCR (slow!)
# ==========================================================================

set -euo pipefail

module load apps/anaconda3/2021.05
source activate "$HOME/envs/chandra"
cd "$HOME/esg-pipeline" || { echo "ERROR: ~/esg-pipeline not found"; exit 1; }

# --- Pull latest code ---
echo "=== Git pull ==="
git pull || echo "WARNING: git pull failed, using existing code"

# --- Model configuration ---
export QWEN_MAX_MODEL="qwen3.8-max"
export ENABLE_THINKING="true"
export THINKING_MAX_TOKENS="16000"
export QWEN_VL_MODEL="qwen3.8-max"
export QUANTITATIVE_RESULT_DIR="quantitative_results_qwen38"

# --- OCR configuration (unchanged) ---
export MODEL_CHECKPOINT="$HOME/models/chandra-ocr-2"
export HF_HUB_OFFLINE=1
export PDF_INPUT_DIR="HKEX ESG Reports"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

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

# --- Determine mode ---
MODE="${1:-matching}"
echo "=== Mode: $MODE ==="
echo "  Text LLM:  $QWEN_MAX_MODEL (thinking=$ENABLE_THINKING)"
echo "  VL model:  $QWEN_VL_MODEL"
echo "  OCR model: chandra-ocr-2 (unchanged)"
echo "  Output:    $QUANTITATIVE_RESULT_DIR"
echo ""

case "$MODE" in
    --reextract)
        # Re-run extraction (includes vision fallback with new VL model) + matching
        echo "=== Step 1: Re-run numeric extraction (vision fallback with $QWEN_VL_MODEL) ==="
        export REUSE_CACHED_OCR=1   # Skip OCR, reuse cached chandra_ocr_2 output
        python -m src.main --step numeric_extraction --force > "logs/extract_qwen38_${JOB_TAG}.log" 2>&1
        echo "Extraction done. See logs/extract_qwen38_${JOB_TAG}.log"

        echo "=== Step 2: LLM matching with $QWEN_MAX_MODEL (thinking=$ENABLE_THINKING) ==="
        python -m src.main --step llm_matching --force > "logs/matching_qwen38_${JOB_TAG}.log" 2>&1
        echo "Matching done. See logs/matching_qwen38_${JOB_TAG}.log"
        ;;

    --full)
        # Full re-run including OCR (very slow, same OCR model = same results)
        echo "=== WARNING: Full re-run including OCR. This will take a long time. ==="
        echo "=== The OCR model (chandra-ocr-2) has NOT changed, so OCR results ==="
        echo "=== will be identical. Consider --reextract or default (matching only). ==="
        export REUSE_CACHED_OCR=0
        python -m src.main --step full_pipeline --force > "logs/full_qwen38_${JOB_TAG}.log" 2>&1
        echo "Full pipeline done. See logs/full_qwen38_${JOB_TAG}.log"
        ;;

    *)
        # Default: LLM matching only (fastest, most impactful change)
        echo "=== LLM matching only (reusing cached OCR + numeric_extracts) ==="
        echo "=== Use --reextract to also re-run vision fallback with new VL model ==="
        echo "=== Use --full to re-run everything including OCR (not recommended) ==="
        echo ""
        python run_qwen38_matching.py --resume 2>&1 | tee "logs/matching_qwen38_${JOB_TAG}.log"
        ;;
esac

echo ""
echo "================================================================"
echo "  Pipeline complete!"
echo "  Results: $QUANTITATIVE_RESULT_DIR/"
echo "  Old results preserved: quantitative_results_Qwen/"
echo "================================================================"
