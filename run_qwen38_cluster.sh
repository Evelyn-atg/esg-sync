#!/bin/bash
#SBATCH -p A800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=1440
#SBATCH -o qwen38_pipeline_%j.out

# ==========================================================================
# Qwen3.8-Max Full Pipeline Runner (Cluster) — v2: full output separation
# ==========================================================================
# Re-runs the ESG pipeline for 100 GT PDFs with:
#   - Text LLM:  qwen3.8-max + enable_thinking=true (LLM matching + calculation)
#   - VL model:  qwen3.8-max (multimodal, replaces qwen-vl-plus in image_recognizer)
#   - OCR model: chandra-ocr-2 (unchanged — local GPU model)
#
# ALL outputs go to SEPARATE directories so you can compare qwen3.7 vs qwen3.8:
#
#   Old (qwen3.7-max + qwen-vl-plus):         New (qwen3.8-max + thinking):
#   ─────────────────────────────────────────  ────────────────────────────────────────
#   quantitative_results_ocr/chandra_ocr_2/   quantitative_results_ocr/chandra_ocr_2_qwen38/  (--full only)
#   numeric_extracts/                         numeric_extracts_qwen38/                       (--reextract / --full)
#   quantitative_results_Qwen/                quantitative_results_qwen38/                   (ALL modes)
#   calculation_results/                      calculation_results_qwen38/                    (ALL modes)
#
# Usage:
#   sbatch run_qwen38_cluster.sh                  # LLM matching + calculation only (fastest)
#   sbatch run_qwen38_cluster.sh --reextract      # + numeric extraction (re-runs VL via image_recognizer)
#   sbatch run_qwen38_cluster.sh --full           # Full re-run incl. OCR (slowest, OCR model unchanged)
# ==========================================================================

set -euo pipefail

module load apps/anaconda3/2021.05
source activate "$HOME/envs/chandra"
cd "$HOME/esg-pipeline" || { echo "ERROR: ~/esg-pipeline not found"; exit 1; }

# --- Pull latest code ---
echo "=== Git pull ==="
git pull || echo "WARNING: git pull failed, using existing code"

# ==========================================================================
# Model configuration — BOTH steps use qwen3.8-max + thinking
# ==========================================================================
export QWEN_MAX_MODEL="qwen3.8-max"
export ENABLE_THINKING="true"              # <-- enables thinking for BOTH LLM matching AND calculation
export THINKING_MAX_TOKENS="16000"
export QWEN_VL_MODEL="qwen3.8-max"         # <-- VL model (used by image_recognizer.py)

# ==========================================================================
# Output directory separation — ALL dirs are qwen38-specific
# ==========================================================================
export QUANTITATIVE_RESULT_DIR="quantitative_results_qwen38"      # LLM matching output
export RESULT_OUTPUT_DIR="calculation_results_qwen38"             # Calculation output

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
echo "=========================================================================="
echo "  Qwen3.8-Max Pipeline  (v2 — full output separation)"
echo "=========================================================================="
echo "  Mode:                $MODE"
echo "  Text LLM:            $QWEN_MAX_MODEL (thinking=$ENABLE_THINKING)"
echo "  VL model:            $QWEN_VL_MODEL"
echo "  OCR model:           chandra-ocr-2 (unchanged)"
echo ""
echo "  Output directories (NEW, separate from old):"
echo "    QUANTITATIVE_RESULT_DIR = $QUANTITATIVE_RESULT_DIR  (LLM matching)"
echo "    RESULT_OUTPUT_DIR       = $RESULT_OUTPUT_DIR  (calculation)"
case "$MODE" in
    --reextract|--full)
        export NUMERIC_EXTRACT_DIR="numeric_extracts_qwen38"
        echo "    NUMERIC_EXTRACT_DIR     = $NUMERIC_EXTRACT_DIR  (numeric extraction)"
        ;;
esac
if [ "$MODE" = "--full" ]; then
    export CHANDRA_OCR_RESULT_DIR="quantitative_results_ocr/chandra_ocr_2_qwen38"
    echo "    CHANDRA_OCR_RESULT_DIR  = $CHANDRA_OCR_RESULT_DIR  (OCR output)"
fi
echo ""
echo "  Old results preserved:"
echo "    quantitative_results_Qwen/  (old LLM matching, qwen3.7-max)"
echo "    calculation_results/        (old calculation, qwen-max)"
echo "    numeric_extracts/           (old numeric extraction)"
echo "    quantitative_results_ocr/chandra_ocr_2/  (old OCR)"
echo "=========================================================================="
echo ""

case "$MODE" in
    --reextract)
        # Re-run numeric extraction (includes VL via image_recognizer) + LLM matching + calculation
        echo "=== Step 1/3: Numeric extraction (VL model=$QWEN_VL_MODEL, thinking=$ENABLE_THINKING) ==="
        export REUSE_CACHED_OCR=1
        python -m src.main --step numeric_extraction --force > "logs/extract_qwen38_${JOB_TAG}.log" 2>&1
        echo "Extraction done. See logs/extract_qwen38_${JOB_TAG}.log"

        echo "=== Step 2/3: LLM matching ($QWEN_MAX_MODEL, thinking=$ENABLE_THINKING) ==="
        python -m src.main --step llm_matching --force > "logs/matching_qwen38_${JOB_TAG}.log" 2>&1
        echo "Matching done. See logs/matching_qwen38_${JOB_TAG}.log"

        echo "=== Step 3/3: Calculation ($QWEN_MAX_MODEL, thinking=$ENABLE_THINKING) ==="
        python -m src.main --step calculation --force > "logs/calc_qwen38_${JOB_TAG}.log" 2>&1
        echo "Calculation done. See logs/calc_qwen38_${JOB_TAG}.log"
        ;;

    --full)
        # Full re-run including OCR (same OCR model = same OCR results, but VL changes)
        echo "=== WARNING: Full re-run including OCR. ==="
        echo "=== The OCR model (chandra-ocr-2) has NOT changed, so OCR results ==="
        echo "=== will be identical. Only VL fallback + downstream changes. ==="
        echo "=== Consider --reextract for faster turnaround. ==="
        echo ""
        export REUSE_CACHED_OCR=0

        echo "=== Step 1/4: OCR (chandra-ocr-2, VL fallback=$QWEN_VL_MODEL) ==="
        python -m src.main --step numeric_extraction --force > "logs/ocr_qwen38_${JOB_TAG}.log" 2>&1
        echo "OCR done. See logs/ocr_qwen38_${JOB_TAG}.log"

        echo "=== Step 2/4: (numeric extraction already included in step 1) ==="

        echo "=== Step 3/4: LLM matching ($QWEN_MAX_MODEL, thinking=$ENABLE_THINKING) ==="
        python -m src.main --step llm_matching --force > "logs/matching_qwen38_${JOB_TAG}.log" 2>&1
        echo "Matching done. See logs/matching_qwen38_${JOB_TAG}.log"

        echo "=== Step 4/4: Calculation ($QWEN_MAX_MODEL, thinking=$ENABLE_THINKING) ==="
        python -m src.main --step calculation --force > "logs/calc_qwen38_${JOB_TAG}.log" 2>&1
        echo "Calculation done. See logs/calc_qwen38_${JOB_TAG}.log"
        ;;

    *)
        # Default: LLM matching + calculation only (fastest, reuses cached OCR + extraction)
        # Reads from OLD numeric_extracts/ (unchanged), writes to NEW dirs
        echo "=== LLM matching + calculation only ==="
        echo "=== Reads from: numeric_extracts/ (old, same extraction) ==="
        echo "=== Writes to:  $QUANTITATIVE_RESULT_DIR/ + $RESULT_OUTPUT_DIR/ ==="
        echo "=== Use --reextract to also re-run extraction with new VL model ==="
        echo "=== Use --full to re-run everything including OCR ==="
        echo ""

        echo "=== Step 1/2: LLM matching ($QWEN_MAX_MODEL, thinking=$ENABLE_THINKING) ==="
        python -m src.main --step llm_matching --force > "logs/matching_qwen38_${JOB_TAG}.log" 2>&1
        echo "Matching done. See logs/matching_qwen38_${JOB_TAG}.log"

        echo "=== Step 2/2: Calculation ($QWEN_MAX_MODEL, thinking=$ENABLE_THINKING) ==="
        python -m src.main --step calculation --force > "logs/calc_qwen38_${JOB_TAG}.log" 2>&1
        echo "Calculation done. See logs/calc_qwen38_${JOB_TAG}.log"
        ;;
esac

echo ""
echo "=========================================================================="
echo "  Pipeline complete!"
echo ""
echo "  New results (qwen3.8-max + thinking):"
echo "    $QUANTITATIVE_RESULT_DIR/  (LLM matching)"
echo "    $RESULT_OUTPUT_DIR/        (calculation)"
case "$MODE" in
    --reextract|--full) echo "    $NUMERIC_EXTRACT_DIR/       (numeric extraction)" ;;
esac
[ "$MODE" = "--full" ] && echo "    $CHANDRA_OCR_RESULT_DIR/  (OCR)"
echo ""
echo "  Old results (qwen3.7-max, preserved):"
echo "    quantitative_results_Qwen/"
echo "    calculation_results/"
echo "    numeric_extracts/"
echo "    quantitative_results_ocr/chandra_ocr_2/"
echo "=========================================================================="
