#!/bin/bash
#SBATCH -p A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH -o ocr_a100_a_%j.out

module load apps/anaconda3/2021.05
source activate "$HOME/envs/chandra"
cd "$HOME/esg-pipeline" || exit 1

export MODEL_CHECKPOINT="$HOME/models/chandra-ocr-2"
export HF_HUB_OFFLINE=1
export PDF_INPUT_DIR="HKEX ESG Reports"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OCR_BATCH_SIZE=16

mkdir -p logs
JOB_TAG="${SLURM_JOB_ID:-local}"

echo "launching OCR stream for list_ocr_remaining_a (job $JOB_TAG, A100)"
python -m src.main --step numeric_extraction --pdf_list_file list_ocr_remaining_a.txt --force > "logs/ocr_list_ocr_remaining_a_job${JOB_TAG}.log" 2>&1

echo "=== A100 done (job $JOB_TAG) ==="
