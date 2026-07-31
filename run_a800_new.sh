#!/bin/bash
#SBATCH -p A800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o ocr_a800_new_%j.out

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

echo "launching OCR stream for list_remaining_b (job $JOB_TAG, A800 new)"
python -m src.main --step numeric_extraction --pdf_list_file list_remaining_b.txt --force > "logs/ocr_list_remaining_b_job${JOB_TAG}.log" 2>&1

echo "=== A800 new done (job $JOB_TAG) ==="
