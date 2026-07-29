#!/bin/bash
#SBATCH -p H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o ocr_3stream_%j.out

module load apps/anaconda3/2021.05
source activate "$HOME/envs/chandra"
cd "$HOME/esg-pipeline" || exit 1

export MODEL_CHECKPOINT="$HOME/models/chandra-ocr-2"
export HF_HUB_OFFLINE=1
export PDF_INPUT_DIR="HKEX ESG Reports"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# 3 streams × batch=16 = 48 concurrent images
# Model is 9.9GB, per-process: 10GB + 16×0.5GB + 3GB = 21GB
# 3 processes: ~63GB, safe for H100 80GB
export OCR_BATCH_SIZE=16

mkdir -p logs

# Tag log files with Slurm job id so 2 GPU runs don't collide
JOB_TAG="${SLURM_JOB_ID:-local}"

for lst in list_00 list_01 list_02; do
  [ -f "$lst" ] || continue
  echo "launching OCR stream for $lst (job $JOB_TAG, 3-stream mode, batch=16)"
  python -m src.main --step numeric_extraction --pdf_list_file "$lst" --force > "logs/ocr_${lst}_job${JOB_TAG}.log" 2>&1 &
done
wait

echo "=== all 3 streams done (job $JOB_TAG) ==="
