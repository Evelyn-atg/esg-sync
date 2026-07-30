#!/bin/bash
#SBATCH -p A800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o ocr_3stream_gpu1_%j.out

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

for lst in list_03 list_04 list_05; do
  [ -f "$lst" ] || continue
  echo "launching OCR stream for $lst (job $JOB_TAG, GPU 1)"
  python -m src.main --step numeric_extraction --pdf_list_file "$lst" --force > "logs/ocr_${lst}_job${JOB_TAG}.log" 2>&1 &
done
wait
echo "=== GPU 1 done (job $JOB_TAG) ==="
