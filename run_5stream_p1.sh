#!/bin/bash
#SBATCH -p H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o ocr_5stream_p1_%j.out

module load apps/anaconda3/2021.05
source activate "$HOME/envs/chandra"
cd "$HOME/esg-pipeline" || exit 1

export MODEL_CHECKPOINT="$HOME/models/chandra-ocr-2"
export HF_HUB_OFFLINE=1
export PDF_INPUT_DIR="HKEX ESG Reports"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
mkdir -p logs

for lst in list_00 list_01 list_02 list_03 list_04; do
  [ -f "$lst" ] || continue
  echo "launching OCR stream for $lst (GPU1, 5-stream mode)"
  python -m src.main --step numeric_extraction --pdf_list_file "$lst" --force > "logs/ocr_${lst}.log" 2>&1 &
done
wait
