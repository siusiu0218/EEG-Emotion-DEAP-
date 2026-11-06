#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --job-name=GTE_DEAP50
#SBATCH --chdir=/share/home/ycg_luanjingjie/GTE-Net
#SBATCH --output=/share/home/ycg_luanjingjie/GTE-Net/logs/slurm-%j.out
#SBATCH --error=/share/home/ycg_luanjingjie/GTE-Net/logs/slurm-%j.err
set -euo pipefail
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PYTHON="$PWD/.venv/bin/python"
RUN_DIR="$PWD/runs/deap_window_split_50/${SLURM_JOB_ID:?}"
mkdir -p "$RUN_DIR"
cp jobs/gte_deap_50_cpu.slurm "$RUN_DIR/submitted-job.slurm"
sha256sum GTE_train.py DatasetProcessSave.py model_optimized.py > "$RUN_DIR/code.sha256"
cp manifests/environment.txt "$RUN_DIR/environment-snapshot-20261014.txt"
date -Is
hostname
echo FULL_50_START
"$PYTHON" -u GTE_train.py \
  --dataset DEAP \
  --data_path "$PWD/dataset/DEAP/data_preprocessed_python" \
  --preprocessed_data_path "$PWD/dataset/DEAP/data_time_window_1s" \
  --epochs 50 --batch_size 32 --lr 1e-4 --weight_decay 1e-4 --time_window 1 \
  --gat_heads 2 --transformer_heads 4 --transformer_layers 1 --d_model 128 \
  --ffn_dim 512 --data_augmentation 0.3 --early_stopping_patience 100 \
  --model_save_path "$RUN_DIR/GTE_best.pth"
echo FULL_50_PASSED
date -Is
