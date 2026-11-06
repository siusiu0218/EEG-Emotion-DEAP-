#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --job-name=GTE_repair
#SBATCH --chdir=/share/home/ycg_luanjingjie/GTE-Net
#SBATCH --output=/share/home/ycg_luanjingjie/GTE-Net/logs/slurm-%j.out
#SBATCH --error=/share/home/ycg_luanjingjie/GTE-Net/logs/slurm-%j.err
set -euo pipefail
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PYTHON="$PWD/.venv/bin/python"
RUN_DIR="$PWD/runs/repair/${SLURM_JOB_ID:?}"
mkdir -p "$RUN_DIR/small" "$RUN_DIR/full"
sha256sum GTE_train.py DatasetProcessSave.py model_optimized.py validate_training_runtime.py validate_smoke_outputs.py > "$RUN_DIR/code.sha256"
cp manifests/environment.txt "$RUN_DIR/environment-snapshot-20261014.txt"
date -Is
"$PYTHON" validate_training_runtime.py
"$PYTHON" validate_setup.py
COMMON=(--dataset DEAP --data_path "$PWD/dataset/DEAP/data_preprocessed_python"
  --preprocessed_data_path "$PWD/dataset/DEAP/data_time_window_1s"
  --epochs 1 --batch_size 32 --lr 1e-4 --weight_decay 1e-4 --time_window 1
  --gat_heads 2 --transformer_heads 4 --transformer_layers 1 --d_model 128
  --ffn_dim 512 --data_augmentation 0.3 --early_stopping_patience 100)
echo SMALL_START
"$PYTHON" GTE_train.py "${COMMON[@]}" --subject_ids 1 --max_samples 320 --model_save_path "$RUN_DIR/small/GTE_best.pth"
"$PYTHON" validate_smoke_outputs.py --run_dir "$RUN_DIR/small"
echo SMALL_PASSED
date -Is
echo FULL_START
"$PYTHON" GTE_train.py "${COMMON[@]}" --model_save_path "$RUN_DIR/full/GTE_best.pth"
echo FULL_PASSED
date -Is
