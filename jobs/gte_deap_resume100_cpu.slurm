#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --job-name=GTE_resume100
#SBATCH --chdir=/share/home/ycg_luanjingjie/GTE-Net
#SBATCH --output=/share/home/ycg_luanjingjie/GTE-Net/logs/slurm-%j.out
#SBATCH --error=/share/home/ycg_luanjingjie/GTE-Net/logs/slurm-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
CODE_DIR="${1:?Pass the frozen code directory}"
SOURCE="$PWD/runs/deap_window_split_50/1887646/GTE_best_train_acc_0.6437174479166666_val_acc_0.6492838541666667.pth"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
PYTHON="$PWD/.venv/bin/python"
RUN_DIR="$PWD/runs/deap_window_split_resume100/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"
cp "$CODE_DIR/gte_deap_resume100_cpu.slurm" "$RUN_DIR/submitted-job.slurm"
(cd "$CODE_DIR" && sha256sum -c code.sha256)
cp "$CODE_DIR/code.sha256" "$RUN_DIR/code.sha256"
cp manifests/environment.txt "$RUN_DIR/environment-snapshot-20261014.txt"
sha256sum "$SOURCE" > "$RUN_DIR/source.sha256"
printf '%s\n' "$CODE_DIR" > "$RUN_DIR/code-directory.txt"
date -Is
hostname
srun "$PYTHON" "$CODE_DIR/validate_resume_training.py"
srun "$PYTHON" "$CODE_DIR/validate_training_runtime.py"
srun "$PYTHON" "$CODE_DIR/validate_resume_real.py" --run_dir "$RUN_DIR/resume_check" --source "$SOURCE"
echo RESUME_PREFLIGHT_PASSED
echo FULL_RESUME_100_START
srun "$PYTHON" -u "$CODE_DIR/GTE_train.py" \
  --dataset DEAP \
  --data_path "$PWD/dataset/DEAP/data_preprocessed_python" \
  --preprocessed_data_path "$PWD/dataset/DEAP/data_time_window_1s" \
  --epochs 100 --batch_size 32 --lr 1e-4 --weight_decay 1e-4 --time_window 1 \
  --gat_heads 2 --transformer_heads 4 --transformer_layers 1 --d_model 128 \
  --ffn_dim 512 --data_augmentation 0.3 --early_stopping_patience 100 \
  --resume_from "$SOURCE" --model_save_path "$RUN_DIR/GTE_best.pth"
sha256sum -c "$RUN_DIR/source.sha256"
echo FULL_RESUME_100_PASSED
date -Is
