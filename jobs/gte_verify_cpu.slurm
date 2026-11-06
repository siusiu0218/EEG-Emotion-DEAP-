#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --job-name=GTE_verify
#SBATCH --chdir=/share/home/ycg_luanjingjie/GTE-Net
#SBATCH --output=/share/home/ycg_luanjingjie/GTE-Net/logs/slurm-%j.out
#SBATCH --error=/share/home/ycg_luanjingjie/GTE-Net/logs/slurm-%j.err
set -euo pipefail
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
RUN_ID="${1:?Pass the completed training job ID}"
[[ "$RUN_ID" =~ ^[0-9]+$ ]]
sha256sum validate_smoke_outputs.py
.venv/bin/python validate_smoke_outputs.py --run_dir "runs/repair/$RUN_ID/full" --full
