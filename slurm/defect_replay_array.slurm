#!/bin/bash
#SBATCH --job-name=cast_D2000
#SBATCH --constraint=cpu
#SBATCH --array=0-999
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cast_D2000_%A_%a.out
#SBATCH --error=logs/cast_D2000_%A_%a.err

set -euo pipefail

cd "$HOME/castillo_stability_campaign"

module purge
module load python/3.11.4
export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

sha256sum -c \
    config/defect_replay_source_hashes.sha256

python3 code/defect_replay.py \
    --task-id "$SLURM_ARRAY_TASK_ID"
