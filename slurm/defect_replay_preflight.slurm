#!/bin/bash
#SBATCH --job-name=cast_Dpre
#SBATCH --constraint=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cast_Dpre_%j.out
#SBATCH --error=logs/cast_Dpre_%j.err

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
    --task-id 0 \
    --preflight

echo "CASTILLO_DEFECT_PREFLIGHT_OK"
