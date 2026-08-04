#!/bin/bash
#SBATCH --job-name=castillo_recover
#SBATCH --partition=cpu_partition
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --array=0-998%100
#SBATCH --output=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/recover_%A_%a.log
#SBATCH --error=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/recover_%A_%a.log

set -euo pipefail

ROOT="${HOME}/castillo_stability_campaign"
SOURCE="${HOME}/fscratch/castillo_stability_campaign/results/canonical"
RECOVERED="${HOME}/fscratch/castillo_stability_campaign/results/recovered"

module purge
module load anaconda

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${ROOT}"

python3 code/recover_timeout_shard.py \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --source-root "${SOURCE}" \
  --output-root "${RECOVERED}" \
  --config config/canonical.json
