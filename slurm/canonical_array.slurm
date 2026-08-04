#!/bin/bash
#SBATCH --job-name=castillo_canonical
#SBATCH --partition=cpu_partition
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-998%999
#SBATCH --output=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/canonical_%A_%a.log
#SBATCH --error=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/canonical_%A_%a.log

set -euo pipefail

ROOT="${HOME}/castillo_stability_campaign"
OUT="${HOME}/fscratch/castillo_stability_campaign/results/canonical"

module purge
module load anaconda

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${ROOT}"

python3 code/run_campaign.py \
  --config config/canonical.json \
  --manifest reports/canonical/canonical_manifest.csv \
  --execution config/execution.json \
  --outdir "${OUT}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --num-tasks 999

echo "CANONICAL_ARRAY_TASK_OK=${SLURM_ARRAY_TASK_ID}"
