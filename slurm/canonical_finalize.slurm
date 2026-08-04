#!/bin/bash
#SBATCH --job-name=castillo_finalize
#SBATCH --partition=cpu_partition
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/canonical_finalize_%j.out
#SBATCH --error=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/canonical_finalize_%j.err

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

python3 code/aggregate.py \
  --manifest-summary reports/canonical/canonical_manifest_summary.json \
  --execution config/execution.json \
  --outdir "${OUT}" \
  --report-dir reports/canonical

echo "CASTILLO_CANONICAL_FINALIZE_OK"
