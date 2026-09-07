#!/bin/bash
#SBATCH --job-name=castillo_manifest
#SBATCH --partition=cpu_partition
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/manifest_%j.out
#SBATCH --error=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/manifest_%j.err

set -euo pipefail

ROOT="${HOME}/castillo_stability_campaign"

module purge
module load anaconda

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${ROOT}"

rm -f reports/smoke/smoke_manifest.csv
rm -f reports/smoke/smoke_manifest_summary.json
rm -f reports/smoke/smoke_manifest_summary.md

rm -f reports/canonical/canonical_manifest.csv
rm -f reports/canonical/canonical_manifest_summary.json
rm -f reports/canonical/canonical_manifest_summary.md

python3 code/manifest.py \
  --config config/smoke.json \
  --outdir reports/smoke

python3 code/manifest.py \
  --config config/canonical.json \
  --outdir reports/canonical

echo "CASTILLO_MANIFEST_SLURM_OK"
