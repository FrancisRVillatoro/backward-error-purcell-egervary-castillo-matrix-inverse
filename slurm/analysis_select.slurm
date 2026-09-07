#!/bin/bash
#SBATCH --job-name=castillo_an_select
#SBATCH --partition=cpu_partition
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/analysis_select_%j.out
#SBATCH --error=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/analysis_select_%j.err

set -euo pipefail

ROOT="${HOME}/castillo_stability_campaign"
RECOVERED="${HOME}/fscratch/castillo_stability_campaign/results/recovered"
ANALYSIS="${HOME}/fscratch/castillo_stability_campaign/results/analysis"
REPORTS="${ROOT}/reports/analysis"

module purge
module load anaconda

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${ROOT}"

python3 -m py_compile code/analyze_recovered.py

rm -rf "${REPORTS}"
mkdir -p "${REPORTS}"

python3 code/analyze_recovered.py select \
  --manifest reports/canonical/canonical_manifest.csv \
  --status-dir "${RECOVERED}/status" \
  --outdir "${ANALYSIS}" \
  --summary "${REPORTS}/selection_summary.json" \
  --num-tasks 999 \
  --sample-size 2000 \
  --max-n 512

echo "CASTILLO_ANALYSIS_SELECT_OK"
