#!/bin/bash
#SBATCH --job-name=castillo_an_final
#SBATCH --partition=cpu_partition
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/analysis_finalize_%j.out
#SBATCH --error=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/analysis_finalize_%j.err

set -euo pipefail

ROOT="${HOME}/castillo_stability_campaign"
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

python3 code/analyze_recovered.py finalize \
  --manifest reports/canonical/canonical_manifest.csv \
  --config config/canonical.json \
  --analysis-config config/analysis.json \
  --selection-summary "${REPORTS}/selection_summary.json" \
  --partial-dir "${ANALYSIS}/partials" \
  --report-dir "${REPORTS}" \
  --tmpdir "${ANALYSIS}/tmp_finalize" \
  --bootstrap-input-dir "${ANALYSIS}/bootstrap_input" \
  --num-tasks 999

echo "CASTILLO_ANALYSIS_FINALIZE_OK"
