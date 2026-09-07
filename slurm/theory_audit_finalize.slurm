#!/bin/bash
#SBATCH --job-name=castillo_theory_final
#SBATCH --partition=cpu_partition
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/theory_audit_finalize_%j.out
#SBATCH --error=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/theory_audit_finalize_%j.err

set -euo pipefail

ROOT="${HOME}/castillo_stability_campaign"
ANALYSIS="${HOME}/fscratch/castillo_stability_campaign/results/analysis"
PARTIAL="${ANALYSIS}/theory_audit/partials"
TMP="${ANALYSIS}/theory_audit/tmp_finalize"
REPORT="${ROOT}/reports/theory_audit"

module purge
module load anaconda

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${ROOT}"
mkdir -p "${REPORT}" "${TMP}"

python3 code/theory_audit.py finalize \
  --partial-dir "${PARTIAL}" \
  --report-dir "${REPORT}" \
  --selection-summary "${ROOT}/reports/analysis/selection_summary.json" \
  --power-fits "${ROOT}/reports/analysis/inverse_power_fits_by_cell.csv" \
  --bootstrap-slopes "${ROOT}/reports/analysis/bootstrap_power_slopes.csv" \
  --tmpdir "${TMP}" \
  --num-tasks 999 \
  --sample-size 2000

python3 -m json.tool "${REPORT}/theory_audit_summary.json"

tar -czf "${HOME}/castillo_theory_audit_reports.tar.gz" \
  -C "${ROOT}" reports/theory_audit

# The canonical recovered shards remain untouched.  Remove only the
# regenerable theory-audit partials after the final report is safely built.
rm -rf "${PARTIAL}" "${TMP}"

echo "CASTILLO_THEORY_AUDIT_FINAL_OK"
