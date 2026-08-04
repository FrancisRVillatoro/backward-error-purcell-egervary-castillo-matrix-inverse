#!/bin/bash
#SBATCH --job-name=castillo_an_report
#SBATCH --partition=cpu_partition
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/analysis_report_%j.out
#SBATCH --error=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/analysis_report_%j.err

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
export MPLBACKEND=Agg

cd "${ROOT}"

python3 code/analyze_recovered.py report \
  --report-dir "${REPORTS}" \
  --bootstrap-dir "${ANALYSIS}/bootstrap" \
  --completion-by-n reports/recovered/recovered_completion_by_n.csv \
  --bootstrap-replicates 2000

echo "CASTILLO_ANALYSIS_REPORT_OK"
