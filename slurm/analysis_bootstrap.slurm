#!/bin/bash
#SBATCH --job-name=castillo_an_boot
#SBATCH --partition=cpu_partition
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --array=0-639%320
#SBATCH --output=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/analysis_bootstrap_%A_%a.log
#SBATCH --error=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/analysis_bootstrap_%A_%a.log

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

mkdir -p "${ANALYSIS}/bootstrap"

python3 code/analyze_recovered.py bootstrap \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --index "${REPORTS}/bootstrap_index.json" \
  --outdir "${ANALYSIS}/bootstrap" \
  --replicates 2000
