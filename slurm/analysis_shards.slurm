#!/bin/bash
#SBATCH --job-name=castillo_an_shard
#SBATCH --partition=cpu_partition
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --array=0-998%100
#SBATCH --output=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/analysis_shard_%A_%a.log
#SBATCH --error=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/analysis_shard_%A_%a.log

set -euo pipefail

ROOT="${HOME}/castillo_stability_campaign"
RECOVERED="${HOME}/fscratch/castillo_stability_campaign/results/recovered"
ANALYSIS="${HOME}/fscratch/castillo_stability_campaign/results/analysis"

module purge
module load anaconda

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${ROOT}"

mkdir -p "${ANALYSIS}/partials"

python3 code/analyze_recovered.py shard \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --manifest reports/canonical/canonical_manifest.csv \
  --config config/canonical.json \
  --recovered-root "${RECOVERED}" \
  --selection-dir "${ANALYSIS}/selection" \
  --outdir "${ANALYSIS}/partials" \
  --sample-size 2000 \
  --core-max-n 181 \
  --censor-n 181
