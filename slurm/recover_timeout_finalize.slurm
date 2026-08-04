#!/bin/bash
#SBATCH --job-name=castillo_recover_final
#SBATCH --partition=cpu_partition
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/recover_finalize_%j.out
#SBATCH --error=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/recover_finalize_%j.err

set -euo pipefail

ROOT="${HOME}/castillo_stability_campaign"
RECOVERED="${HOME}/fscratch/castillo_stability_campaign/results/recovered"

module purge
module load anaconda

export PYTHONNOUSERSITE=1

cd "${ROOT}"

python3 code/summarize_recovered_campaign.py \
  --manifest reports/canonical/canonical_manifest.csv \
  --recovered-root "${RECOVERED}" \
  --report-dir reports/recovered

echo "CASTILLO_RECOVER_FINALIZE_OK"
