#!/bin/bash
#SBATCH --job-name=castillo_prepare
#SBATCH --partition=cpu_partition
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/canonical_prepare_%j.out
#SBATCH --error=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/canonical_prepare_%j.err

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

python3 -m py_compile \
  code/families.py \
  code/castillo.py \
  code/run_campaign.py \
  code/aggregate.py

python3 - <<'PY'
import json
from pathlib import Path

execution = json.loads(
    Path("config/execution.json").read_text()
)

summary = json.loads(
    Path(
        "reports/canonical/"
        "canonical_manifest_summary.json"
    ).read_text()
)

assert execution["array_tasks"] == 999
assert summary["status"] == "MANIFEST_OK"
assert summary["num_matrix_cases"] == 19200975
assert summary["expected_inverse_records"] == 192009750
assert summary["expected_solution_records"] == 1728087750

print("CANONICAL_COUNTS_OK")
PY

rm -rf "${OUT}"

mkdir -p \
  "${OUT}/inverse" \
  "${OUT}/solution" \
  "${OUT}/status" \
  "${OUT}/tmp"

rm -f \
  reports/canonical/canonical_completion.json \
  reports/canonical/canonical_completion.md \
  reports/canonical/canonical_shards_manifest.csv

echo "CANONICAL_PREPARE_OK"
