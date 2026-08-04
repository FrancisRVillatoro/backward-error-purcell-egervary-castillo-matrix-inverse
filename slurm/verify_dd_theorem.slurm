#!/bin/bash
#SBATCH --job-name=castillo_dd_verify
#SBATCH --partition=cpu_partition
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/verify_dd_theorem_%j.out
#SBATCH --error=/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/logs/verify_dd_theorem_%j.err

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

python3 code/verify_dd_theorem.py

echo "CASTILLO_DD_THEOREM_SLURM_OK"
