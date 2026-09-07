#!/bin/bash
#SBATCH --job-name=cst_nu6_fin
#SBATCH --partition=cpu_parti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/home/users/tic_118_uma/frv/castillo_lapack_baseline/logs/nu6_finalize_%j.out
#SBATCH --error=/mnt/home/users/tic_118_uma/frv/castillo_lapack_baseline/logs/nu6_finalize_%j.err
set -euo pipefail
module purge
module load python/3.11.4
cd "$HOME/castillo_lapack_baseline"
python3 code/spectral_estimator_validation_finalize.py
echo SPECTRAL_ESTIMATOR_VALIDATION_FINALIZE_SLURM_OK
