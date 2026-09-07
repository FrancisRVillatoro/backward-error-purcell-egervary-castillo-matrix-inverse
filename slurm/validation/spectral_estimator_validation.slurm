#!/bin/bash
#SBATCH --job-name=cst_nu6
#SBATCH --partition=cpu_parti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-47%12
#SBATCH --output=/mnt/home/users/tic_118_uma/frv/castillo_lapack_baseline/logs/nu6_%A_%a.out
#SBATCH --error=/mnt/home/users/tic_118_uma/frv/castillo_lapack_baseline/logs/nu6_%A_%a.err
set -euo pipefail
module purge
module load python/3.11.4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$HOME/castillo_lapack_baseline/logs"
cd "$HOME/castillo_lapack_baseline"
python3 code/spectral_estimator_validation.py
echo SPECTRAL_ESTIMATOR_VALIDATION_SLURM_OK
