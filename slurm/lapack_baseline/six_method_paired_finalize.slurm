#!/bin/bash
#SBATCH --job-name=cst_pair_fin
#SBATCH --partition=cpu_parti
#SBATCH --constraint=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/users/tic_118_uma/frv/castillo_lapack_baseline/logs/pair_finalize_%j.out
#SBATCH --error=/mnt/home/users/tic_118_uma/frv/castillo_lapack_baseline/logs/pair_finalize_%j.err

module purge
module load python/3.11.4

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 \
  "$HOME/castillo_lapack_baseline/code/finalize_six_method_paired.py"
