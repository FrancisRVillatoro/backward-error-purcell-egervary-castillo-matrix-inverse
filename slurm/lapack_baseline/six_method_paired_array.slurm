#!/bin/bash
#SBATCH --job-name=cst_pair_build
#SBATCH --partition=cpu_parti
#SBATCH --constraint=cpu
#SBATCH --array=0-998%32
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/users/tic_118_uma/frv/castillo_lapack_baseline/logs/pair_build_%A_%a.log

module purge
module load python/3.11.4

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 \
  "$HOME/castillo_lapack_baseline/code/build_six_method_paired_task.py" \
  "$SLURM_ARRAY_TASK_ID"
