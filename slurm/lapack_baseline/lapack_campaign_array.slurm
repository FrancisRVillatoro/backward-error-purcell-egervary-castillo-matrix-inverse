#!/bin/bash
#SBATCH --job-name=cst_lap_full
#SBATCH --partition=cpu_parti
#SBATCH --constraint=cpu
#SBATCH --array=0-998%64
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/users/tic_118_uma/frv/castillo_lapack_baseline/logs/campaign_%A_%a.out
#SBATCH --error=/mnt/home/users/tic_118_uma/frv/castillo_lapack_baseline/logs/campaign_%A_%a.err

module purge
module load python/3.11.4

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 \
  "$HOME/castillo_lapack_baseline/code/run_lapack_baseline_task.py" \
  "$SLURM_ARRAY_TASK_ID"
