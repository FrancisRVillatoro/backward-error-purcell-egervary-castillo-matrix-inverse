#!/bin/bash
#SBATCH --job-name=cst_metric_scan
#SBATCH --partition=cpu_parti
#SBATCH --constraint=cpu
#SBATCH --array=0-998%64
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/users/tic_118_uma/frv/fscratch/castillo_lapack_paired/full_metric_scan/logs/metric_scan_%A_%a.log

module purge
module load python/3.11.4

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 \
  "$HOME/castillo_lapack_baseline/code/scan_full_castillo_metric_task.py" \
  "$SLURM_ARRAY_TASK_ID"
