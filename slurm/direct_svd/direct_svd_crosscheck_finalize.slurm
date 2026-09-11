#!/bin/bash -l
#SBATCH --job-name=cst_svd2496_fin
#SBATCH --partition=cpu_parti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
module purge
module load anaconda/2022.10 || exit $?
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CASTILLO_REPRO_ROOT="${CASTILLO_REPRO_ROOT:-${SLURM_SUBMIT_DIR}}"
export CASTILLO_DIRECT_SVD_WORKDIR="${CASTILLO_DIRECT_SVD_WORKDIR:-${CASTILLO_REPRO_ROOT}/work/direct_svd_crosscheck}"
cd "$CASTILLO_REPRO_ROOT" || exit $?
python3 code/direct_svd/direct_svd_crosscheck_finalize.py || exit $?
python3 code/direct_svd/direct_svd_methodwise_pair_bootstrap.py || exit $?
echo DIRECT_SVD_FINALIZE_SLURM_OK
