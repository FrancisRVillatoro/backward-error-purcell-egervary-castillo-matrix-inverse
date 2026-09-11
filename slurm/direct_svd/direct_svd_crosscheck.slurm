#!/bin/bash -l
#SBATCH --job-name=cst_svd2496
#SBATCH --partition=cpu_parti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-831
#SBATCH --output=/mnt/home/users/tic_118_uma/frv/castillo_lapack_baseline/logs/svd2496_%A_%a.out
#SBATCH --error=/mnt/home/users/tic_118_uma/frv/castillo_lapack_baseline/logs/svd2496_%A_%a.err

module purge
module load anaconda/2022.10
MODULE_RC=$?

if [ "$MODULE_RC" -ne 0 ]; then
    echo "DIRECT_SVD_CROSSCHECK_MODULE_FAIL rc=$MODULE_RC"
    RC=$MODULE_RC
else
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1

    cd "$HOME/castillo_lapack_baseline"
    CD_RC=$?
    if [ "$CD_RC" -ne 0 ]; then
        echo "DIRECT_SVD_CROSSCHECK_CD_FAIL rc=$CD_RC"
        RC=$CD_RC
    else
        python3 code/direct_svd_crosscheck.py
        RC=$?
        if [ "$RC" -eq 0 ]; then
            echo "DIRECT_SVD_CROSSCHECK_SLURM_OK"
        else
            echo "DIRECT_SVD_CROSSCHECK_SLURM_FAIL rc=$RC"
        fi
    fi
fi

test "$RC" -eq 0
