#!/bin/bash
#SBATCH --job-name=cast_comp
#SBATCH --constraint=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cast_comp_%j.out
#SBATCH --error=logs/cast_comp_%j.err

set -euo pipefail

cd "$HOME/castillo_stability_campaign"

module purge
module load python/3.11.4

export PYTHONNOUSERSITE=1

python3 code/analyze_compensation.py
