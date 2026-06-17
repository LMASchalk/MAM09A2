#!/usr/bin/env bash
# Snellius: create the conda environment.
# Run on a LOGIN NODE with: bash setup_env.sh (needs internet; do NOT submit with sbatch)

module purge
module load 2025
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"

cd "$HOME/MAM09A2"
conda env create -f dl_gpu.yml
