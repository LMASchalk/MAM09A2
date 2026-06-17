#!/usr/bin/env bash
# Snellius: download the OCTMNIST dataset (cached for later offline jobs).
# Run on a LOGIN NODE with: bash make_dataset.sh (needs internet; do NOT submit with sbatch)

module purge
module load 2025
module load Anaconda3/2025.06-1

eval "$(conda shell.bash hook)"
conda activate dl_gpu

cd "$HOME/MAM09A2"
python src/data_src/make_dataset.py --size 28
