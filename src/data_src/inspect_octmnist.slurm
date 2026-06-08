#!/bin/bash
#SBATCH --job-name=octmnist_inspect
#SBATCH --partition=rome            # CPU partition on Snellius (confirm with `sinfo`)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G                   # the 64 train array is ~5 GB in RAM
#SBATCH --time=00:15:00
#SBATCH --output=octmnist_inspect_%j.out

# --- Inspection job (CPU only, no internet needed) -------------------------
# Run from the repo root. The data must already be downloaded (run
# inspect_octmnist.py --download-only on a login node first). This job reads the
# cached data/raw/octmnist_64.npz and writes reports/inspection_report.md plus
# reports/figures/{class_distribution,sample_grid}.png.

set -euo pipefail

# Adjust the module stack/version to whatever `module avail Python` shows.
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

source "$HOME/venvs/octmnist/bin/activate"

cd "$HOME/MAM09A2"                  # repo root

# Paths default to <repo>/data/raw and <repo>/reports, so no path flags needed.
python src/data_src/inspect_octmnist.py \
    --size 64 \
    --sample-split train \
    --n-per-class 6
