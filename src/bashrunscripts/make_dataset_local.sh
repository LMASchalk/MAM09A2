#!/usr/bin/env bash
# Local machine (no cluster): download the OCTMNIST dataset. Run with: bash make_dataset.sh

CONDA_BASE=$(conda info --base 2>/dev/null)

if [ -z "$CONDA_BASE" ]; then
  echo "Error: conda not found. Please install Miniconda first."
  exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate dl_cpu

python src/data_src/make_dataset.py --size 28
