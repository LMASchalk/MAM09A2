#!/usr/bin/env bash
# Local machine (no cluster): train the LinearSVC baseline. Run with: bash fit_linearsvc.sh

CONDA_BASE=$(conda info --base 2>/dev/null)

if [ -z "$CONDA_BASE" ]; then
  echo "Error: conda not found. Please install Miniconda first."
  exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate dl_cpu

python src/models/fit_linearSVC.py
