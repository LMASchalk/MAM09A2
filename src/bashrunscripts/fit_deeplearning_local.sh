#!/usr/bin/env bash
# Local machine (no cluster): train the deep learning model on CPU. Run with: bash fit_deeplearning.sh
# Note: CPU training at 28px is slow but fine for debugging the pipeline.

CONDA_BASE=$(conda info --base 2>/dev/null)

if [ -z "$CONDA_BASE" ]; then
  echo "Error: conda not found. Please install Miniconda first."
  exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate dl_cpu

python src/models/fit_deeplearning.py --modeltype "CNN" --epochs 40
