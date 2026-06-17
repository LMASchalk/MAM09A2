#!/usr/bin/env bash
# Local machine (no cluster): create the CPU conda environment.
# Run with: bash setup_env.sh

# Require conda to be installed and on PATH.
if ! command -v conda >/dev/null 2>&1; then
  echo "Error: 'conda' was not found on your PATH."
  echo
  echo "Install Miniconda first, then re-run this script:"
  echo "  https://www.anaconda.com/docs/getting-started/miniconda/install/overview"
  echo
  echo "After installing, open a new terminal (or run 'conda init' and"
  echo "restart your shell) so that 'conda' is available, then try again."
  exit 1
fi

# The environment file must be present in the current directory.
if [ ! -f dl_cpu.yml ]; then
  echo "Error: dl_cpu.yml not found in the current directory."
  echo "Run this script from the folder that contains dl_cpu.yml."
  exit 1
fi

conda env create -f dl_cpu.yml
