#!/bin/bash
#SBATCH --job-name=octmnist_svc
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=src/models/octmnist_svc_%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1

module load 2024
module load Python/3.12.3-GCCcore-13.3.0

source "$HOME/venvs/octmnist/bin/activate"
cd "$HOME/MAM09A2"

python src/models/fit_linearSVC.py

