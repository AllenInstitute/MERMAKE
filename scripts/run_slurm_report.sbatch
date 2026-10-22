#!/usr/bin/env bash
#SBATCH --job-name=mermake-report
#SBATCH --partition=cpu            # EDIT: a partition without GPUs is enough
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G                  # EDIT
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# The plots of mermake-report need all the results, so run it once after the processing jobs:
#   sbatch --dependency=afterok:<array job id> run_slurm_report.sbatch
#   sbatch --dependency=afterok:<array job id> --export=ALL,RESULTS_DIR=/path/to/output run_slurm_report.sbatch
# The environment needs the report extra: uv sync --extra cuda12 --extra report

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/code/MERMAKE}"   # clone of MERMAKE, with its environment set up
RESULTS_DIR="${RESULTS_DIR:-$HOME/results}"  # the output folder named in the settings TOML

cd "$REPO_DIR"
exec uv run --no-sync mermake-report "$RESULTS_DIR"
