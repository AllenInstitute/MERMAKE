#!/usr/bin/env bash
#SBATCH --job-name=mermake
#SBATCH --partition=gpu            # EDIT: your GPU partition
#SBATCH --gres=gpu:1               # EDIT: GPUs per job (e.g. gpu:a100:1)
#SBATCH --cpus-per-task=8          # EDIT
#SBATCH --mem=64G                  # EDIT
#SBATCH --time=12:00:00            # EDIT
#SBATCH --output=%x-%j.out         # slurm log in submit directory
#SBATCH --error=%x-%j.err

# Usage:  sbatch run_slurm.sbatch
#   or override at submit time:
#   sbatch --export=ALL,SETTINGS=/path/to/settings.toml run_slurm.sbatch

set -euxo pipefail

# ---- EDIT THESE PATHS ------------------------------------------------------
REPO_DIR="${REPO_DIR:-$HOME/code/MERMAKE}"            # local clone of MERMAKE
BRANCH="${BRANCH:-data/sum-channels}"                 # branch to run
SETTINGS="${SETTINGS:-$HOME/data/test_settings.toml}" # settings TOML (its paths must be valid here)
RESULTS_DIR="${RESULTS_DIR:-$HOME/results}"           # the output dir named in the TOML, used for the report
UV_EXTRA="${UV_EXTRA:-cuda12}"                        # extra passed to `uv sync`
# Optional: load site modules / uv if not on PATH
# module load cuda/12.2 uv
# ----------------------------------------------------------------------------

cd "$REPO_DIR"
git fetch
git switch "$BRANCH"

# Keep uv's cache and venv off a possibly small $HOME if desired:
# export UV_CACHE_DIR=/scratch/$USER/uv-cache

uv sync --extra "$UV_EXTRA"
nvidia-smi -L || true

uv run mermake "$SETTINGS"
uv run mermake-report "$RESULTS_DIR"
