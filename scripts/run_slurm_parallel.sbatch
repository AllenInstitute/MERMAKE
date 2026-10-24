#!/usr/bin/env bash
#SBATCH --job-name=mermake
#SBATCH --partition=gpu            # EDIT: your GPU partition
#SBATCH --gres=gpu:1               # EDIT: GPUs per array task. One is the simplest, see below
#SBATCH --cpus-per-task=8          # EDIT: for 4 GPUs use about 4 times this
#SBATCH --mem=32G                  # EDIT: about 20 GB per GPU for 4 x 40 x 2800 x 2800 images, see docs/scaling.md
#SBATCH --time=12:00:00            # EDIT
#SBATCH --array=0-3                # EDIT: how many array tasks (shards) to split the fields of view into
#SBATCH --output=%x-%A_%a.out      # slurm log per task, in the submit directory
#SBATCH --error=%x-%A_%a.err

# One job array runs mermake on N nodes or GPUs at once. Every array task processes its own share
# of the fields of view (--shard), with every round of a field of view in the same task, so the
# tasks never touch the same result. With --gres=gpu:G each task starts G workers, one per GPU.
#
# Prepare the environment ONCE, before submitting (the tasks do not touch it, so they cannot race):
#   cd $REPO_DIR && uv sync --extra cuda12 --extra report
#
# Usage:
#   sbatch run_slurm_parallel.sbatch                          # array and GPUs as set above
#   sbatch --array=0-15%4 run_slurm_parallel.sbatch           # 16 shards, at most 4 running at once
#   sbatch --gres=gpu:4 --array=0-1 ...                       # 2 nodes with 4 GPUs: 8 shards in all
#   sbatch --export=ALL,SETTINGS=/path/to/settings.toml run_slurm_parallel.sbatch
# When it has finished, make the report once (see run_slurm_report.sbatch):
#   sbatch --dependency=afterok:<array job id> run_slurm_report.sbatch
# Run again with the same command after a failure or the time limit: finished results are kept.
# The input and the output folder of the settings must be on a filesystem every node can see.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/code/MERMAKE}"            # clone of MERMAKE, with its environment set up
SETTINGS="${SETTINGS:-$HOME/data/test_settings.toml}" # settings TOML (its paths must be valid here)
PREFETCH="${PREFETCH:-5}"                             # images read ahead per worker, 2.3 GiB each here
# Optional: load site modules if uv or the CUDA driver libraries are not on PATH
# module load cuda/12.2 uv

cd "$REPO_DIR"
nvidia-smi -L || true
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

shard_args=()
if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
    # the array index counted from 0, also if the array starts elsewhere or has a step
    shard=$(( (SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN) / ${SLURM_ARRAY_TASK_STEP:-1} ))
    shard_args=(--shard "${shard}/${SLURM_ARRAY_TASK_COUNT}")
fi

# --gpus all starts one worker per GPU of this task (one GPU: one worker)
exec uv run --no-sync mermake --gpus all --prefetch "$PREFETCH" ${shard_args[@]+"${shard_args[@]}"} "$SETTINGS"
