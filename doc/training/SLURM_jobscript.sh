#!/usr/bin/env bash
#SBATCH -A C3SE2024-1-13             # Replace with an active site allocation.
#SBATCH -t 4-12:00:00                # Site-specific wall-time limit.
#SBATCH -n 32                        # Site-specific CPU allocation.
#SBATCH --gpus-per-node=T4:1         # Site-specific GPU allocation.
#SBATCH -J v0-run19                  # Job name.

set -euo pipefail

# Export these absolute, persistent paths when submitting the job.
REPO_PATH="${REPO_PATH:?Set REPO_PATH to the repository checkout}"
CONTAINER="${CONTAINER:?Set CONTAINER to the Apptainer image}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to persistent model storage}"
MODEL="${MODEL:-0}"
RUN="${RUN:-19}"
PYTHON="${PYTHON:-python}"

RUN_PATH="${OUTPUT_ROOT}/variant-${MODEL}/run${RUN}"
mkdir -p "$RUN_PATH"
cd "$REPO_PATH"

apptainer exec \
    --bind "$REPO_PATH:$REPO_PATH" \
    --bind "$OUTPUT_ROOT:$OUTPUT_ROOT" \
    "$CONTAINER" "$PYTHON" scripts/train.py \
    --mode cluster --index "$MODEL" --run-vers "$RUN" \
    --path "$RUN_PATH" --no-evaluation
