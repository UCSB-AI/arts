#!/bin/bash
# #SBATCH --account=YOUR_ACCOUNT       # set for your SLURM cluster
# #SBATCH --partition=YOUR_PARTITION
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm-%j.out

# Required env vars (pass via sbatch --export=ALL,TASK=...,EXECUTOR=...):
#   TASK     : MLGym task id, e.g. mlebenchAPTOS
#   EXECUTOR : gemini-3-pro-preview | gemini-3-flash-preview
set -eu
# --- Portable paths (override via environment) ---
AIR_REPO="${AIR_REPO:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
MLGYM_PATH="${MLGYM_PATH:-$AIR_REPO/../MLGym}"

: "${TASK:?TASK must be set}"
: "${EXECUTOR:?EXECUTOR must be set}"

OUTDIR=$AIR_REPO/outputs/llmg_${TASK}_o3_${EXECUTOR//-/}_$(date +%Y%m%d_%H%M%S)

PYTHON=$AIR_REPO/.venv/bin/python3
LLMG=$AIR_REPO/arts/search/arts.py

set -a
source $AIR_REPO/.env 2>/dev/null || true
set +a
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export MLGYM_APPTAINER_IMAGE=${MLGYM_APPTAINER_IMAGE:-aigym/mlgym-agent:latest}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MLGYM_AGENT_LONG_TIMEOUT=1500
export MLGYM_AGENT_ACTION_NO_OUTPUT_TIMEOUT=900

module load apptainer 2>/dev/null || true
source $MLGYM_PATH/apptainer/activate.sh 2>/dev/null || true

mkdir -p "$OUTDIR"
echo "[llmg-$TASK] Node: $(hostname), Job: ${SLURM_JOB_ID:-local}, Executor: $EXECUTOR, OutDir: $OUTDIR"

cd $MLGYM_PATH

$PYTHON $LLMG \
    --task-config tasks/${TASK}.yaml \
    --node-budget 100 \
    --time-budget 27600 \
    --max-actions 200 \
    --initial-breadth 1 \
    --scientist-model o3 \
    --scientist-url "" \
    --executor-model "$EXECUTOR" \
    --executor-url "" \
    --env-gpu 0 \
    --output-dir "$OUTDIR" \
    --research-phase-steps 3 \
    2>&1 | tee "$OUTDIR/run.log"

echo "[llmg-$TASK] Done (exit=$?)"
