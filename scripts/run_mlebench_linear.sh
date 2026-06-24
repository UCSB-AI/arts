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

# Required env vars:
#   TASK     : MLGym task id, e.g. mlebenchAPTOS
#   EXECUTOR : gemini-3-pro-preview | gemini-3-flash-preview
set -eu
# --- Portable paths (override via environment) ---
AIR_REPO="${AIR_REPO:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
MLGYM_PATH="${MLGYM_PATH:-$AIR_REPO/../MLGym}"

: "${TASK:?TASK must be set}"
: "${EXECUTOR:?EXECUTOR must be set}"

OUTDIR=$AIR_REPO/outputs/linear_${TASK}_${EXECUTOR//-/}_$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID:-local}

PYTHON=$AIR_REPO/.venv/bin/python3
LINEAR=$AIR_REPO/arts/search/linear.py

set -a
source $AIR_REPO/.env 2>/dev/null || true
set +a

# Alternate Gemini keys for preview models between concurrent jobs.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  if [ $(( SLURM_JOB_ID % 2 )) -eq 0 ]; then
    export GEMINI_API_KEY_OVERRIDE="$GEMINI_API_KEY"
  else
    export GEMINI_API_KEY_OVERRIDE="${GEMINI_API_KEY_5:-$GEMINI_API_KEY}"
  fi
fi

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export MLGYM_APPTAINER_IMAGE=${MLGYM_APPTAINER_IMAGE:-aigym/mlgym-agent:latest}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MLGYM_AGENT_LONG_TIMEOUT=1500
export MLGYM_AGENT_ACTION_NO_OUTPUT_TIMEOUT=900

module load apptainer 2>/dev/null || true
source $MLGYM_PATH/apptainer/activate.sh 2>/dev/null || true

mkdir -p "$OUTDIR"
echo "[linear-$TASK] Node: $(hostname), Job: ${SLURM_JOB_ID:-local}, Executor: $EXECUTOR, OutDir: $OUTDIR"

cd $MLGYM_PATH

$PYTHON $LINEAR \
    --task-config tasks/${TASK}.yaml \
    --max-actions 500 \
    --model "$EXECUTOR" \
    --vllm-url "" \
    --env-gpu 0 \
    --output-dir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/run.log"

echo "[linear-$TASK] Done (exit=$?)"
