#!/bin/bash
# Template for llm_guided_tree_search experiments.
# Pass parameters via --export=ALL,KEY=value:
#   TASK        - yaml name without extension, e.g. titanic
#   NODE_BUDGET - 5 or 15
#   RUN_ID      - r1, r2, ...
#   SCIENTIST   - o3, o3o3 (o3 for both models), qwen (Qwen3-4B)
#   OUTBASE     - base output directory (default: Feb28_Experiments)
#
# Override SLURM directives at submission time, e.g.:
#   sbatch --time=3:00:00 --gpus-per-node=... template_llm_guided.sh

#SBATCH --job-name=llmg_run
# #SBATCH --account=YOUR_ACCOUNT       # set for your SLURM cluster
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --output=slurm-%j.out
#SBATCH --error=/tmp/llmg_%x_%j.log

set -e
# --- Portable paths (override via environment) ---
AIR_REPO="${AIR_REPO:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
MLGYM_PATH="${MLGYM_PATH:-$AIR_REPO/../MLGym}"

# --- Defaults ---
TASK=${TASK:-titanic}
NODE_BUDGET=${NODE_BUDGET:-5}
RUN_ID=${RUN_ID:-r1}
SCIENTIST=${SCIENTIST:-o3}
OUTBASE=${OUTBASE:-$AIR_REPO/outputs/Feb28_Experiments}

PYTHON=$AIR_REPO/.venv/bin/python3
LLMG=$AIR_REPO/arts/search/arts.py

# Map SCIENTIST tag to actual model names
case "$SCIENTIST" in
    o3)     SCIENTIST_MODEL="o3";     EXECUTOR_MODEL="Qwen/Qwen3-4B-Instruct-2507" ;;
    o3o3)   SCIENTIST_MODEL="o3";     EXECUTOR_MODEL="o3" ;;
    qwen)   SCIENTIST_MODEL="Qwen/Qwen3-4B-Instruct-2507"; EXECUTOR_MODEL="Qwen/Qwen3-4B-Instruct-2507" ;;
    *)      SCIENTIST_MODEL="$SCIENTIST"; EXECUTOR_MODEL="Qwen/Qwen3-4B-Instruct-2507" ;;
esac

OUTDIR="$OUTBASE/llm_guided/${SCIENTIST}_scientist/${TASK}/llm_guided_n${NODE_BUDGET}_${RUN_ID}"

echo "[batch] Node: $(hostname), Job: $SLURM_JOB_ID"
echo "[batch] Task=$TASK, NodeBudget=$NODE_BUDGET, RunID=$RUN_ID, Scientist=$SCIENTIST_MODEL"

set -a; source $AIR_REPO/.env; set +a
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

# --- Start vLLM on GPU 0 (needed for executor) ---
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 2

CUDA_VISIBLE_DEVICES=0 $PYTHON \
    -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 32768 --max-num-seqs 4 \
    --gpu-memory-utilization 0.90 --trust-remote-code \
    > /tmp/vllm_llmg_${TASK}_${NODE_BUDGET}_${RUN_ID}.log 2>&1 &
VLLM_PID=$!

for i in $(seq 1 30); do
    sleep 10
    if curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "[batch] vLLM ready after ${i}x10s"; break
    fi
    kill -0 $VLLM_PID 2>/dev/null || { echo "[batch] vLLM died"; tail -20 /tmp/vllm_llmg_${TASK}_${NODE_BUDGET}_${RUN_ID}.log; exit 1; }
done

source $MLGYM_PATH/apptainer/activate.sh
cd $MLGYM_PATH
mkdir -p "$OUTDIR"

# For Qwen scientist, point it at local vLLM. For API models (o3), leave empty.
SCIENTIST_URL=""
case "$SCIENTIST" in
    qwen) SCIENTIST_URL="http://localhost:8000/v1" ;;
esac

$PYTHON $LLMG \
    --task-config "tasks/${TASK}.yaml" \
    --node-budget "$NODE_BUDGET" --max-actions 20 \
    --scientist-model "$SCIENTIST_MODEL" \
    --scientist-url "$SCIENTIST_URL" \
    --executor-model "$EXECUTOR_MODEL" \
    --executor-url http://localhost:8000/v1 \
    --temperature 0.9 \
    --env-gpu 1 \
    --output-dir "$OUTDIR"

if [ -f "$OUTDIR/result.json" ]; then
    $PYTHON -c "import json,sys; d=json.load(open(sys.argv[1])); print('score='+str(round(d['best_score'],4))+' improvement='+str(round(d['improvement'],4)))" "$OUTDIR/result.json"
else
    echo "NO RESULT"
fi
