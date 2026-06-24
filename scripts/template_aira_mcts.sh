#!/bin/bash
# Template for aira_dojo MCTS experiments.
# Pass parameters via --export=ALL,KEY=value:
#   TASK        - yaml name without extension, e.g. titanic
#   NODE_BUDGET - 5 or 15
#   RUN_ID      - r1, r2, ...
#   MODEL       - "qwen" (Qwen3-4B via vLLM) or "o3" (API, no vLLM)
#   OUTBASE     - base output directory (default: Feb28_Experiments)
#
# "qwen" model: needs 1 GPU (for vLLM). "o3" model: no GPU needed (CPU only).
# Override SLURM directives at submission time:
#   sbatch --time=3:00:00 template_aira_mcts.sh

#SBATCH --job-name=aira_run
# #SBATCH --account=YOUR_ACCOUNT       # set for your SLURM cluster
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out
#SBATCH --error=/tmp/aira_%x_%j.log

set -e
# --- Portable paths (override via environment) ---
AIR_REPO="${AIR_REPO:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
MLGYM_PATH="${MLGYM_PATH:-$AIR_REPO/../MLGym}"

# --- Defaults ---
TASK=${TASK:-titanic}
NODE_BUDGET=${NODE_BUDGET:-5}
RUN_ID=${RUN_ID:-r1}
MODEL=${MODEL:-o3}
OUTBASE=${OUTBASE:-$AIR_REPO/outputs/Feb28_Experiments}

PYTHON=$AIR_REPO/.venv/bin/python3
AIRA=$AIR_REPO/arts/search/aira/search.py

echo "[batch] Node: $(hostname), Job: $SLURM_JOB_ID"
echo "[batch] Task=$TASK, NodeBudget=$NODE_BUDGET, RunID=$RUN_ID, Model=$MODEL"

set -a; source $AIR_REPO/.env; set +a
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

# --- Determine method dir and model flags ---
if [ "$MODEL" = "qwen" ]; then
    METHOD_DIR="aira_vanilla"
    MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
    VLLM_URL="http://localhost:8000/v1"
    EXTRA_FLAGS="--no-reflexion"
    ENV_GPU="1"

    # Start vLLM on GPU 0
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 2
    CUDA_VISIBLE_DEVICES=0 $PYTHON \
        -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_NAME" --host 0.0.0.0 --port 8000 \
        --max-model-len 32768 --max-num-seqs 4 \
        --gpu-memory-utilization 0.90 --trust-remote-code \
        > /tmp/vllm_aira_${TASK}_${NODE_BUDGET}_${RUN_ID}.log 2>&1 &
    VLLM_PID=$!
    for i in $(seq 1 30); do
        sleep 10
        curl -sf http://localhost:8000/v1/models > /dev/null 2>&1 && echo "[batch] vLLM ready" && break
        kill -0 $VLLM_PID 2>/dev/null || { echo "[batch] vLLM died"; tail -20 /tmp/vllm_aira_${TASK}_${NODE_BUDGET}_${RUN_ID}.log; exit 1; }
    done
else
    # API model (o3 etc) — no vLLM
    METHOD_DIR="aira_o3"
    MODEL_NAME="$MODEL"
    VLLM_URL=""
    EXTRA_FLAGS="--thinking-budget 2048 --no-reflexion"
    ENV_GPU="0"
fi

OUTDIR="$OUTBASE/${METHOD_DIR}/${TASK}/aira_mcts_n${NODE_BUDGET}_${RUN_ID}"

source $MLGYM_PATH/apptainer/activate.sh
cd $MLGYM_PATH
mkdir -p "$OUTDIR"

$PYTHON $AIRA \
    --task-config "tasks/${TASK}.yaml" \
    --node-budget "$NODE_BUDGET" --max-actions 20 \
    --search-policy mcts --uct-c 0.25 --num-children 5 \
    --model "$MODEL_NAME" --vllm-url "$VLLM_URL" \
    --temperature 0.9 \
    --env-gpu "$ENV_GPU" \
    --output-dir "$OUTDIR" \
    $EXTRA_FLAGS

if [ -f "$OUTDIR/result.json" ]; then
    $PYTHON -c "import json,sys; d=json.load(open(sys.argv[1])); print('score='+str(round(d['best_score'],4))+' improvement='+str(round(d['improvement'],4)))" "$OUTDIR/result.json"
else
    echo "NO RESULT"
fi
