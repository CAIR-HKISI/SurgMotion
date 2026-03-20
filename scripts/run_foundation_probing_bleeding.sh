#!/bin/bash
# run_foundation_probing_bleeding.sh - 批量运行 Bleeding probing 任务（多 YAML，按规划目录读取）
# 用法：在下方 TASKS / FNAMES 中配置任务，然后执行本脚本。日志写入 logs/foundation/bleeding/

# >>> conda initialize >>>
CONDA_PATH="${CONDA_PATH:-$HOME/miniconda3}"
if [ -x "${CONDA_PATH}/bin/conda" ]; then
    __conda_setup="$("${CONDA_PATH}/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        [ -f "${CONDA_PATH}/etc/profile.d/conda.sh" ] && . "${CONDA_PATH}/etc/profile.d/conda.sh"
    fi
fi
unset __conda_setup 2>/dev/null
# <<< conda initialize <<<

[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
conda deactivate 2>/dev/null
conda activate NSJepa
#conda activate endomamba

# 项目根目录（脚本所在目录的上一级）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 可选：WandB
export WANDB_MODE="${WANDB_MODE:-online}"
[ -n "$WANDB_API_KEY" ] && export WANDB_API_KEY

# GPU 池：每个任务独占一张卡，多任务并行，卡用完则排队
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}

# ==========================================
# 任务列表：TASKS = 配置子目录，FNAMES = YAML 文件名
# 完整路径 = configs/${TASKS[i]}/${FNAMES[i]}
# ==========================================
TASKS=(
    #"endomamba/Bleeding"
    "gastronet/Bleeding"
    "dinov3/Bleeding"
    "endofm/Bleeding"
    "endovit/Bleeding"
    "endossl/Bleeding"
    "gsvit/Bleeding"
    "surgvlp/Bleeding"
    "videomaev2/Bleeding"
    "surgenet/Bleeding"
    "selfsupsurg/Bleeding"
)

FNAMES=(
    #"endomamba_small_clip_bleeding.yaml"
    "gastronet_vits_clip_bleeding.yaml"
    "dinov3_vitl_clip_bleeding.yaml"
    "endofm_vitb_clip_bleeding.yaml"
    "endovit_vitl_clip_bleeding.yaml"
    "endossl_vitl_laparo_clip_bleeding.yaml"
    "gsvit_vit_clip_bleeding.yaml"
    "surgvlp_res50_clip_bleeding.yaml"
    "videomaev2_large_clip_bleeding.yaml"
    "surgenetxl_caformer_clip_bleeding.yaml"
    "selfsupsurg_res50_clip_bleeding.yaml"
)

LOG_DIR="logs/foundation/bleeding_V2"
mkdir -p "$LOG_DIR"

if [ ${#TASKS[@]} -ne ${#FNAMES[@]} ]; then
    echo "Error: TASKS and FNAMES must have the same length."
    exit 1
fi

TOTAL_TASKS=${#TASKS[@]}
TIME=$(date +"%Y%m%d_%H%M%S")

echo "========================================"
echo "   Foundation Probing - Bleeding (Batch)"
echo "========================================"
echo "Time: $(date)"
echo "Project: $PROJECT_ROOT"
echo "Total tasks: $TOTAL_TASKS"
echo "GPU pool: [${GPU_LIST[*]}] (${NUM_GPUS} GPUs)"
echo "Log dir: $LOG_DIR"
echo ""

# --------------- GPU 池调度 ---------------
declare -A PID_TO_GPU     # pid -> gpu_id
declare -A PID_TO_TASK    # pid -> task description
AVAILABLE_GPUS=("${GPU_LIST[@]}")

acquire_gpu() {
    while [ ${#AVAILABLE_GPUS[@]} -eq 0 ]; do
        wait -n -p DONE_PID ${!PID_TO_GPU[@]} 2>/dev/null
        EXIT_CODE=$?
        if [ -n "$DONE_PID" ]; then
            FREED_GPU="${PID_TO_GPU[$DONE_PID]}"
            DONE_TASK="${PID_TO_TASK[$DONE_PID]}"
            AVAILABLE_GPUS+=("$FREED_GPU")
            if [ $EXIT_CODE -eq 0 ]; then
                echo "[GPU $FREED_GPU] Finished OK:  $DONE_TASK"
            else
                echo "[GPU $FREED_GPU] FAILED (exit=$EXIT_CODE): $DONE_TASK"
            fi
            unset PID_TO_GPU[$DONE_PID]
            unset PID_TO_TASK[$DONE_PID]
        fi
    done
    ACQUIRED_GPU="${AVAILABLE_GPUS[0]}"
    AVAILABLE_GPUS=("${AVAILABLE_GPUS[@]:1}")
}

for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    FNAME="${FNAMES[$i]}"
    CONFIG_PATH="configs/foundation_model_probing/${TASK}/${FNAME}"
    LOG_FILE="${LOG_DIR}/${FNAME%.yaml}_${TIME}.log"

    if [ ! -f "$CONFIG_PATH" ]; then
        echo "[$((i+1))/$TOTAL_TASKS] Skip (config not found): $CONFIG_PATH"
        continue
    fi

    acquire_gpu

    echo "[$((i+1))/$TOTAL_TASKS] Launching on GPU $ACQUIRED_GPU: $CONFIG_PATH"
    echo "  Log: $LOG_FILE"

    python -m evals.main \
        --fname "$CONFIG_PATH" \
        --devices "cuda:$ACQUIRED_GPU" \
        > "$LOG_FILE" 2>&1 &

    PID_TO_GPU[$!]="$ACQUIRED_GPU"
    PID_TO_TASK[$!]="$CONFIG_PATH"
done

# 等待所有剩余任务完成
for pid in "${!PID_TO_GPU[@]}"; do
    wait "$pid"
    EXIT_CODE=$?
    GPU="${PID_TO_GPU[$pid]}"
    TASK_DESC="${PID_TO_TASK[$pid]}"
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[GPU $GPU] Finished OK:  $TASK_DESC"
    else
        echo "[GPU $GPU] FAILED (exit=$EXIT_CODE): $TASK_DESC"
    fi
done

echo ""
echo "========================================"
echo "All $TOTAL_TASKS tasks finished."
echo "========================================"
