#!/bin/bash
# srun_foundation_probing_allatonce.sh - 批量提交所有probing任务（不等待完成）

# >>> conda initialize >>>

conda_path="/lustre/projects/med-multi-llm/jinlin_wu/miniconda3"
__conda_setup="$('${conda_path}/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "${conda_path}/etc/profile.d/conda.sh" ]; then
        . "${conda_path}/etc/profile.d/conda.sh"
    else
        export PATH="${conda_path}/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

source ~/.bashrc
conda deactivate
conda activate jepa_torch
cd /home/projects/med-multi-llm/jinlin_wu/NSJepa_20251112

export https_proxy="http://cair:coy_suffocate_petrified@klb-fwproxy-01.aisc.local:3128"

# 配置参数
export PYTHONHTTPSVERIFY=0
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""

# 2. 设置代理 (建议 http 和 https 都设置，以防万一)
export http_proxy="http://cair:coy_suffocate_petrified@klb-fwproxy-01.aisc.local:3128"
export https_proxy="http://cair:coy_suffocate_petrified@klb-fwproxy-01.aisc.local:3128"
export HTTP_PROXY="http://cair:coy_suffocate_petrified@klb-fwproxy-01.aisc.local:3128"
export HTTPS_PROXY="http://cair:coy_suffocate_petrified@klb-fwproxy-01.aisc.local:3128"


# 2. [新增] 忽略 SSL 证书验证错误 (解决 x509 报错的关键)
export WANDB_INSECURE_DISABLE_SSL=true

# 3. 强制 WandB 为在线模式 (确保上传)
export WANDB_MODE=online

# ==========================================
# 定义任务列表（在这里添加你的任务对）
# ==========================================
# TASKS=(
#     "fdtn_probing/gastronet/AVOS"
#     "fdtn_probing/gastronet/polypdiag"
#     "fdtn_probing/gastronet/surgical-action-160"
# )

# FNAMES=(
#     "gastronet_vits_64f_avos.yaml"
#     "gastronet_vits_64f_PolypDiag.yaml"
#     "gastronet_vits_64f_Surgical-Action-160.yaml"
# )


# TASKS=(
#     "fdtn_probing/videomaev2/AVOS"
#     "fdtn_probing/videomaev2/polypdiag"
#     "fdtn_probing/videomaev2/surgical-action-160"
# )

# FNAMES=(
#     "videomaev2_large_64f_avos.yaml"
#     "videomaev2_large_64f_PolypDiag.yaml"
#     "videomaev2_large_64f_Surgical-Action-160.yaml"
# )


# TASKS=(
#     "fdtn_probing/selfsupsurg/AVOS"
#     "fdtn_probing/selfsupsurg/polypdiag"
#     "fdtn_probing/selfsupsurg/surgical-action-160"
# )

# FNAMES=(
#     "selfsupsurg_res50_64f_avos.yaml"
#     "selfsupsurg_res50_64f_PolypDiag.yaml"
#     "selfsupsurg_res50_64f_Surgical-Action-160.yaml"
# )

# TASKS=(
#     "fdtn_probing/surgenet/AVOS"
#     "fdtn_probing/surgenet/polypdiag"
#     "fdtn_probing/surgenet/surgical-action-160"
# )

# FNAMES=(
#     "surgenetxl_caformer_64f_avos.yaml"
#     "surgenetxl_caformer_64f_PolypDiag.yaml"
#     "surgenetxl_caformer_64f_Surgical-Action-160.yaml"
# )

TASKS=(
    "fdtn_probing/dinov3/cholec80"
    "fdtn_probing/dinov3/AutoLaparo"
    "fdtn_probing/dinov3/M2CAI16"
    "fdtn_probing/dinov3/EgoSurgery"
    "fdtn_probing/dinov3/PitVis"
    "fdtn_probing/dinov3/Atlas"
    "fdtn_probing/dinov3/OphNet"
    "fdtn_probing/dinov3/PmLR50"
)

FNAMES=(
    "dinov3_vith_64f_probing_cholec80.yaml"
    "dinov3_vith_64f_AutoLaparo.yaml"
    "dinov3_vith_64f_M2CAI16.yaml"
    "dinov3_vith_64f_EgoSurgery.yaml"
    "dinov3_vith_64f_PitVis.yaml"
    "dinov3_vith_64f_Atlas.yaml"
    "dinov3_vith_64f_OphNet.yaml"
    "dinov3_vith_64f_PmLR50.yaml"
)

# TASKS=(
#     "fdtn_probing/internvideo_next/AVOS"
#     "fdtn_probing/internvideo_next/polypdiag"
#     "fdtn_probing/internvideo_next/surgical-action-160"
# )

# FNAMES=(
#     "internvideo_next_64f_avos.yaml"
#     "internvideo_next_64f_PolypDiag.yaml"
#     "internvideo_next_64f_Surgical-Action-160.yaml"
# )



# 验证列表长度一致
if [ ${#TASKS[@]} -ne ${#FNAMES[@]} ]; then
    echo "❌ Error: TASKS and FNAMES arrays must have the same length!"
    exit 1
fi

# 创建日志目录
mkdir -p logs/foundation

echo "========================================"
echo "   Batch Submit Probing Tasks (Async)"
echo "========================================"
echo "Time: $(date)"
echo "Total tasks: ${#TASKS[@]}"
echo ""

# ==========================================
# 循环提交所有任务（不等待完成）
# ==========================================
SUBMITTED_JOBS=()

for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    FNAME="${FNAMES[$i]}"
    TIME=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="logs/foundation/${FNAME}_${TIME}.log"
    
    echo ""
    echo "========================================"
    echo "📦 Task [$((i+1))/$TOTAL_TASKS]"
    echo "========================================"
    echo "Task: ${TASK}"
    echo "Config: ${FNAME}"
    echo "Log: ${LOG_FILE}"
    echo ""
    echo "📋 Checking GPU availability..."
    sinfo -p gpu -o "%20P %5a %10l %5D %5c %10G" || echo "Note: GPU partition info not available"
    echo ""
    echo "⏳ Requesting GPU resources..."
    echo "========================================"
    
    # 使用sbatch直接提交（异步，不等待）
    JOB_ID=$(sbatch \
        --job-name="${FNAMES[$i]}" \
	--nodes=1 \
	--partition=AISS2025073101 \
	--gres=gpu:1 \
	--mem=256G \
        --cpus-per-task=16 \
        --output="${LOG_FILE}" \
        --error="${LOG_FILE}" \
        --wrap="
            source ~/.bashrc
            conda activate jepa_torch
            cd /home/projects/med-multi-llm/jinlin_wu/NSJepa_20251112
	    export https_proxy="http://cair:coy_suffocate_petrified@klb-fwproxy-01.aisc.local:3128"
        export PYTHONHTTPSVERIFY=0
	    export CURL_CA_BUNDLE=""
        export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
	    export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE

	    echo '✅ GPU resources allocated!'
            echo 'Job ID:' \$SLURM_JOB_ID
            echo 'Node: '\$SLURMD_NODENAME
            echo 'GPUs: '\$CUDA_VISIBLE_DEVICES
            echo ''
            nvidia-smi --query-gpu=index,name,memory.total --format=csv
            echo ''
            echo '🏃 Starting probing test...'
            echo 'Task: ${TASK}'
            echo 'Config: ${FNAME}'
            echo '========================================'
            python -m evals.main --fname configs/${TASK}/${FNAME} --devices cuda:0
        " | awk '{print $4}')
    
    if [ -n "$JOB_ID" ]; then
        SUBMITTED_JOBS+=("$JOB_ID")
        echo "  ✅ Submitted! Job ID: $JOB_ID"
    else
        echo "  ❌ Failed to submit!"
    fi
done

# ==========================================
# 输出摘要
# ==========================================
echo ""
echo "========================================"
echo "Submission Summary"
echo "========================================"
echo "Total tasks: ${#TASKS[@]}"
echo "Submitted: ${#SUBMITTED_JOBS[@]}"
echo ""
echo "Job IDs: ${SUBMITTED_JOBS[*]}"
echo ""
echo "Check status: squeue -u \$USER"
echo "Cancel all: scancel ${SUBMITTED_JOBS[*]}"
echo "========================================"
