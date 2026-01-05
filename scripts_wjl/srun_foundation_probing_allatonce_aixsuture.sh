#!/bin/bash
# srun_foundation_probing_allatonce_aixsuture.sh - 批量提交所有AIxsuture probing任务（不等待完成）

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
# 定义任务列表
# ==========================================

TASKS=(
   
    # dinov3
    "fdtn_probing/dinov3/AIxsuture"
    "fdtn_probing/dinov3/AIxsuture"

    # endofm
    "fdtn_probing/endofm/AIxsuture"

    # endossl
    "fdtn_probing/endossl/AIxsuture"
    "fdtn_probing/endossl/AIxsuture"

    # endovit
    "fdtn_probing/endovit/AIxsuture"

    # gastronet
    "fdtn_probing/gastronet/AIxsuture"

    # gsvit
    "fdtn_probing/gsvit/AIxsuture"

    # selfsupsurg
    "fdtn_probing/selfsupsurg/AIxsuture"

    # surgenet
    "fdtn_probing/surgenet/AIxsuture"

    # surgvlp
    "fdtn_probing/surgvlp/AIxsuture"

    # videomaev2
    "fdtn_probing/videomaev2/AIxsuture"
    "fdtn_probing/videomaev2/AIxsuture"
)

FNAMES=(

    # dinov3
    "dinov3_vitl_64f_aixsuture.yaml"
    "dinov3_vith_64f_aixsuture.yaml"

    # endofm
    "endofm_vitb_64f_AIxsuture.yaml"

    # endossl
    "endossl_vitl_colono_64f_AIxsuture.yaml"
    "endossl_vitl_laparo_64f_AIxsuture.yaml"

    # endovit
    "endovit_vitl_64f_AIxsuture.yaml"

    # gastronet
    "gastronet_vits_64f_AIxsuture.yaml"

    # gsvit
    "gsvit_vit_64f_AIxsuture.yaml"

    # selfsupsurg
    "selfsupsurg_res50_64f_AIxsuture.yaml"

    # surgenet
    "surgenetxl_caformer_64f_aixsuture.yaml"

    # surgvlp
    "surgvlp_res50_64f_aixsuture.yaml"

    # videomaev2
    "videomaev2_giant_64f_aixsuture.yaml"
    "videomaev2_large_64f_aixsuture.yaml"
)



# 验证列表长度一致
if [ ${#TASKS[@]} -ne ${#FNAMES[@]} ]; then
    echo "❌ Error: TASKS and FNAMES arrays must have the same length!"
    exit 1
fi

# 创建日志目录
mkdir -p logs/foundation

echo "========================================"
echo "   Batch Submit AIxsuture Probing Tasks (Async)"
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
    echo "📦 Task [$((i+1))/${#TASKS[@]}]"
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
        --job-name="${FNAME}" \
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
	    export REQUESTS_CA_BUNDLE=\$SSL_CERT_FILE

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


