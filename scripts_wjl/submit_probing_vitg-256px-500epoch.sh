#!/bin/bash

# ==========================================
# 全局配置 (在这里修改模型和路径)
# ==========================================
# 获取当前脚本所在目录，确保 sbatch 能找到 run_probing.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 日志根目录
LOG_ROOT="logs"

# Checkpoint 文件夹名称
CKPTL_NAME="cpt_vitg-256px-16f_100epoch"

# 模型架构名称
MODEL_NAME="vit_giant_xformers"

# 指定要加载的 checkpoint（例如 e500.pt；留空则 run_probing 默认 latest）
CKPT_EPOCH="e500.pt"

# ==========================================
# 任务配置列表
# ==========================================
CONFIGS=(
    "autolaparo_probe_attentive_64f.yaml"
    "egosurgery_probe_attentive_64f.yaml"
    "atlas_probe_attentive_64f.yaml"
    "pitvis_probe_attentive_64f.yaml"
    "avos_probe_attentive_64f.yaml"
    "polypdiag_probe_attentive_64f.yaml"
    "jigsaws_probe_attentive_64f.yaml"
    "aIxsuture-5s_probe_attentive_64f.yaml"
    "m2cai_probe_attentive_64f.yaml"
)


# ==========================================
# 循环提交任务
# ==========================================
for FNAME in "${CONFIGS[@]}"; do
    # 提取数据集名称 (例如 aIxsuture)
    DATA_NAME=$(echo ${FNAME} | cut -d'_' -f1)
    
    # 构造作业名称
    JOB_NAME="prb_${DATA_NAME}"

    echo "Submitting task for: ${FNAME}"
    echo "  -> Job Name: ${JOB_NAME}"
    echo "  -> Model: ${CKPTL_NAME}"

    # 使用 sbatch 提交
    # --export: 将变量传递给 run_probing.sh
    # 使用绝对路径调用 run_probing.sh，避免在其他目录运行时报错
    sbatch \
        --job-name="${JOB_NAME}" \
        --export=ALL,FNAME="${FNAME}",LOG_ROOT="${LOG_ROOT}",CKPTL_NAME="${CKPTL_NAME}",MODEL_NAME="${MODEL_NAME}",CKPT_EPOCH="${CKPT_EPOCH}" \
        "${SCRIPT_DIR}/run_probing.sh"
done

echo "All jobs submitted."
