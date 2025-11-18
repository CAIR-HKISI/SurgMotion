#!/bin/bash
# 主提交脚本：为每个数据集提交一个独立的 SLURM 任务

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SINGLE_SCRIPT="${SCRIPT_DIR}/srun_probing_single.sh"

# 确保单个任务脚本存在
if [ ! -f "${SINGLE_SCRIPT}" ]; then
    echo "Error: ${SINGLE_SCRIPT} not found!"
    exit 1
fi

# 确保脚本有执行权限
chmod +x "${SINGLE_SCRIPT}"

# 确保日志目录存在
mkdir -p log

# ========================
# 定义所有数据集配置
# ========================

# 数据集配置数组: (FNAME, DATA_NAME, FOLDER_NAME)
declare -a datasets=(
    "autolaparo_vitl_cpt_attentive_64f.yaml|autolaparo|autolaparo"
    "cataracts_vitl_cpt_attentive_64f.yaml|cataracts|cataracts"
    "cholec80_vitl_cpt_attentive_64f.yaml|cholec80|cholec80"
    "egosurgery_vitl_cpt_attentive_64f.yaml|egosurgery|egosurgery"
    "grasp_vitl_cpt_attentive_64f.yaml|grasp|grasp"
    "jigsaws_vitl_cpt_attentive_64f.yaml|jigsaws|jigsaws"
    "m2cai_vitl_cpt_attentive_64f.yaml|m2cai|m2cai"
    "ophnet_vitl_cpt_attentive_64f.yaml|ophnet|ophnet"
    "pitvis_vitl_cpt_attentive_64f.yaml|pitvis|pitvis"
    "pmlr50_vitl_cpt_attentive_64f.yaml|pmlr50|pmlr50"
    "polypdiag_vitl_cpt_attentive_64f.yaml|polypdiag|polypdiag"
    "surgical_actions160_vitl_cpt_attentive_64f.yaml|surgical_actions160|surgical_actions160"
)

# ========================
# 为每个数据集提交独立任务
# ========================

echo "=========================================="
echo "Submitting probing tasks for all datasets"
echo "Total datasets: ${#datasets[@]}"
echo "=========================================="
echo ""

submitted_jobs=()

for dataset_config in "${datasets[@]}"; do
    # 解析配置
    IFS='|' read -r FNAME DATA_NAME FOLDER_NAME <<< "$dataset_config"
    
    echo "Submitting job for ${DATA_NAME}..."
    echo "  Config: ${FNAME}"
    echo "  Folder: ${FOLDER_NAME}"
    
    # 使用 sbatch 提交任务，传递参数
    job_output=$(sbatch \
        --job-name="prb_vitl80_${DATA_NAME}" \
        --output="log/prb_vitl80_${DATA_NAME}_%j.out" \
        --error="log/prb_vitl80_${DATA_NAME}_%j.err" \
        --time=48:00:00 \
        --partition=AISS2025073101 \
        --nodelist=klb-dgx-011,klb-dgx-120 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=16 \
        --gres=gpu:1 \
        --mem=256G \
        "${SINGLE_SCRIPT}" "${FNAME}" "${DATA_NAME}" "${FOLDER_NAME}" \
        2>&1)
    
    # 提取 job ID (格式: "Submitted batch job 12345")
    job_id=$(echo "${job_output}" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+' || echo "")
    
    if [ -n "${job_id}" ]; then
        submitted_jobs+=("${DATA_NAME}:${job_id}")
        echo "  ✓ Job submitted successfully: Job ID = ${job_id}"
    else
        echo "  ✗ Failed to submit job for ${DATA_NAME}"
    fi
    echo ""
done

# ========================
# 显示提交摘要
# ========================

echo "=========================================="
echo "Submission Summary"
echo "=========================================="
echo "Total datasets: ${#datasets[@]}"
echo "Successfully submitted: ${#submitted_jobs[@]}"
echo ""
echo "Submitted jobs:"
for job_info in "${submitted_jobs[@]}"; do
    IFS=':' read -r dataset_name job_id <<< "$job_info"
    echo "  ${dataset_name}: Job ID ${job_id}"
done
echo ""
echo "Use 'squeue -u $USER' to check job status"
echo "=========================================="

