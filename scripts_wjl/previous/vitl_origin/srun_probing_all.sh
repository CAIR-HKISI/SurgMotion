#!/bin/bash
#SBATCH --job-name=prb_vitl_origin_all          # 作业名
#SBATCH --output=log11/%x_%j.out      # 标准输出日志（自动包含作业号）
#SBATCH --error=log11/%x_%j.err       # 标准错误日志
#SBATCH --time=48:00:00               # 最大运行时间
#SBATCH --partition=AISS2025073101    # 队列（根据集群配置调整）
#SBATCH --nodes=1                     # 节点数量
#SBATCH --ntasks=1                    # 启动的任务数
#SBATCH --cpus-per-task=16             # 每个任务的CPU核心数（按需调整）
#SBATCH --gres=gpu:1                  # GPU数量
#SBATCH --mem=256G                     # 内存大小（按需调整）


# ========================
# conda 环境准备
# ========================

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

conda deactivate
conda activate jepa_torch
wandb offline

export https_proxy="http://cair:coy_suffocate_petrified@klb-fwproxy-01.aisc.local:3128"

# ========================
# 公共参数设置
# ========================

TASK=probing    # config dir
CKPTL_NAME="vitl_origin"
MODEL_NAME="vit_large"
checkpoint="ckpts/vitl.pt"
base_folder="logs9/${CKPTL_NAME}"

# 确保日志目录存在
mkdir -p log11

# 设置端口（可根据需要随机分配）
export MASTER_PORT=${MASTER_PORT:-$((12000 + RANDOM % 20000))}

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
# 循环执行所有数据集的probing任务
# ========================

for dataset_config in "${datasets[@]}"; do
    # 解析配置
    IFS='|' read -r FNAME DATA_NAME FOLDER_NAME <<< "$dataset_config"
    
    # 时间戳
    TIME=$(date +"%Y%m%d_%H%M")
    
    # 去掉 .yaml 后缀
    CFG_NAME=${FNAME%.yaml}
    
    # 文件夹路径
    folder="${base_folder}/${FOLDER_NAME}"
    
    # Slurm 日志路径（独立训练日志）
    LOG_FILE="log11/${TASK}_${TIME}_${CKPTL_NAME}_${DATA_NAME}.log"
    
    # 设备号
    DEVICES=$CUDA_VISIBLE_DEVICES
    
    echo "=========================================="
    echo "Starting probing for ${DATA_NAME} at $(date)"
    echo "TASK=${TASK}"
    echo "FNAME=${FNAME}"
    echo "DEVICES=${DEVICES}"
    echo "MASTER_PORT=${MASTER_PORT}"
    echo "LOG_FILE=${LOG_FILE}"
    echo "Checkpoint: ${checkpoint}"
    echo "Folder: ${folder}"
    echo "=========================================="
    
    # 启动 Python 程序
    srun python -m evals.main \
      --fname "configs/${TASK}/${FNAME}" \
      --folder "${folder}" \
      --checkpoint "${checkpoint}" \
      --model_name "${MODEL_NAME}" \
      --devices ${DEVICES} \
      --override_config_folder \
      > "${LOG_FILE}" 2>&1
    
    # 检查上一个命令的退出状态
    if [ $? -ne 0 ]; then
        echo "ERROR: Probing failed for ${DATA_NAME} at $(date)"
        echo "Check log file: ${LOG_FILE}"
        # 可以选择继续执行下一个任务或退出
        # exit 1  # 如果希望遇到错误就停止
    else
        echo "SUCCESS: Probing completed for ${DATA_NAME} at $(date)"
    fi
    
    echo ""
done

echo "All probing tasks completed at $(date)"

