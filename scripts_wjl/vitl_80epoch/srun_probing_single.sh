#!/bin/bash
# 注意：job-name 和 output/error 路径在提交时通过 sbatch 参数设置
#SBATCH --output=log/%x_%j.out      # 标准输出日志（自动包含作业号）
#SBATCH --error=log/%x_%j.err       # 标准错误日志
#SBATCH --time=48:00:00               # 最大运行时间
#SBATCH --partition=AISS2025073101    # 队列（根据集群配置调整）
#SBATCH --nodelist=klb-dgx-011,klb-dgx-120    # 指定节点
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
# 参数解析 & 环境准备
# ========================

# 从命令行参数获取数据集信息
TASK=probing    # config dir
FNAME="${1}"  # 配置文件名称，例如 autolaparo_vitl_cpt_attentive_64f.yaml
DATA_NAME="${2}"  # 数据集名称，例如 autolaparo
FOLDER_NAME="${3}"  # 文件夹名称，例如 autolaparo

# 时间戳
TIME=$(date +"%Y%m%d_%H%M")

# 去掉 .yaml 后缀
CFG_NAME=${FNAME%.yaml}

# 模型名称
log_root="logs"
CKPTL_NAME="surgical_cpt_vitl16-256px-64f_lr1e-20_epoch_21-dataset_2nd_80epochs"
MODEL_NAME="vit_large"

# Slurm 日志路径（独立训练日志）
LOG_FILE="${log_root}/${CKPTL_NAME}/${TIME}_${TASK}_${DATA_NAME}.log"

# 设置端口（可根据需要随机分配）
export MASTER_PORT=${MASTER_PORT:-$((12000 + RANDOM % 20000))}

# ========================
# 预训练模型路径
# ========================

folder="${log_root}/${CKPTL_NAME}/${FOLDER_NAME}"
checkpoint="${log_root}/${CKPTL_NAME}/latest.pt"

# 确保日志目录存在
mkdir -p "${log_root}/${CKPTL_NAME}"

# ========================
# 启动训练任务
# ========================

echo "Starting probing at $(date)"
echo "TASK=${TASK}"
echo "FNAME=${FNAME}"
echo "DATA_NAME=${DATA_NAME}"
echo "DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "LOG_FILE=${LOG_FILE}"
echo "Checkpoint: ${checkpoint}"
echo "Folder: ${folder}"

# 启动 Python 程序
srun python -m evals.main \
  --fname "configs/${TASK}/${FNAME}" \
  --folder "${folder}" \
  --checkpoint "${checkpoint}" \
  --model_name "${MODEL_NAME}" \
  --devices ${CUDA_VISIBLE_DEVICES} \
  --override_config_folder \
  > "${LOG_FILE}" 2>&1

if [ $? -eq 0 ]; then
    echo "SUCCESS: Probing completed for ${DATA_NAME} at $(date)"
else
    echo "ERROR: Probing failed for ${DATA_NAME} at $(date)"
    echo "Check log file: ${LOG_FILE}"
    exit 1
fi

