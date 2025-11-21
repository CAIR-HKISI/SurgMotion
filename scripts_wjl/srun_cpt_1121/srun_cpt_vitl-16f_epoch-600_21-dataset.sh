#!/bin/bash
#SBATCH --job-name=CPT_VITG_384          # 作业名
#SBATCH --output=logs9/%x_%j.out      # 标准输出日志（自动包含作业号）
#SBATCH --error=logs9/%x_%j.err       # 标准错误日志
#SBATCH --time=48:00:00               # 最大运行时间
#SBATCH --partition=AISS2025073101    # 队列（根据集群配置调整）
#SBATCH --nodelist=klb-dgx-011,klb-dgx-120    # 指定节点
#SBATCH --nodes=1                     # 节点数量
#SBATCH --ntasks=1                    # 启动的任务数
#SBATCH --cpus-per-task=16             # 每个任务的CPU核心数（按需调整）
#SBATCH --gres=gpu:4                  # GPU数量
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

# 传入运行参数
TASK=multidata_cpt_1121     # 例如：classification 或 segmentation
FNAME=cpt_vitg-256px-16f_600epoch.yaml  # 配置文件名，例如 config.yaml

DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ')
echo "DEVICES=${DEVICES}"


# 时间戳
TIME=$(date +"%Y%m%d_%H%M")

# 去掉 .yaml 后缀
CFG_NAME=${FNAME%.yaml}

# Slurm 日志路径（独立训练日志）
LOG_FILE="logs/${CFG_NAME}/${TIME}_${TASK}_${CFG_NAME}.log"

# 确保日志目录存在
mkdir -p "logs/${CFG_NAME}"


# 设置端口（可根据需要随机分配）
export MASTER_PORT=${MASTER_PORT:-$((12000 + RANDOM % 20000))}

# ========================
# 启动训练任务
# ========================

echo "Starting training at $(date)"
echo "TASK=${TASK}"
echo "FNAME=${FNAME}"
echo "DEVICES=${DEVICES}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "LOG_FILE=${LOG_FILE}"
echo "MASTER_PORT=${MASTER_PORT}"

srun python -m app.main \
  --fname "configs/${TASK}/${FNAME}" \
  --devices ${DEVICES} \
  > "${LOG_FILE}" 2>&1

