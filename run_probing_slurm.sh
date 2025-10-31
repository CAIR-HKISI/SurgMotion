#!/bin/bash
#SBATCH --job-name=jepa_train          # 作业名
#SBATCH --output=logs9/%x_%j.out      # 标准输出日志（自动包含作业号）
#SBATCH --error=logs9/%x_%j.err       # 标准错误日志
#SBATCH --time=48:00:00               # 最大运行时间
#SBATCH --partition=a100               # 队列（根据集群配置调整）
#SBATCH --nodes=1                     # 节点数量
#SBATCH --ntasks=1                    # 启动的任务数
#SBATCH --cpus-per-task=8             # 每个任务的CPU核心数（按需调整）
#SBATCH --gres=gpu:2                  # GPU数量
#SBATCH --mem=128G                     # 内存大小（按需调整）


# ========================
# conda 环境准备
# ========================

# >>> conda initialize >>>
__conda_setup="$('/home/jinlin_wu/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/jinlin_wu/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/home/jinlin_wu/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/home/jinlin_wu/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda deactivate
conda activate jepa_torch


# ========================
# 参数解析 & 环境准备
# ========================

# 传入运行参数
TASK=probing    # 例如：classification 或 segmentation
# FNAME=cpt_vitl-256px-64f_lr1e-4_epoch-10.yaml    # 配置文件名，例如 config.yaml
# FNAME="cpt_vitl-256px-64f_lr1e-4_epoch-10_neuro.yaml"
FNAME="pitvis_vitl_cpt_attentive_64f.yaml"
DEVICES=$CUDA_VISIBLE_DEVICES  # 设备号，例如 0,1

# 时间戳
TIME=$(date +"%Y%m%d_%H%M")

# 去掉 .yaml 后缀
CFG_NAME=${FNAME%.yaml}

# Slurm 日志路径（独立训练日志）
LOG_FILE="logs10/${TASK}_${TIME}_${CFG_NAME}.log"

# 确保日志目录存在
mkdir -p logs10

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

# 启动 Python 程序
srun python -m evals.main \
  --fname "configs/${TASK}/${FNAME}" \
  --devices ${DEVICES} \
  > "${LOG_FILE}" 2>&1

