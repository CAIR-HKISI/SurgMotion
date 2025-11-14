#!/bin/bash
#SBATCH --job-name=prb_pitvis          # 作业名
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
# 参数解析 & 环境准备
# ========================

# 传入运行参数
TASK=probing    # config dir
FNAME="pitvis_vitl_cpt_attentive_64f.yaml"
DEVICES=$CUDA_VISIBLE_DEVICES  # 设备号，例如 0,1

# 时间戳
TIME=$(date +"%Y%m%d_%H%M")

# 去掉 .yaml 后缀
CFG_NAME=${FNAME%.yaml}

# 从 FNAME 提取数据名称（第一个下划线前的部分）
DATA_NAME=$(echo ${FNAME} | cut -d'_' -f1)

# 模型名称
CKPTL_NAME="vitl_origin"
MODEL_NAME="vit_large"

# Slurm 日志路径（独立训练日志）
LOG_FILE="log11/${TASK}_${TIME}_${CKPTL_NAME}_${DATA_NAME}.log"

# 确保日志目录存在
mkdir -p log11

# 设置端口（可根据需要随机分配）
export MASTER_PORT=${MASTER_PORT:-$((12000 + RANDOM % 20000))}

# ========================
# 预训练模型路径
# ========================

base_folder="logs9/${CKPTL_NAME}"
folder="${base_folder}/pitvis"
checkpoint="ckpts/vitl.pt"

# ========================
# 启动训练任务
# ========================

echo "Starting probing at $(date)"
echo "TASK=${TASK}"
echo "FNAME=${FNAME}"
echo "DEVICES=${DEVICES}"
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
  --devices ${DEVICES} \
  --override_config_folder \
  > "${LOG_FILE}" 2>&1

