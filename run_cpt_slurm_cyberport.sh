#!/bin/bash
#SBATCH --job-name=jepa_pretrain         # 作业名
#SBATCH --output=logs9/%x_%j.out      # 标准输出日志（自动包含作业号）
#SBATCH --error=logs9/%x_%j.err       # 标准错误日志
#SBATCH --time=48:00:00               # 最大运行时间
#SBATCH --partition=AISS2025073101    # 队列（根据集群配置调整）
#SBATCH --nodes=1                     # 节点数量
#SBATCH --ntasks=1                    # 启动的任务数
#SBATCH --cpus-per-task=2             # 每个任务的CPU核心数（按需调整）
#SBATCH --gres=gpu:2                  # GPU数量
#SBATCH --mem=128G                    # 内存大小（按需调整）

# ========================
# conda 环境准备
# ========================

conda_path="/lustre/projects/med-multi-llm/jinlin_wu/miniconda3"

# >>> conda initialize >>>
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


# ========================
# 参数解析 & 环境准备
# ========================

# 传入运行参数
TASK=multidata_cpt     # 例如：classification 或 segmentation
# # FNAME=cpt_vitl-256px-64f_lr1e-4_epoch-10.yaml    # 配置文件名，例如 config.yaml
# # FNAME="cpt_vitl-256px-64f_lr1e-4_epoch-10_neuro.yaml"
# FNAME="cpt_vitl-256px-64f_lr1e-4_epoch-10_21-dataset_40epoch.yaml"
# FNAME="cpt_vitl-256px-64f_lr1e-4_epoch-10_21-dataset_80epoch.yaml"
FNAME="cpt_vith-256px-64f_lr1e-4_epoch-20_21-dataset.yaml"
# FNAME="cpt_vitg-256px-64f_lr1e-4_epoch-20_21-dataset.yaml"

DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ')
echo "DEVICES=${DEVICES}"


# TASK="multidata_ntp"
# FNAME="window_predict_vitl-256px-64f_lr1e-4_epoch-10_21-dataset.yaml"
# # DEVICES=$CUDA_VISIBLE_DEVICES  # 设备号，例如 0,1
# DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ')
# # echo "DEVICES=${DEVICES}"

# 时间戳
TIME=$(date +"%Y%m%d_%H%M")

# 去掉 .yaml 后缀
CFG_NAME=${FNAME%.yaml}

# Slurm 日志路径（独立训练日志）
LOG_FILE="logs9/${TASK}_${TIME}_${CFG_NAME}.log"

# 确保日志目录存在
mkdir -p logs9



# ========================
# 启动训练任务
# ========================

echo "Starting training at $(date)"
echo "TASK=${TASK}"
echo "FNAME=${FNAME}"
echo "DEVICES=${DEVICES}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "LOG_FILE=${LOG_FILE}"


srun python -m app.main \
  --fname "configs/${TASK}/${FNAME}" \
  --devices ${DEVICES} \
  > "${LOG_FILE}" 2>&1

