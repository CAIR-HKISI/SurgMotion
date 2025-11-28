#!/bin/bash
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=AISS2025073101
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=256G

# ========================
# 检查传入参数
# ========================
if [ -z "$FNAME" ] || [ -z "$CKPTL_NAME" ] || [ -z "$MODEL_NAME" ]; then
  echo "Error: Required variables (FNAME, CKPTL_NAME, MODEL_NAME) are not set."
  echo "Please submit via submit_batch.sh"
  exit 1
fi

# 设置默认值 (如果 LOG_ROOT 未设置，默认为 logs)
LOG_ROOT=${LOG_ROOT:-"logs"}

# ========================
# conda 环境准备
# ========================
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

conda deactivate
conda activate jepa_torch
wandb offline

export https_proxy="http://cair:coy_suffocate_petrified@klb-fwproxy-01.aisc.local:3128"

# ========================
# 路径构建
# ========================

TASK=probing
DEVICES=$CUDA_VISIBLE_DEVICES
TIME=$(date +"%Y%m%d_%H%M")

# 从 FNAME 提取数据名称
DATA_NAME=$(echo ${FNAME} | cut -d'_' -f1)

# Slurm 日志路径
LOG_FILE="${LOG_ROOT}/${CKPTL_NAME}/${TIME}_${TASK}_${DATA_NAME}.log"

# 设置端口
export MASTER_PORT=${MASTER_PORT:-$((12000 + RANDOM % 20000))}

# ========================
# 预训练模型路径
# ========================

# 动态路径：logs/模型名/数据集名
folder="${LOG_ROOT}/${CKPTL_NAME}/${DATA_NAME}"
if [ -n "${CKPT_EPOCH}" ]; then
  if [[ "${CKPT_EPOCH}" =~ ^[0-9]+$ ]]; then
    checkpoint_file="e${CKPT_EPOCH}.pt"
  else
    checkpoint_file="${CKPT_EPOCH}"
    if [[ "${checkpoint_file}" != *.pt ]]; then
      checkpoint_file="${checkpoint_file}.pt"
    fi
  fi
  checkpoint="${LOG_ROOT}/${CKPTL_NAME}/${checkpoint_file}"
else
  checkpoint="${LOG_ROOT}/${CKPTL_NAME}/latest.pt"
fi

# 确认配置与 checkpoint 路径
CONFIG_PATH="configs/${TASK}/${FNAME}"
if [ ! -f "${CONFIG_PATH}" ]; then
  echo "Error: Config file not found -> ${CONFIG_PATH}"
  exit 1
fi

if [ ! -f "${checkpoint}" ]; then
  echo "Error: Checkpoint file not found -> ${checkpoint}"
  exit 1
fi

# 确保目录存在
mkdir -p "${folder}"
mkdir -p "$(dirname "${LOG_FILE}")"

# ========================
# 启动训练任务
# ========================

echo "Starting probing at $(date)"
echo "Job: ${SLURM_JOB_NAME} ($SLURM_JOB_ID)"
echo "Config: ${FNAME}"
echo "Dataset: ${DATA_NAME}"
echo "Model Checkpoint: ${CKPTL_NAME}"
echo "Output Folder: ${folder}"

srun python -m evals.main \
  --fname "${CONFIG_PATH}" \
  --folder "${folder}" \
  --checkpoint "${checkpoint}" \
  --model_name "${MODEL_NAME}" \
  --devices ${DEVICES} \
  --override_config_folder \
  > "${LOG_FILE}" 2>&1
