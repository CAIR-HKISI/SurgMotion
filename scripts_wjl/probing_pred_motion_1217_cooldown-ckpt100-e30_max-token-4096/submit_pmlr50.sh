#!/bin/bash
#SBATCH --job-name=prb_pmlr50
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=AISS2025073101
#SBATCH --nodelist=klb-dgx-015,klb-dgx-120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=256G


# ========================
# 任务特定配置
# ========================
FNAME="pmlr50_probe_attentive_64f.yaml"
TASK="probing_pred_motion"
CKPTL_DIR="logs/pred-motion-v3_vitgx-256px-16f_multi-scale_jepa-l1/cooldown-ckpt100-e30_max-token-4096"
CKPT_EPOCH="latest.pt"
MODEL_NAME="vit_giant_xformers"
timestamp="1217"
CKPTL_NAME="cooldown-ckpt100-e30_max-token-4096"
# ========================
# 检查传入参数
# ========================

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

# 2. 设置代理 (建议 http 和 https 都设置，以防万一)
export http_proxy="http://cair:coy_suffocate_petrified@klb-fwproxy-01.aisc.local:3128"
export https_proxy="http://cair:coy_suffocate_petrified@klb-fwproxy-01.aisc.local:3128"
export HTTP_PROXY="http://cair:coy_suffocate_petrified@klb-fwproxy-01.aisc.local:3128"
export HTTPS_PROXY="http://cair:coy_suffocate_petrified@klb-fwproxy-01.aisc.local:3128"


# 2. [新增] 忽略 SSL 证书验证错误 (解决 x509 报错的关键)
export WANDB_INSECURE_DISABLE_SSL=true


# 3. 强制 WandB 为在线模式 (确保上传)
export WANDB_MODE=online

# ========================
# 路径构建
# ========================

TASK=${TASK:-probing}
DEVICES=$CUDA_VISIBLE_DEVICES
TIME=$(date +"%Y%m%d_%H%M")

# 从 FNAME 提取数据名称
DATA_NAME=$(echo ${FNAME} | cut -d'_' -f1)

# Slurm 日志路径
LOG_FILE="${CKPTL_DIR}/${TIME}_${TASK}_${DATA_NAME}.log"

folder="${CKPTL_DIR}/${DATA_NAME}"
checkpoint="${CKPTL_DIR}/${CKPT_EPOCH}"

mkdir -p "${folder}"
mkdir -p "$(dirname "${LOG_FILE}")"


# 设置端口
export MASTER_PORT=${MASTER_PORT:-$((12000 + RANDOM % 20000))}

# ========================
# 预训练模型路径
# ========================


# 确认配置与 checkpoint 路径
CONFIG_PATH="configs/${TASK}/${FNAME}"
if [ ! -f "${CONFIG_PATH}" ]; then
  echo "Error: Config file not found -> ${CONFIG_PATH}"
  exit 1
fi

# 确保目录存在
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
