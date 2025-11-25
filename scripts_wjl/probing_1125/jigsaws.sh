#!/bin/bash
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=AISS2025073101
#SBATCH --nodelist=klb-dgx-011,klb-dgx-120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=256G

# jigsaws probing 独立脚本

FNAME="jigsaws_probe_attentive_64f.yaml"

CKPTL_NAME="${CKPTL_NAME:-${1:-cooldown_vitg-256px-64f_4epoch}}"
MODEL_NAME="${MODEL_NAME:-${2:-vit_giant_xformers}}"
LOG_ROOT="${LOG_ROOT:-logs_test}"

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

TASK=probing
DEVICES=$CUDA_VISIBLE_DEVICES
TIME=$(date +"%Y%m%d_%H%M")

DATA_NAME=$(echo ${FNAME} | cut -d'_' -f1)
LOG_FILE="${LOG_ROOT}/${CKPTL_NAME}/${TIME}_${TASK}_${DATA_NAME}.log"

export MASTER_PORT=${MASTER_PORT:-$((12000 + RANDOM % 20000))}

folder="${LOG_ROOT}/${CKPTL_NAME}/${DATA_NAME}"
checkpoint="${LOG_ROOT}/${CKPTL_NAME}/latest.pt"

mkdir -p "${folder}"
mkdir -p "$(dirname "${LOG_FILE}")"

echo "Starting probing at $(date)"
echo "Job: ${SLURM_JOB_NAME} ($SLURM_JOB_ID)"
echo "Config: ${FNAME}"
echo "Dataset: ${DATA_NAME}"
echo "Model Checkpoint: ${CKPTL_NAME}"
echo "Output Folder: ${folder}"

srun python -m evals.main \
  --fname "configs/${TASK}/${FNAME}" \
  --folder "${folder}" \
  --checkpoint "${checkpoint}" \
  --model_name "${MODEL_NAME}" \
  --devices ${DEVICES} \
  --override_config_folder \
  > "${LOG_FILE}" 2>&1



