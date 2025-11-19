#!/bin/bash
# 这个脚本会为每个数据集提交一个独立的slurm任务

# ========================
# 定义所有数据集配置
# ========================

# 数据集配置数组: (FNAME, DATA_NAME, FOLDER_NAME, JOB_NAME)
declare -a datasets=(
    "autolaparo_vitl_cpt_attentive_64f.yaml|autolaparo|autolaparo|prb_autolaparo"
    "cataracts_vitl_cpt_attentive_64f.yaml|cataracts|cataracts|prb_cataract"
    "cholec80_vitl_cpt_attentive_64f.yaml|cholec80|cholec80|prb_cholec80"
    "egosurgery_vitl_cpt_attentive_64f.yaml|egosurgery|egosurgery|prb_egosurgery"
    "grasp_vitl_cpt_attentive_64f.yaml|grasp|grasp|prb_grasp"
    "jigsaws_vitl_cpt_attentive_64f.yaml|jigsaws|jigsaws|prb_jigsaws"
    "m2cai_vitl_cpt_attentive_64f.yaml|m2cai|m2cai|prb_m2cai"
    "ophnet_vitl_cpt_attentive_64f.yaml|ophnet|ophnet|prb_ophnet"
    "pitvis_vitl_cpt_attentive_64f.yaml|pitvis|pitvis|prb_pitvis"
    "pmlr50_vitl_cpt_attentive_64f.yaml|pmlr50|pmlr50|prb_pmlr50"
    "polypdiag_vitl_cpt_attentive_64f.yaml|polypdiag|polypdiag|prb_polypdiag"
    "surgical_actions160_vitl_cpt_attentive_64f.yaml|surgical_actions160|surgical_actions160|prb_surgact160"
)

# ========================
# 公共参数
# ========================

TASK=probing
CKPTL_NAME="vitl_origin"
MODEL_NAME="vit_large"
checkpoint="ckpts/vitl.pt"
base_folder="logs9/${CKPTL_NAME}"

# 确保日志目录存在
mkdir -p log11

# ========================
# 为每个数据集提交独立的slurm任务
# ========================

for dataset_config in "${datasets[@]}"; do
    # 解析配置
    IFS='|' read -r FNAME DATA_NAME FOLDER_NAME JOB_NAME <<< "$dataset_config"
    
    # 文件夹路径
    folder="${base_folder}/${FOLDER_NAME}"
    
    echo "Submitting slurm job for ${DATA_NAME}..."
    
    # 使用heredoc创建临时脚本并提交
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=log11/%x_%j.out
#SBATCH --error=log11/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=AISS2025073101
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=256G

# ========================
# conda 环境准备
# ========================

# >>> conda initialize >>>
conda_path="/lustre/projects/med-multi-llm/jinlin_wu/miniconda3"

__conda_setup="\$('\${conda_path}/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ \$? -eq 0 ]; then
    eval "\$__conda_setup"
else
    if [ -f "\${conda_path}/etc/profile.d/conda.sh" ]; then
        . "\${conda_path}/etc/profile.d/conda.sh"
    else
        export PATH="\${conda_path}/bin:\$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda deactivate
conda activate jepa_torch
wandb offline

export https_proxy="http://cair:coy_suffocate_petrified@klb-fwproxy-01.aisc.local:3128"

# ========================
# 参数设置
# ========================

TASK=${TASK}
FNAME="${FNAME}"
DEVICES=\$CUDA_VISIBLE_DEVICES

# 时间戳
TIME=\$(date +"%Y%m%d_%H%M")

# 去掉 .yaml 后缀
CFG_NAME=\${FNAME%.yaml}

# 从 FNAME 提取数据名称（特殊处理）
if [ "${DATA_NAME}" = "cataracts" ] || [ "${DATA_NAME}" = "m2cai" ] || [ "${DATA_NAME}" = "surgical_actions160" ]; then
    DATA_NAME="${DATA_NAME}"
else
    DATA_NAME=\$(echo \${FNAME} | cut -d'_' -f1)
fi

# 模型名称
CKPTL_NAME="${CKPTL_NAME}"
MODEL_NAME="${MODEL_NAME}"

# Slurm 日志路径（独立训练日志）
LOG_FILE="log11/\${TASK}_\${TIME}_\${CKPTL_NAME}_\${DATA_NAME}.log"

# 确保日志目录存在
mkdir -p log11

# 设置端口（可根据需要随机分配）
export MASTER_PORT=\${MASTER_PORT:-\$((12000 + RANDOM % 20000))}

# ========================
# 预训练模型路径
# ========================

base_folder="${base_folder}"
folder="${folder}"
checkpoint="${checkpoint}"

# ========================
# 启动训练任务
# ========================

echo "Starting probing at \$(date)"
echo "TASK=\${TASK}"
echo "FNAME=\${FNAME}"
echo "DEVICES=\${DEVICES}"
echo "MASTER_PORT=\${MASTER_PORT}"
echo "LOG_FILE=\${LOG_FILE}"
echo "Checkpoint: \${checkpoint}"
echo "Folder: \${folder}"

# 启动 Python 程序
srun python -m evals.main \\
  --fname "configs/\${TASK}/\${FNAME}" \\
  --folder "\${folder}" \\
  --checkpoint "\${checkpoint}" \\
  --model_name "\${MODEL_NAME}" \\
  --devices \${DEVICES} \\
  --override_config_folder \\
  > "\${LOG_FILE}" 2>&1

EOF

    if [ $? -eq 0 ]; then
        echo "  ✓ Successfully submitted job for ${DATA_NAME}"
    else
        echo "  ✗ Failed to submit job for ${DATA_NAME}"
    fi
    echo ""
done

echo "All slurm jobs have been submitted!"

