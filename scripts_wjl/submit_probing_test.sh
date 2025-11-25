#!/bin/bash

# ==========================================
# 用于测试单个 probing 配置的脚本
# 调用示例：
#   bash submit_probing_test.sh \
#     configs/probing/atlas_probe_attentive_64f.yaml \
#     cooldown_vitg-256px-64f_4epoch \
#     vit_giant_xformers
#
# 第 1 个参数：必填，config 路径（相对 configs 根目录或带子目录）
# 第 2 个参数：可选，CKPTL_NAME（默认：cooldown_vitg-256px-64f_4epoch）
# 第 3 个参数：可选，MODEL_NAME（默认：vit_giant_xformers）
# ==========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 日志根目录（可按需修改或通过环境变量覆盖）
LOG_ROOT="${LOG_ROOT:-logs}"

FNAME="$1"
CKPTL_NAME="${2:-cooldown_vitg-256px-64f_4epoch}"
MODEL_NAME="${3:-vit_giant_xformers}"

if [ -z "${FNAME}" ]; then
  echo "用法：bash $(basename "$0") FNAME [CKPTL_NAME] [MODEL_NAME]"
  echo "  示例：bash $(basename "$0") probing/atlas_probe_attentive_64f.yaml cooldown_vitg-256px-64f_4epoch vit_giant_xformers"
  exit 1
fi

# 允许用户既可以传 'atlas_probe_attentive_64f.yaml'
# 也可以传 'probing/atlas_probe_attentive_64f.yaml'
# 对应 run_probing.sh 里会再拼上 'configs/${TASK}/'

# 提取数据集名称 (例如 atlas)
DATA_NAME=$(echo "${FNAME}" | xargs basename | cut -d'_' -f1)

JOB_NAME="prb_test_${DATA_NAME}"

echo "Submitting TEST task:"
echo "  -> FNAME      : ${FNAME}"
echo "  -> CKPTL_NAME : ${CKPTL_NAME}"
echo "  -> MODEL_NAME : ${MODEL_NAME}"
echo "  -> JOB_NAME   : ${JOB_NAME}"
echo "  -> LOG_ROOT   : ${LOG_ROOT}"

sbatch \
  --job-name="${JOB_NAME}" \
  --export=ALL,FNAME="${FNAME}",LOG_ROOT="${LOG_ROOT}",CKPTL_NAME="${CKPTL_NAME}",MODEL_NAME="${MODEL_NAME}" \
  "${SCRIPT_DIR}/run_probing.sh"

echo "Test job submitted."


