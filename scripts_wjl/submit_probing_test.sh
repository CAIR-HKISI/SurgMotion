#!/bin/bash

# ==========================================
# 用于测试单个 probing 配置的脚本
# 调用示例：
#   bash submit_probing_test.sh \
#     atlas_probe_attentive_64f.yaml \
#     cooldown_vitg-256px-64f_4epoch \
#     vit_giant_xformers
#
# 第 1 个参数：必填，config 名称或路径（支持：
#   atlas_probe_attentive_64f.yaml
#   probing/atlas_probe_attentive_64f.yaml
#   configs/probing/atlas_probe_attentive_64f.yaml）
# 第 2 个参数：可选，CKPTL_NAME（默认：cooldown_vitg-256px-64f_4epoch）
# 第 3 个参数：可选，MODEL_NAME（默认：vit_giant_xformers）
# ==========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 日志根目录（可按需修改或通过环境变量覆盖）
LOG_ROOT="${LOG_ROOT:-logs}"

FNAME="$1"
# 统一规范 FNAME，避免在 run_probing.sh 中出现 configs/probing/probing/... 的情况
# 支持传入三种形式：
#   atlas_probe_attentive_64f.yaml
#   probing/atlas_probe_attentive_64f.yaml
#   configs/probing/atlas_probe_attentive_64f.yaml
FNAME="${FNAME#configs/}"   # 去掉前缀 configs/
FNAME="${FNAME#probing/}"   # 去掉前缀 probing/
FNAME="$(basename "${FNAME}")"
CKPTL_NAME="${2:-cooldown_vitg-256px-64f_4epoch}"
MODEL_NAME="${3:-vit_giant_xformers}"

if [ -z "${FNAME}" ]; then
  echo "用法：bash $(basename "$0") FNAME [CKPTL_NAME] [MODEL_NAME]"
  echo "  示例：bash $(basename "$0") atlas_probe_attentive_64f.yaml cooldown_vitg-256px-64f_4epoch vit_giant_xformers"
  exit 1
fi

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


