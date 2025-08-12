#! /bin/bash

############################################## Run Single ###############################################
# 1. 预定义变量
# FNAME="pitvis_vith_origin_attentive_64f.yaml"   # 你的配置文件
# TASK="probing"                            # probing / tuning / pre-training 等
# DEVICES="cuda:0"

# # 2. 生成时间戳
# TIME=$(date +"%Y%m%d_%H%M")

# # 3. 去掉 .yaml 后缀, 构造日志文件名
# CFG_NAME=${FNAME%.yaml}
# LOG_FILE="logs2/${TIME}_${TASK}_${CFG_NAME}.log"


# # 4. 运行（把 nohup 的输出直接写进 LOG_FILE）
# MASTER_PORT=1233 \
# nohup python -m evals.main \
#   --fname "configs/${TASK}/${FNAME}" \
#   --devices ${DEVICES} \
#   > "${LOG_FILE}" 2>&1 &


############################################## Run Batch ###############################################
# 1. 预定义变量
FNAME="pitvis_vith_origin_attentive_64f.yaml"   # 你的配置文件
TASK="probing"                            # probing / tuning / pre-training 等
DEVICES="cuda:0"
CKPTS={
  ""
}
# 2. 生成时间戳
TIME=$(date +"%Y%m%d_%H%M")

# 3. 去掉 .yaml 后缀, 构造日志文件名
CFG_NAME=${FNAME%.yaml}
LOG_FILE="logs2/${TIME}_${TASK}_${CFG_NAME}.log"


# 4. 运行（把 nohup 的输出直接写进 LOG_FILE）
MASTER_PORT=1233 \
nohup python -m evals.main \
  --fname "configs/${TASK}/${FNAME}" \
  --devices ${DEVICES} \
  > "${LOG_FILE}" 2>&1 &
