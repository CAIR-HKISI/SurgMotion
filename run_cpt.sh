#! /bin/bash

# 1. 预定义变量
FNAME="autolapro_cpt_vith-256px-64f_lr1e-4_epoch-20.yaml"   # 你的配置文件
TASK="cpt_autolaparo"                            # probing / tuning / pre-training 等
DEVICES="cuda:1 cuda:4"
MASTER_PORT=1261

# 2. 生成时间戳
TIME=$(date +"%Y%m%d_%H%M")

# 3. 去掉 .yaml 后缀, 构造日志文件名
CFG_NAME=${FNAME%.yaml}
LOG_FILE="logs6/${TIME}_${TASK}_${CFG_NAME}.log"


# 4. 运行（把 nohup 的输出直接写进 LOG_FILE）
MASTER_PORT=${MASTER_PORT} \
nohup \
python -m app.main \
  --fname "configs/${TASK}/${FNAME}" \
  --devices ${DEVICES} \
  > "${LOG_FILE}" 2>&1 &

