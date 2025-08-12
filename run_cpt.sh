#! /bin/bash

# 1. 预定义变量
FNAME="pitvis_cpt_vith-256px-64f_lr1e-4_epoch-10.yaml"   # 你的配置文件
TASK="cpt"                            # probing / tuning / pre-training 等
DEVICES="cuda:0 cuda:1"

# 2. 生成时间戳
TIME=$(date +"%Y%m%d_%H%M")

# 3. 去掉 .yaml 后缀, 构造日志文件名
CFG_NAME=${FNAME%.yaml}
LOG_FILE="logs3/${TIME}_${TASK}_${CFG_NAME}.log"


# 4. 运行（把 nohup 的输出直接写进 LOG_FILE）
MASTER_PORT=1238 \
nohup \
python -m app.main \
  --fname "configs/${TASK}/${FNAME}" \
  --devices ${DEVICES} \
  > "${LOG_FILE}" 2>&1 &

