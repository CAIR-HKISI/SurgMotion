#! /bin/bash




FNAME="cholec80_vitl_cpt_attentive_64f.yaml"
TASK="probing_cholec80"
MASTER_PORT="1309"
DEVICES="cuda:6"

# 2. 生成时间戳
TIME=$(date +"%Y%m%d_%H%M")

# 3. 去掉 .yaml 后缀, 构造日志文件名
CFG_NAME=${FNAME%.yaml}
LOG_FILE="logs/${TASK}/${TIME}_${CFG_NAME}.log"

# 4. 运行（把 nohup 的输出直接写进 LOG_FILE）
MASTER_PORT=${MASTER_PORT} \
nohup \
python -m evals.main \
  --fname "configs/${TASK}/${FNAME}" \
  --devices ${DEVICES} \
  --val_only \
  > "${LOG_FILE}" 2>&1 &
