#! /bin/bash

# 1. 预定义变量

# FNAME="cholec80_cpt_vitl-256px-64f_lr1e-4_epoch-20.yaml"
# TASK="cpt_cholec80"                            # probing / tuning / pre-training 等
# DEVICES="cuda:2 cuda:3"
# MASTER_PORT=1265

# FNAME="cholec80_cpt_vitl-256px-128f_lr1e-4_epoch-20.yaml"
# TASK="cpt_cholec80"                            # probing / tuning / pre-training 等
# DEVICES="cuda:0 cuda:1"
# MASTER_PORT=1269


# FNAME="bernbypass_cpt_vith-256px-64f_lr1e-4_epoch-20.yaml"
# TASK="cpt_bypass"                            # probing / tuning / pre-training 等
# DEVICES="cuda:4 cuda:5"
# MASTER_PORT=1293

# FNAME="cholec80_cpt_vitl-256px-64f_lr1e-4_epoch-20_large-mask.yaml"
# TASK="cpt_cholec80"                            # probing / tuning / pre-training 等
# DEVICES="cuda:0 cuda:1"
# MASTER_PORT=1219

# FNAME="cholec80_cpt_vith-256px-16f_stage-1.yaml"
# TASK="cpt_cholec80_v2"                            # probing / tuning / pre-training 等
# DEVICES="cuda:6 cuda:7"
# MASTER_PORT=1322

FNAME="cholec80_cpt_vith-256px-16-64f_stage-2.yaml"
TASK="cpt_cholec80_v2"
DEVICES="cuda:6 cuda:7"
MASTER_PORT=1323


# FNAME="cholec80_cpt_vith-256px-64f_stage-2.yaml"
# TASK="cpt_cholec80_v2"                            # probing / tuning / pre-training 等
# DEVICES="cuda:0 cuda:1"
# MASTER_PORT=1321

# FNAME="cholec80_cpt_vith-256px-64f_lr1e-4_epoch-50.yaml"
# TASK="cpt_cholec80"
# DEVICES="cuda:0 cuda:1"
# MASTER_PORT=1299

# 2. 生成时间戳
TIME=$(date +"%Y%m%d_%H%M")

# 3. 去掉 .yaml 后缀, 构造日志文件名
CFG_NAME=${FNAME%.yaml}
LOG_FILE="logs8/${TASK}_${TIME}_${CFG_NAME}.log"


# 4. 运行（把 nohup 的输出直接写进 LOG_FILE）
MASTER_PORT=${MASTER_PORT} \
nohup \
python -m app.main \
  --fname "configs/${TASK}/${FNAME}" \
  --devices ${DEVICES} \
  > "${LOG_FILE}" 2>&1 &

