#! /bin/bash


# 1. 预定义变量
# FNAME="pitvis_cpt_epoch-2_vith16-256px-64f_lr1e-4.yaml"   # 你的配置文件
# DEVICES="cuda:0"
# MASTER_PORT=1233
# TASK="probing_cpt"  

# FNAME="pitvis_cpt_epoch-2_vitl16-256px-64f_lr1e-4.yaml"   # 你的配置文件
# DEVICES="cuda:1"
# MASTER_PORT=1234
# TASK="probing_cpt"  

# FNAME="pitvis_cpt_epoch-2_vitl16-384px-64f_lr1e-4.yaml"   # 你的配置文件
# DEVICES="cuda:3"
# MASTER_PORT=1239
# TASK="probing_cpt"                            # probing / tuning / pre-training 等

# FNAME="pitvis_cpt_epoch-10_vitl16-256px-64f_lr1e-4.yaml"   # 你的配置文件
# DEVICES="cuda:3"
# MASTER_PORT=1236
# TASK="probing_cpt"                            # probing / tuning / pre-training 等

# FNAME="pitvis_cpt_vith-256px-64f_lr1e-4_epoch-10.yaml"
# DEVICES="cuda:4"
# MASTER_PORT=1231
# TASK="probing_cpt"                            # probing / tuning / pre-training 等

# FNAME="autolaparo_vitl_origin_attentive_64f_lr-5e-6.yaml"
# DEVICES="cuda:0"
# MASTER_PORT=1238
# TASK="probing_autolaparo"                            # probing / tuning / pre-training 等

# FNAME="autolaparo_vitl_origin_attentive_64f_lr-2e-5.yaml"
# DEVICES="cuda:5"
# MASTER_PORT=1249
# TASK="probing_autolaparo"                            # probing / tuning / pre-training 等

# FNAME="autolaparo_vitl_origin_attentive_16f_multi-head.yaml"
# DEVICES="cuda:4"
# MASTER_PORT=1259
# TASK="probing_autolaparo"                            # probing / tuning / pre-training 等


FNAME="cholec80_vitl_cpt_attentive_64f_sft.yaml"
DEVICES="cuda:3 cuda:4"
MASTER_PORT=1270
TASK="probing_cholec80_v2"                            # probing / tuning / pre-training 等

# 2. 生成时间戳
TIME=$(date +"%Y%m%d_%H%M")

# 3. 去掉 .yaml 后缀, 构造日志文件名
CFG_NAME=${FNAME%.yaml}
LOG_FILE="logs7/${TIME}_${TASK}_${CFG_NAME}.log"


# 4. 运行（把 nohup 的输出直接写进 LOG_FILE）
MASTER_PORT=${MASTER_PORT} \
nohup \
python -m evals.main \
  --fname "configs/${TASK}/${FNAME}" \
  --devices ${DEVICES} \
  > "${LOG_FILE}" 2>&1 &
