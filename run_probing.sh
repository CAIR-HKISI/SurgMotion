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


# FNAME="autolaparo_vitl_origin_attentive_64f.yaml"
# DEVICES="cuda:0"
# MASTER_PORT="1250"
# TASK="probing_autolaparo"                            # probing / tuning / pre-training 等

# FNAME="autolaparo_vith_origin_attentive_64f.yaml"
# DEVICES="cuda:1"
# MASTER_PORT="1251"

# FNAME="autolaparo_vitl_origin_attentive_64f_wd.yaml"
# DEVICES="cuda:4"
# MASTER_PORT="1253"
# TASK="probing_autolaparo"                            # probing / tuning / pre-training 等


# FNAME="autolaparo_vith_origin_attentive_64f_wd.yaml"
# DEVICES="cuda:6"
# MASTER_PORT="1258"
# TASK="probing_autolaparo"                            # probing / tuning / pre-training 等

# FNAME="autolaparo_vitl_origin_attentive_64f_4epoch.yaml"
# DEVICES="cuda:0"
# MASTER_PORT="1281"
# TASK="probing_autolaparo"                            # probing / tuning / pre-training 等


# FNAME="pitvis_vitl_origin_attentive_64f.yaml"
# DEVICES="cuda:1"
# MASTER_PORT="1291"
# TASK="probing_bypass"                            # probing / tuning / pre-training 等

# FNAME="autolaparo_vitl_pitvis_attentive_64f_4epoch.yaml"
# DEVICES="cuda:0"
# MASTER_PORT="1251"
# TASK="probing_autolaparo"                            # probing / tuning / pre-training 等

# FNAME="pitvis_cpt_epoch-2_vitl16-256px-128f_lr1e-4.yaml"
# TASK="probing_pitvis"
# MASTER_PORT="1252"
# DEVICES="cuda:1"

# FNAME="bypass_vith_origin_attentive_64f_4epoch.yaml"
# TASK="probing_bypass"
# MASTER_PORT="1253"
# DEVICES="cuda:0"


# FNAME="bypass_vith_origin_attentive_64f_10epoch.yaml"
# TASK="probing_bypass"
# MASTER_PORT="1254"
# DEVICES="cuda:2"

# FNAME="autolaparo_vith_origin_attentive_64f_10epoch.yaml"
# TASK="probing_autolaparo"
# MASTER_PORT="1255"
# DEVICES="cuda:3"

FNAME="autolaparo_vitl_cpt_attentive_64f_10epoch.yaml"
TASK="probing_autolaparo"
MASTER_PORT="1256"
DEVICES="cuda:5"


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
