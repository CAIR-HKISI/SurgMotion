#! /bin/bash


# 1. 预定义变量

# FNAME="cholec80_vitl_cpt_attentive_64f_debug.yaml"
# TASK="probing_cholec80"
# MASTER_PORT="1256"
# DEVICES="cuda:2"

# FNAME="cholec80_vith_cpt_attentive_64f.yaml"
# TASK="probing_cholec80"
# MASTER_PORT="1257"
# DEVICES="cuda:2"


# FNAME="cholec80_vitl_cpt_attentive_64f_cls-mid-weight_epoch-1.yaml"
# TASK="probing_cholec80"
# MASTER_PORT="1239"
# DEVICES="cuda:3"

# FNAME="cholec80_vitl_cpt_attentive_64f_cls-mid-weight_epoch-1_bacth-64.yaml"
# TASK="probing_cholec80"
# MASTER_PORT="1229"
# DEVICES="cuda:4"


# FNAME="cholec80_vitl_cpt_attentive_64f_cls-mid-weight_epoch-4.yaml"
# TASK="probing_cholec80"
# MASTER_PORT="1209"
# DEVICES="cuda:5"

# FNAME="cholec80_vith_cpt_attentive_64f_cls-mid-weight_epoch-1.yaml"
# TASK="probing_cholec80"
# MASTER_PORT="1289"
# DEVICES="cuda:6"

# FNAME="cholec80_vith_cpt_attentive_64f_epoch-50.yaml"
# TASK="probing_cholec80_v2"
# MASTER_PORT="1279"
# DEVICES="cuda:2"

# FNAME="cholec80_vith_cpt_attentive_64f_epoch-50_multi-head.yaml"
# TASK="probing_cholec80_v2"
# MASTER_PORT="1399"
# DEVICES="cuda:5"

# FNAME="cholec80_vith_cpt_attentive_64f_epoch-50_multi-head_stage2.yaml"
# TASK="probing_cholec80_v2"
# MASTER_PORT="1399"
# DEVICES="cuda:3"


# FNAME="cholec80_vitl_cpt_attentive_64f_epoch-50_multi-head_mask-ratio-0.9.yaml"
# TASK="probing_cholec80"
# MASTER_PORT="1341"
# DEVICES="cuda:6"

# FNAME="cholec80_vitl_cpt_attentive_64f_epoch-50_mask-ratio-0.9.yaml"
# FNAME="cholec80_vitl_cpt_attentive_64f_multi-head.yaml"
# TASK="probing_cholec80"
# MASTER_PORT="1345"
# DEVICES="cuda:7"


# FNAME="autolaparo_vitl_origin_attentive_64f_4epoch.yaml"
# FNAME="cholec80_vith_cpt_attentive_64f_cls-mid-weight_epoch-1.yaml"
# TASK="probing_cholec80"
# MASTER_PORT="1345"
# DEVICES="cuda:7"


# FNAME="m2cai_vitl_cpt_attentive_64f_epoch-1_multi-head.yaml"
# TASK="probing_m2cai"
# MASTER_PORT="1349"
# DEVICES="cuda:0"


# FNAME="cholec80_vith_cpt_attentive_64f_epoch-1_multihead.yaml"
# TASK="probing_cholec80"
# MASTER_PORT="1347"
# DEVICES="cuda:6"

FNAME="autolaparo_vitl_cpt_attentive_64f_50epoch.yaml"
TASK="probing_autolaparo"
MASTER_PORT="1389"
DEVICES="cuda:6"

# FNAME="cholec80_vitl_cpt_attentive_64f_epoch-1_multihead.yaml"
# TASK="probing_cholec80"
# MASTER_PORT="1349"
# DEVICES="cuda:9"

# 2. 生成时间戳
TIME=$(date +"%Y%m%d_%H%M")

# 3. 去掉 .yaml 后缀, 构造日志文件名
CFG_NAME=${FNAME%.yaml}
LOG_FILE="logs9/${TASK}_${TIME}_${CFG_NAME}.log"

# 4. 运行（把 nohup 的输出直接写进 LOG_FILE）
MASTER_PORT=${MASTER_PORT} \
nohup \
python -m evals.main \
  --fname "configs/${TASK}/${FNAME}" \
  --devices ${DEVICES} \
  > "${LOG_FILE}" 2>&1 &
