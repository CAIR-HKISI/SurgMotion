#! /bin/bash

######################### eval vitl origin #########################################################
# frames_per_clip = (16, 32, 64)

# 每个循环随机生成一个新的端口号
# MASTER_PORT=$((RANDOM % 55536 + 10000))
# frames_per_clip=(32 64 16)
# for frame in "${frames_per_clip[@]}"; do
    
#     # 定义日志目录
#     folder="logs/vitl_origin_probing_eval_${frame}frames"
    
#     # 创建日志目录（如果不存在）
#     mkdir -p "${folder}"
    
#     echo "Running evaluation for ${frame} frames on port ${MASTER_PORT}"

#     # 后台运行并将输出记录到日志文件
#     MASTER_PORT=${MASTER_PORT} nohup python -m evals.main \
#         --fname configs/pitvis_eval/vitl_origin.yaml \
#         --checkpoint /data/wjl/vjepa2/ckpts/vitl.pt \
#         --folder ${folder} \
#         --override_config_folder \
#         --frames_per_clip ${frame} \
#         --batch_size 16 \
#         --devices cuda:0 \
#         > "${folder}/probing_eval.log" 2>&1 
# done


######################### eval pitvispretrained vitl #########################################################

ckpt_dirs=(
# "logs/cpt_vitl16-256px-32f_cooldown"
# "logs/cpt_vitl16-256px-32f_cooldown_complement_masking"
# "logs/cpt_vitl16-256px-32f_cooldown_lr4e-4"
# "logs/cpt_vitl16-256px-32f_cooldown_lr5e-4"
"logs/cpt_vitl16-256px-32f_cooldown_lr2e-4"
# "logs/cpt_vitl16-256px-32f_cooldown_lr2e-5"
# "logs/cpt_vitl16-256px-32f_cooldown_lr5e-5"
# "logs/cpt_vitl16-256px-32f_cooldown_mask_large"
# "logs/cpt_vitl16-256px-32f_cooldown_mask_small"
# "logs/cpt_vitl16-256px-32f_cooldown_resolution_384"
# "logs/cpt_vitl16-256px-32f_cooldown_resolution_448" 
)

for ckpt_dir in "${ckpt_dirs[@]}"; do
    # 每次循环随机生成新的端口号
    MASTER_PORT=$((RANDOM % 55536 + 10000))

    # 根据目录名判断需要使用的分辨率
    if [[ "$ckpt_dir" == *"384"* ]]; then
        resolution=384
    elif [[ "$ckpt_dir" == *"448"* ]]; then
        resolution=448
    else
        resolution=256
    fi

    # 定义日志目录
    folder="${ckpt_dir}/probing_eval"
    
    # 创建日志目录（如果不存在）
    mkdir -p "${folder}"
    
    echo "Running evaluation for checkpoint ${ckpt_dir} with resolution ${resolution}, on port ${MASTER_PORT}"

    # 后台运行并保存日志
    MASTER_PORT=${MASTER_PORT} nohup python -m evals.main \
        --fname configs/pitvis_eval/vitl_pretrain_pitvis.yaml \
        --checkpoint /data/wjl/vjepa2/${ckpt_dir}/latest.pt \
        --folder ${folder} \
        --override_config_folder \
        --frames_per_clip 32 \
        --batch_size 16 \
        --epochs 1 \
        --devices cuda:7 \
        --resolution ${resolution} \
        > "${folder}/probing_eval.log" 2>&1 
done