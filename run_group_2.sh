#! /bin/bash

# configs=(
#         "vitl_cooldown-256px-32f_mask_small"
#     "vitl_cooldown-256px-32f_mask_large"
#     "vitl_cooldown-256px-32f_complement_masking"
# )

# for config in "${configs[@]}"; do
#     MASTER_PORT=12333 python -m app.main --fname configs/pitvis_pretrain/${config}.yaml --devices cuda:2 cuda:3
# done


### eval pretrained vitl 
ckpt_dirs=(
"logs/cpt_vitl16-256px-32f_cooldown"
"logs/cpt_vitl16-256px-32f_cooldown_complement_masking"
"logs/cpt_vitl16-256px-32f_cooldown_lr1e-4"
"logs/cpt_vitl16-256px-32f_cooldown_lr2e-5"
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
        --checkpoint ${ckpt_dir}/latest.pt \
        --folder ${folder} \
        --override_config_folder \
        --frames_per_clip 32 \
        --batch_size 16 \
        --epochs 1 \
        --devices cuda:1 \
        --resolution ${resolution} \
        > "${folder}/probing_eval.log" 2>&1 
done