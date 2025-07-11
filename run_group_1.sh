#! /bin/bash

### continue pretrain
# configs=(
#     "vitl_cooldown-256px-32f_lr2e-5"
#     "vitl_cooldown-256px-32f_lr5e-5"
#     "vitl_cooldown-256px-32f_lr1e-4"
#     "vitl_cooldown-256px-32f_resolution_384"
# )

# for config in "${configs[@]}"; do
#     MASTER_PORT=12315 python -m app.main --fname configs/pitvis_pretrain/${config}.yaml --devices cuda:0 cuda:1 
# done

### eval vitl origin 
# frames_per_clip = (16, 32, 64)

# 每个循环随机生成一个新的端口号
MASTER_PORT=$((RANDOM % 55536 + 10000))
frames_per_clip=(32 64 16)
for frame in "${frames_per_clip[@]}"; do
    
    # 定义日志目录
    folder="logs/vitl_origin_probing_eval_${frame}frames"
    
    # 创建日志目录（如果不存在）
    mkdir -p "${folder}"
    
    echo "Running evaluation for ${frame} frames on port ${MASTER_PORT}"

    # 后台运行并将输出记录到日志文件
    MASTER_PORT=${MASTER_PORT} nohup python -m evals.main \
        --fname configs/pitvis_eval/vitl_origin.yaml \
        --checkpoint /data/wjl/vjepa2/ckpts/vitl.pt \
        --folder ${folder} \
        --override_config_folder \
        --frames_per_clip ${frame} \
        --batch_size 16 \
        --devices cuda:0 \
        > "${folder}/probing_eval.log" 2>&1 
done


# ## eval vith origin 
# frames_per_clip = (16, 32, 64)
# for frame in "${frames_per_clip[@]}"; do
#     MASTER_PORT=12311 python -m eval.main --fname configs/pitvis_eval/vith_origin.yaml \
#                                           --checkpoint ${ckpt_dir}/latest.pt \
#                                           --folder ${ckpt_dir}/probing_eval/ \
#                                           --devices cuda:0\
# done


### eval pretrained vitl 
# ckpt_dirs = (
# "logs/cpt_vitl16-256px-32f_cooldown",
# "logs/cpt_vitl16-256px-32f_cooldown_complement_masking",
# "logs/cpt_vitl16-256px-32f_cooldown_lr1e-4",
# "logs/cpt_vitl16-256px-32f_cooldown_lr2e-5",
# "logs/cpt_vitl16-256px-32f_cooldown_lr5e-5",
# "logs/cpt_vitl16-256px-32f_cooldown_mask_large",
# "logs/cpt_vitl16-256px-32f_cooldown_mask_small",
# "logs/cpt_vitl16-256px-32f_cooldown_resolution_384",
# "logs/cpt_vitl16-256px-32f_cooldown_resolution_448" 
# )

# for ckpt_dir in "${ckpt_dirs[@]}"; do
#     MASTER_PORT=12311 python -m eval.main --fname configs/pitvis_eval/vitl_pretrain_pitvis.yaml \
#                                           --checkpoint ${ckpt_dir}/latest.pt \
#                                           --folder ${ckpt_dir}/probing_eval/ \
#                                           --devices cuda:0\
# done

