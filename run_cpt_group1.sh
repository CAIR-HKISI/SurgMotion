#! /bin/bash

configs=(
    "vitl_cooldown-256px-32f_lr2e-4"
    # "vitl_cooldown-256px-32f_lr4e-4"
    # "vitl_cooldown-256px-32f_lr5e-4"
)

for config in "${configs[@]}"; do
    MASTER_PORT=$((RANDOM % 55536 + 10000))
    folder="logs/cpt_${config}"
    mkdir -p "${folder}"
    MASTER_PORT=${MASTER_PORT} nohup python -m app.main \
                                --fname configs/pitvis_pretrain/${config}.yaml \
                                --devices cuda:0 cuda:1 \
                                > "${folder}/cpt_train.log" 2>&1 
done


#