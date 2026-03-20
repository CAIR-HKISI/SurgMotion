python -m torch.distributed.launch --nproc_per_node=4 --master_port 29210 --use_env downstream_phase/run_phase_training.py \
    --gpu_id '0, 3, 4, 5' \
    --batch_size 24 \
    --epochs 50 \
    --save_ckpt_freq 10 \
    --model endomamba_small \
    --mixup 0.8 \
    --cutmix 1.0 \
    --smoothing 0.1 \
    --lr 5e-5 \
    --layer_decay 0.85 \
    --warmup_epochs 5 \
    --data_path /mnt/tqy/AutoLaparo/AutoLaparo_Task1 \
    --eval_data_path /mnt/tqy/AutoLaparo/AutoLaparo_Task1 \
    --nb_classes 7 \
    --data_strategy online \
    --output_mode all_frame \
    --num_frames 32 \
    --sampling_rate -1 \
    --data_set AutoLaparo \
    --data_fps 1fps \
    --output_dir /mnt/tqy/out/AutoLaparo/ \
    --log_dir /mnt/tqy/out/AutoLaparo/ \
    --num_workers 10 \
    --dist_eval \
    --no_auto_resume \
    --no_pin_mem \
    --seed 4 \
    # --only_cls_token \
    # --aa False \
