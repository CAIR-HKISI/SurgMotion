python ./downstream_phase/test.py \
    --data_path /data/tqy/AutoLaparo/AutoLaparo_Task1 \
    --pretrained_path /data/tqy/endomamba_pretrain/downstream/SurgicalPhase/EndoMamba/checkpoint-19.pth\
    --device cuda:2 \
    --train_seq_len 32 \
    --num_frames 32 \
    --model EndoMamba \
    --output_mode key_frame \
    # --only_cls_token \
    # --eval_video_id 21
