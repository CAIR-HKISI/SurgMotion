export HF_ENDPOINT="https://hf-mirror.com"

Model_name="facebook/-fpc64-256"
output_dir="./logs/r50_distill_32frames_10epochs"
mkdir -p ${output_dir}


# nohup 
python evals/surgical_video_full_finetune/tecno_vid_distill_r50.py \
  --data_dir              /data/wjl/NeuroMAE/data/pitvis \
  --train_csv             train_metadata.csv \
  --val_csv               val_metadata.csv \
  --output_dir            ${output_dir} \
  --pretrained_model_path  'ckpts/model.ckpt'\
  --image_size            256 \
  --max_epochs            10 \
  --train_bs              32 \
  --eval_bs               32 \
  --lr                    1e-5 \
  --weight_decay          0.0 \
  --limit_train_batches   1.0 \
  --limit_val_batches     1.0 \
  --class_weight          median \
  --fp16 \
  --gradient_checkpointing \
  --gpus                  "1" 

