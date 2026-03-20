python train.py \
    --exp EndoMamba_MIX12_teacher \
    --gpu 3 \
    --batch_size 1 \
    --model endomambaseg_small \
    --root_path /path/to/CVC-ClinicVideoDB/ \
    --seed 11 \
    --n_skip 0\
    --base_lr 1e-4 \
    --out_dir /path/to/save/checkpoints/
    # --wandb False