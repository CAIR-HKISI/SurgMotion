#! /bin/bash
#SBATCH -o SLURM_Logs_gastronet/%j.out # Standard output log
#SBATCH -e SLURM_Logs_gastronet/%j.err # Standard error log
#SBATCH -J cholect50_triplet_probing # Job name: CholecT50 Triplet Action Conditioning
#SBATCH --partition=a100 # Partition name debug 
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks-per-node=1 # Number of tasks (1 task = 1 GPU)
#SBATCH --gres=gpu:1 # Number of GPUs per node
#SBATCH --cpus-per-task=12 # CPUs per task
#SBATCH --mem=128G # Memory per node
#SBATCH --exclude=compute-a100-03 # SBATCH --nodelist=compute-a100-04

source /home/yaxin_hu/projects/NSJepa-triplet_recognition/.venv/bin/activate
nvidia-smi
wandb online

FNAME="gastronet_vitl_cholect50_ivt.yaml"
TASK="triplets"
MASTER_PORT=13280  # "1389"
DEVICES="cuda:0"
# python -m main \
#   --fname "configs/${TASK}/${FNAME}" \
#   --devices ${CUDA_VISIBLE_DEVICES}

# Set NCCL environment variables
# export NCCL_DEBUG=INFO          # For debugging
# export NCCL_IB_DISABLE=1        # If you encounter issues with InfiniBand
# export NCCL_P2P_LEVEL=SYS       # Optimize point-to-point communication

TIME=$(date +"%Y%m%d_%H%M")

# torchrun --nproc_per_node 2 --master_port=1389 main_ivt.py

MASTER_PORT=${MASTER_PORT} \
nohup \
python -m main_ivt_multiepoch \
  --fname "/home/yaxin_hu/projects/NSJepa-triplet_recognition/configs/${TASK}/${FNAME}" \
  --devices ${DEVICES} \
  # --use_fsdp
  # > "${LOG_FILE}" 2>&1 &

  # usage: main_ivt.py [-h] [--val_only] [--fname FNAME]
  #                  [--devices DEVICES [DEVICES ...]] [--debugmode DEBUGMODE]
  #                  [--folder FOLDER] [--override_config_folder]
  #                  [--checkpoint CHECKPOINT] [--model_name MODEL_NAME]
  #                  [--batch_size BATCH_SIZE] [--resolution RESOLUTION]
  #                  [--frames_per_clip FRAMES_PER_CLIP] [--epochs EPOCHS]
  #                  [--use_fsdp]
