#! /bin/bash
#SBATCH -o SLURM_Logs_ivt/%j.out # Standard output log
#SBATCH -e SLURM_Logs_ivt/%j.err # Standard error log
#SBATCH -J cholect50_triplet_probing # Job name: CholecT50 Triplet Action Conditioning
#SBATCH --partition=a100 # Partition name debug 
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks-per-node=1 # Number of tasks (1 task = 1 GPU)
#SBATCH --gres=gpu:1 # Number of GPUs per node
#SBATCH --cpus-per-task=12 # CPUs per task
#SBATCH --mem=128G # Memory per node
# SBATCH --exclude=compute-a100-03 # SBATCH --nodelist=compute-a100-04

source /home/yaxin_hu/projects/NSJepa-triplet_recognition/.venv/bin/activate
nvidia-smi
wandb on

FNAME="cholect50_com_multiepoch.yaml" ### modify
TASK="triplets"
MASTER_PORT=13280  # "1389"
DEVICES="cuda:0"


TIME=$(date +"%Y%m%d_%H%M")


MASTER_PORT=${MASTER_PORT} \
nohup \
python -m main_com_multiepoch \
  --fname "/home/yaxin_hu/projects/NSJepa-triplet_recognition/configs/${TASK}/${FNAME}" \
  --devices ${DEVICES} \
  # --use_fsdp
