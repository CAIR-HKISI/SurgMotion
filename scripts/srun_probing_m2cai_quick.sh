#!/bin/bash
#SBATCH -o SLURM_Logs/%j.out # Standard output log
#SBATCH -e SLURM_Logs/%j.err # Standard error log
#SBATCH -J vjepa_quick # Job name
#SBATCH --partition=a100 # Partition name
#SBATCH --nodes=1 # Number of nodes
#SBATCH --gres=gpu:1 # Number of GPUs per node

# >>> conda initialize >>>
__conda_setup="$('/home/felix_holm/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/felix_holm/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/felix_holm/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/felix_holm/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

source /home/felix_holm/.bashrc
# conda init
conda deactivate
conda activate jepa_torch

# Quick run config for fast debugging (1 video only)
FNAME="m2cai_vitl_cpt_quick_run.yaml"
TASK="probing_m2cai"

echo "========================================="
echo "QUICK RUN MODE - DEBUG ONLY"
echo "========================================="
echo "Config: ${FNAME}"
echo "Using 1 video only for fast iteration"
echo "Bootstrap disabled for speed"
echo "Reduced number of heads and epochs"
echo "========================================="

python -m evals.main \
  --fname "configs/${TASK}/${FNAME}" \
  --devices ${CUDA_VISIBLE_DEVICES}

echo "========================================="
echo "Quick run completed!"
echo "Check wandb for results"
echo "========================================="
