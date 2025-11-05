#!/bin/bash
#SBATCH -o SLURM_Logs/%j.out # Standard output log
#SBATCH -e SLURM_Logs/%j.err # Standard error log
#SBATCH -J surgaction160 # Job name
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

FNAME="surgaction160_vitl_cpt_attentive_64f.yaml"
TASK="probing_surgaction160"
python -m evals.main \
  --fname "configs/${TASK}/${FNAME}" \
  --devices ${CUDA_VISIBLE_DEVICES} \

