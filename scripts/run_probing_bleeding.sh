# >>> conda initialize >>>
__conda_setup="$('/home/user01/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/user01/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/user01/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/user01/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

source /home/user01/.bashrc
conda deactivate
conda activate NSJepa

CUDA_VISIBLE_DEVICES=6
TASK="foundation_model_probing"
FNAME="bleeding_probe_attentive_clip_70_30.yaml"

LOG_DIR="logs/foundation/bleeding_V5_5fps"
mkdir -p "$LOG_DIR"
TIME=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${FNAME%.yaml}_${TIME}.log"

python -m evals.main \
    --fname "configs/${TASK}/vjepa/${FNAME}" \
    --devices ${CUDA_VISIBLE_DEVICES} \
    > "$LOG_FILE" 2>&1