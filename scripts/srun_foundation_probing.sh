#!/bin/bash
# quick_test_probing.sh - 用于快速测试probing任务

# >>> conda initialize >>>
__conda_setup="$('/home/chen_chuxi/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/chen_chuxi/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/chen_chuxi/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/chen_chuxi/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

source /home/chen_chuxi/.bashrc
# conda init
conda deactivate
conda activate nsjepa
cd /home/chen_chuxi/NSJepa

# 配置参数
export WANDB_API_KEY="0db5119bb774c16d107977e6bdcfc4954b0cd514"
WANDB_API_KEY="0db5119bb774c16d107977e6bdcfc4954b0cd514"
TASK="fdtn_probing/dinov3/probing_cholec80"
FNAME="dinov3_vitl_64f_probing_cholec80.yaml"
TIME=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/foundation/${FNAME}_${TIME}.log"

echo "========================================"
echo "           Automatic GPU Test"
echo "========================================"
echo "Time: $(date)"
echo "Config: ${FNAME}"
echo "Log: ${LOG_FILE}"
echo ""
echo "📋 Checking GPU availability..."
sinfo -p gpu -o "%20P %5a %10l %5D %5c %10G" || echo "Note: GPU partition info not available"
echo ""
echo "⏳ Requesting GPU resources (this may take a moment)..."
echo "========================================"

# 自动请求GPU并运行
# --gres=gpu:2: 请求2块GPU
# --mem=64G: 64GB内存
# --cpus-per-task=8: 8个CPU核心
# --time=02:00:00: 最多运行2小时
# --partition=gpu: 使用GPU分区（根据你的集群调整）
# --pty: 实时显示输出

srun --gres=gpu:1 \
     --partition=a100 \
     --cpus-per-task=8 \
     --job-name=probing_endovit_vitl \
     bash -c "
         echo '✅ GPU resources allocated!'
         echo 'Job ID: \$SLURM_JOB_ID'
         echo 'Node: \$SLURMD_NODENAME'
         echo 'GPUs: \$CUDA_VISIBLE_DEVICES'
         echo ''
         nvidia-smi --query-gpu=index,name,memory.total --format=csv
         echo ''
         echo '🏃 Starting probing test...'
         echo '========================================'
         
         python -m evals.main \
           --fname configs/${TASK}/${FNAME} \
           --devices cuda:0 cuda:1 \
           --debugmode True
     " 2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Test completed successfully!"
else
    echo "❌ Test failed with exit code: $EXIT_CODE"
fi
echo "📄 Full log: ${LOG_FILE}"
echo "Finished at: $(date)"
echo "========================================"

exit $EXIT_CODE