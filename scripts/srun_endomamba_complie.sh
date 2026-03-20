set -e

echo "============================================"
echo "EndoMamba Compile"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "============================================"

# >>> conda initialize >>>
CONDA_PATH="${CONDA_PATH:-$HOME/miniconda3}"
if [ -x "${CONDA_PATH}/bin/conda" ]; then
    __conda_setup="$("${CONDA_PATH}/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        [ -f "${CONDA_PATH}/etc/profile.d/conda.sh" ] && . "${CONDA_PATH}/etc/profile.d/conda.sh"
    fi
fi
unset __conda_setup 2>/dev/null
# <<< conda initialize <<<

source ~/.bashrc

conda deactivate 2>/dev/null || true
conda env remove -n endomamba -y 2>/dev/null || true
conda create -n endomamba python=3.10 -y
conda activate endomamba
conda install -c nvidia cuda-toolkit=12.1 -y

# 设置 CUDA_HOME 为 conda 环境中的 CUDA
export CUDA_HOME=$CONDA_PREFIX
export PATH="$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH}"

echo "CUDA_HOME: $CUDA_HOME"

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install fvcore pandas threadpoolctl platformdirs google click wandb iopath timm submitit opencv-python transformers peft einops beartype psutil h5py fire python-box scikit-image ftfy scikit-learn omegaconf accelerate scipy joblib --no-deps

# ---------- 编译 causal-conv1d ----------
echo ""
echo "Building causal-conv1d..."
echo "CUDA_HOME: $CUDA_HOME"
echo "nvcc: $(which nvcc)"

cd /home/user01/NSJepa/foundation_models/EndoMamba
pip install ./videomamba/causal-conv1d --no-build-isolation --force-reinstall --no-deps
cd ./videomamba/_mamba
export MAMBA_FORCE_BUILD=TRUE
python setup.py clean
rm -rf build/ *.egg-info
pip install . --no-build-isolation --verbose --no-deps

echo ""
echo "✓ causal-conv1d installed successfully"

# ---------- 测试 EndoMamba adapter ----------
echo ""
echo "Testing EndoMamba adapter..."
cd /home/user01/NSJepa/foundation_models
python evals/foundation_phase_probing/modelcustom/adapters/endomamba_adapter.py

echo ""
echo "============================================"
echo "EndoMamba compilation complete!"
echo "============================================"
