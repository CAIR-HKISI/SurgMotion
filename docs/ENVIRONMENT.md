# SurgMotion 环境配置指南

面向快速上手：分三步安装。**目标脚本**：`scripts/run_foundation_probing_bleeding.sh`、`scripts/run_probing.sh`（均调用 `python -m evals.main`）。

## 0. 前置条件

- **GPU**：CUDA 与 PyTorch 版本需匹配（按你机器选择 [PyTorch 官网](https://pytorch.org) 的安装命令）。
- **仓库与子模块**（基金会模型代码在子模块中）：

```bash
git clone <你的仓库 URL> SurgMotion
cd SurgMotion
git submodule update --init --recursive
```

- **权重与数据**：将各 YAML 中 `checkpoint` / 数据路径指向本地；`ckpts/` 通常较大，勿提交到 Git。
- **Conda 环境名**：示例脚本里使用 `conda activate NSJepa`，请改为你自己的环境名（如 `surgmotion`）。

---

## 1. 仅 V-JEPA2 核心依赖

用于：`src/` 视频模型与数据管线、以及依赖相同栈的轻量评测（**不包含** `foundation_phase_probing` 里 HuggingFace / 指标等）。

```bash
conda create -n surgmotion-vjepa python=3.11 -y
conda activate surgmotion-vjepa
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # 按你的 CUDA 修改
pip install -r requirements-vjepa2.txt
pip install -e .   # 可选：以可编辑方式安装包 vjepa2
```

依赖文件：`requirements-vjepa2.txt`（`torch` / `torchvision` / `decord` / `timm` / `einops` / `opencv-python` 等）。

---

## 2. 多基金会模型 Probing（不含 EndoMamba）

用于：`scripts/run_foundation_probing_bleeding.sh` 中除 EndoMamba 外的任务（GastroNet、DINOv3、EndoFM、EndoViT、EndoSSL、GSViT、SurgVLP、VideoMAEv2、SurgeNet、SelfSupSurg 等），以及同一环境下的 `run_probing.sh`。

```bash
conda create -n surgmotion python=3.11 -y
conda activate surgmotion
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # 按你的 CUDA 修改
pip install -r requirements-foundation.txt
# 或等价：pip install -r requirements.txt
pip install -e .
```

依赖说明：

| 类别 | 包 |
|------|----|
| 核心 | 见 `requirements-vjepa2.txt` |
| HF / DINOv3 / VideoMAE | `transformers`, `accelerate`, `peft`, `huggingface_hub` |
| 训练评测常用 | `pandas`, `scipy`, `scikit-learn`, `wandb`, `tensorboard` |
| 配置 / 工具 | `fvcore`, `omegaconf`, `iopath`, `beartype` |
| 集群 | `submitit`（若使用 `evals/main_distributed.py`） |

**WandB**：离线可设 `export WANDB_MODE=offline`。

**子模块**：`GSViT`、`EndoViT`、`Endo-FM`、`InternVideo`、`SurgVLP`、`SurgeNet` 等需在 `foundation_models/` 就绪；若某模型缺仓库，对应 YAML 会导入失败。

---

## 3. EndoMamba 独立环境（编译 CUDA 扩展）

EndoMamba 与主环境 **分开**：脚本使用 **Python 3.10**、**CUDA 12.1（conda）**、**PyTorch 2.7.0 + cu128**，并**编译** `causal-conv1d` 与 `mamba`。

**推荐直接执行（与仓库脚本一致）：**

```bash
cd /path/to/SurgMotion
bash scripts/srun_endomamba_complie.sh
```

脚本要点摘要（详见 `scripts/srun_endomamba_complie.sh`）：

1. `conda create -n endomamba python=3.10`
2. `conda install -c nvidia cuda-toolkit=12.1`，并设置 `CUDA_HOME=$CONDA_PREFIX`
3. `pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128`
4. 安装一组 pip 依赖（与 `requirements-endomamba.txt` 类似；**以脚本中的 `pip install ...` 行为准**）
5. 在 `foundation_models/EndoMamba` 下编译安装 `causal-conv1d` 与 `videomamba/_mamba`

运行 Bleeding 中的 EndoMamba 任务时：使用 `endomamba` 环境，并在 `run_foundation_probing_bleeding.sh` 中取消注释 `endomamba/Bleeding` 与 `conda activate endomamba`（注释掉主环境）。

---

## 验证安装

```bash
cd /path/to/SurgMotion
python -c "import torch; import decord; import timm; print('ok', torch.__version__)"

# Foundation 环境再测
python -c "import transformers, scipy, sklearn; print('foundation ok')"
```

**冒烟运行（需有效 config 与数据路径）：**

```bash
python -m evals.main --fname configs/foundation_model_probing/gastronet/Bleeding/gastronet_vits_clip_bleeding.yaml --devices cuda:0
```

---

## 已知仓库注意项（与 pip 无关）

1. **`evals/foundation_phase_probing/eval.py`** 依赖 `evals.surgical_video_classification_frozen` 与 `evals.video_classification_frozen`。若本仓库中缺少这些包，需在合并上游或补全模块后脚本才能运行。
2. **`scripts/run_probing.sh`** 中部分 `configs/${TASK}/...` 路径可能与当前 `configs/` 目录布局不一致；请改为存在的 YAML（例如 `configs/foundation_model_probing/...`）或从上游同步配置。

---

## 可选：开发/Notebook

```bash
pip install jupyter ipykernel
```

（不再放入默认 `requirements.txt`，避免在生产镜像中安装 Notebook。）
