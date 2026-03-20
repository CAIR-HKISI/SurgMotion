# SurgMotion

## 环境安装

完整分步说明见 **[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)**：

| 阶段 | 说明 |
|------|------|
| **1. V-JEPA2 核心** | `pip install -r requirements-vjepa2.txt`（仅 `src/` 与视频管线） |
| **2. Foundation probing（主环境）** | `pip install -r requirements.txt`（等同 `requirements-foundation.txt`） |
| **3. EndoMamba** | **独立 Conda 环境**，见 `scripts/srun_endomamba_complie.sh`，勿与主环境混装 |

- **Python**：`setup.py` 要求 `>=3.10`；主 probing 环境推荐 **3.11**。EndoMamba 脚本使用 **3.10**。
- **PyTorch**：须与机器 **CUDA** 版本匹配，请从 [pytorch.org](https://pytorch.org) 选择安装命令后再装依赖文件。

### 快速安装主环境（不含 EndoMamba）

```bash
conda create -n SurgMotion python=3.11 -y
conda activate SurgMotion
pip install torch torchvision   # 按 CUDA 从官网选择 wheel 源
pip install -r requirements.txt
pip install -e .
```

### EndoMamba（单独执行）

```bash
bash scripts/srun_endomamba_complie.sh   # 创建 endomamba 环境并编译扩展
conda activate endomamba                 # 仅跑 EndoMamba 相关配置时使用
```

参考列表：`requirements-endomamba.txt`（**以编译脚本内的 `pip install` 为准**）。

### 依赖文件一览

| 文件 | 用途 |
|------|------|
| `requirements-vjepa2.txt` | 仅核心框架与视频管线 |
| `requirements-foundation.txt` | 核心 + 多模型 probing |
| `requirements.txt` | 默认：`-r requirements-foundation.txt` |
| `requirements-endomamba.txt` | EndoMamba 参考依赖（编译见脚本） |

---

## 常用脚本

| 脚本 | 说明 |
|------|------|
| `scripts/run_foundation_probing.sh` | 多 GPU 批量 foundation probing（在脚本内配置 `TASKS` / `FNAMES`） |
| `scripts/run_probing.sh` | 单任务 `python -m evals.main`（请确认 `--fname` 指向存在的 YAML） |
| `scripts/srun_endomamba_complie.sh` | EndoMamba 独立环境 + CUDA 扩展编译 |

脚本中的 `conda activate SurgMotion` 请改为你的主环境名。WandB 可设 `export WANDB_MODE=offline`。

---

## 配置与扩展

- **评测入口**：`python -m evals.main --fname <config.yaml> --devices cuda:0`
- **Foundation probing 配置与加模型**：见 **[evals/foundation_phase_probing/README.md](evals/foundation_phase_probing/README.md)**
- **YAML 根目录**：`configs/foundation_model_probing/...`
- **权重与数据**：在 YAML 中指向本地路径；`ckpts/`、`data/` 等大文件勿提交（见 `.gitignore`）