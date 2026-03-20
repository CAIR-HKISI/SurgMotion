# SurgMotion

基于 [V-JEPA2](https://github.com/facebookresearch/vjepa2) 的手术视频表征与下游 probing 实验代码库。

## 环境安装（新开发者必读）

分三步：**V-JEPA2 核心** → **多基金会模型 probing（不含 EndoMamba）** → **EndoMamba 独立编译环境**。详见 **[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)**。

快速安装主环境（含 Bleeding 批量脚本所需依赖，不含 EndoMamba）：

```bash
pip install torch torchvision  # 按 CUDA 版本从 pytorch.org 选择命令
pip install -r requirements.txt
pip install -e .
```

依赖文件：

| 文件 | 用途 |
|------|------|
| `requirements-vjepa2.txt` | 仅核心框架与视频管线 |
| `requirements-foundation.txt` | 核心 + 多模型 probing |
| `requirements.txt` | 默认等同 foundation 完整栈 |
| `requirements-endomamba.txt` | EndoMamba 参考列表（**以 `scripts/srun_endomamba_complie.sh` 为准**） |

## 常用脚本

- `scripts/run_foundation_probing_bleeding.sh` — 多 GPU 批量 foundation probing（Bleeding）
- `scripts/run_probing.sh` — 单任务 `evals.main`（请确认 `configs/...` 路径与仓库一致）

## 许可证

见上游 V-JEPA2 与各 `foundation_models` 子模块仓库说明。
