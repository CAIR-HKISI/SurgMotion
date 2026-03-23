<div align="center">
<h1>SurgMotion: A Foundation Model Probing Framework for Surgical Video Understanding</h1>

<a href="https://github.com/KimWu1994/SurgMotion"><img src='https://img.shields.io/badge/GitHub-Repository-blue' alt='GitHub'></a>

</div>

Built on top of [V-JEPA 2](https://github.com/facebookresearch/vjepa2), **SurgMotion** provides a unified framework to benchmark and probe multiple surgical video foundation models on phase recognition tasks across diverse surgical datasets.

![Framework](assets/flowchart.png)

## Quick Start

- **Setup**:
  - [Environment Installation](#environment-installation)
  - [Data Preparation](#data-preparation)
- **Usage**:
  - [Run Foundation Probing](#run-foundation-probing)
  - [Evaluation Metrics](#evaluation-metrics)
- **Extend**:
  - [Add a New Dataset](#add-a-new-dataset)
  - [Add a New Foundation Model](#add-a-new-foundation-model)

## Project Structure

```
SurgMotion/
├── src/                        # V-JEPA2 core: ViT, VideoMAE, datasets, masks
├── evals/                      # Evaluation entry points & foundation phase probing
│   ├── main.py                 # Single-task entry: python -m evals.main --fname <yaml>
│   └── foundation_phase_probing/
│       ├── eval.py             # Probing evaluation logic
│       ├── models.py           # Probing head definitions
│       └── modelcustom/        # Per-model adapters (DINOv2, EndoViT, SurgVLP, …)
├── configs/
│   └── foundation_model_probing/
│       ├── dinov2/             # YAML configs per dataset
│       ├── dinov3/
│       ├── endofm/
│       ├── …                   # 15 model families supported
│       └── videomaev2/
├── data_process/               # End-to-end dataset preprocessing scripts
│   ├── autolaparo_prepare.py
│   ├── cholect80_prepare.py
│   ├── egosurgery_prepare.py
│   ├── m2cai2016_prepare.py
│   ├── ophnet_prepare.py
│   ├── pitvis_prepare.py
│   ├── pmlr50_prepare.py
│   ├── polypdiag_prepare.py
│   └── surgicalactions160_prepare.py
├── scripts/                    # Batch probing & environment setup shells
├── foundation_models/          # Third-party model implementations (git submodules)
├── data/                       # Data directory
├── setup.py                    # pip install -e .
└── requirements*.txt           # Layered dependency files
```

## Environment Installation

### Main Environment (Recommended)

```bash
conda create -n SurgMotion python=3.12 -y
conda activate SurgMotion

# Install PyTorch matching your CUDA version first:
# https://pytorch.org/get-started/locally/

pip install -e .
```

### EndoMamba (Separate Environment)

EndoMamba requires its own Conda env with custom CUDA extensions. **Do not mix** with the main environment.

```bash
bash scripts/srun_endomamba_complie.sh   # Creates env + compiles extensions
conda activate endomamba                 # Use only for EndoMamba configs
```

### Dependency Files

| File | Scope |
|------|-------|
| `requirements-vjepa2.txt` | V-JEPA2 core only (video pipeline, `src/`) |
| `requirements-foundation.txt` | Core + multi-model probing dependencies |
| `requirements.txt` | Default: includes `requirements-foundation.txt` |
| `requirements-endomamba.txt` | EndoMamba reference (use compile script instead) |

## Data Preparation

All preprocessing scripts under `data_process/` follow a unified end-to-end pipeline:

```bash
python data_process/<dataset>_prepare.py [OPTIONS]
```

Each script supports `--help` and produces:
- `clip_infos/*.txt` — per-case frame path lists for clip-style loaders
- `{train,val,test}_metadata.csv` — standardized CSV with columns:

| Column | Description |
|--------|-------------|
| `Index` | Row index within the split (0-based) |
| `clip_path` | Path to the clip's frame list txt |
| `label` | Integer phase / class id |
| `label_name` | Human-readable phase name |
| `case_id` | Numeric case / video identifier |
| `clip_idx` | Clip index within the case (0 for single-clip) |

### Supported Datasets

| Dataset | Script | Domain | Phases |
|---------|--------|--------|--------|
| AutoLaparo | `autolaparo_prepare.py` | Laparoscopic hysterectomy | 7 |
| Cholec80 | `cholect80_prepare.py` | Laparoscopic cholecystectomy | 7 |
| EgoSurgery | `egosurgery_prepare.py` | Egocentric open surgery | 9 |
| M2CAI2016 | `m2cai2016_prepare.py` | Laparoscopic cholecystectomy | 8 |
| OphNet2024 | `ophnet_prepare.py` | Ophthalmic surgery | 96 |
| PitVis | `pitvis_prepare.py` | Pituitary neurosurgery | 12 |
| PmLR50 | `pmlr50_prepare.py` | Laparoscopic liver resection | 5 |
| PolypDiag | `polypdiag_prepare.py` | GI endoscopy (binary) | 2 |
| SurgicalActions160 | `surgicalactions160_prepare.py` | Surgical action recognition | N (auto) |

### Example: Prepare Cholec80

```bash
python data_process/cholect80_prepare.py \
    --frames_root data/Surge_Frames/Cholec80/frames \
    --annot_dir data/Landscopy/cholec80/phase_annotations \
    --output_dir data/Surge_Frames/Cholec80 \
    --debug
```

After preprocessing, use `gen_clips.py` (if needed) to create sliding-window clips for dense training:

```bash
python data_process/gen_clips.py \
    --input_csv data/Surge_Frames/Cholec80/train_metadata.csv \
    --output_dir data/Surge_Frames/Cholec80/clips_64f \
    --window_size 64 --stride 1
```

## Run Foundation Probing

### Single Task

```bash
python -m evals.main \
    --fname configs/foundation_model_probing/dinov3/AutoLaparo/dinov3_vitl_64f_AutoLaparo.yaml \
    --devices cuda:0
```

### Batch (Multi-GPU Parallel)

Edit the task list in `scripts/run_foundation_probing.sh`, then run:

```bash
bash scripts/run_foundation_probing.sh
```

The script auto-assigns one GPU per task from the available pool (default: all 8 GPUs). Logs are saved under `logs/foundation/<Dataset>/`.

### Supported Foundation Models

| Model | Identifier | Architecture |
|-------|-----------|--------------|
| DINOv3 | `dinov3` | ViT-L |
| EndoFM | `endofm` | ViT-B |
| EndoMamba | `endomamba` | Mamba-S |
| EndoSSL | `endossl` | ViT-L |
| EndoViT | `endovit` | ViT-L |
| GastroNet | `gastronet` | ViT-S |
| GSViT | `gsvit` | ViT |
| SelfSupSurg | `selfsupsurg` | ResNet-50 |
| SurgeNet | `surgenet` | CAFormer-XL |
| SurgVLP | `surgvlp` | ResNet-50 |
| VideoMAEv2 | `videomaev2` | ViT-L |

## Add a New Dataset

1. Create `data_process/<dataset>_prepare.py` following the existing template (see `polypdiag_prepare.py` for reference).
2. Output standardized CSVs with the 6-column schema (`Index`, `clip_path`, `label`, `label_name`, `case_id`, `clip_idx`).
3. Create YAML configs under `configs/foundation_model_probing/<model>/<Dataset>/`.

## Add a New Foundation Model

1. Write an adapter under `evals/foundation_phase_probing/modelcustom/adapters/`:

```python
# Input:  any shape, e.g. [B, C, F, H, W]
# Output: [B, F*N, D]  (spatial-temporal tokens)
```

2. Register the model in `evals/foundation_phase_probing/modelcustom/foundation_model_wrapper.py`:

```python
elif model_type == 'your_model':
    from .adapters.your_model_adapter import YourModelAdapter
    adapter = YourModelAdapter.from_config(
        resolution=resolution,
        checkpoint=checkpoint,
        model_name=model_name
    )
```

3. Create YAML configs under `configs/foundation_model_probing/your_model/<Dataset>/`.
4. Add entries to `scripts/run_foundation_probing.sh` and run.

## Acknowledgement

We thank [V-JEPA 2 (Meta)](https://github.com/facebookresearch/vjepa2) for the base framework, and [MONAI](https://github.com/Project-MONAI/research-contributions) for reference implementations.
