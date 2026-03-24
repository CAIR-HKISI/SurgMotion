---
license: apache-2.0
tags:
  - surgical-video
  - foundation-model
  - video-understanding
  - phase-recognition
  - v-jepa
  - vision-transformer
datasets:
  - SurgMotion-15M
language:
  - en
pipeline_tag: video-classification
---
<div align="center">
<h1>SurgMotion</h1>
<h3>A Video-Native Foundation Model for Universal Understanding of Surgical Videos</h3>

<a href="https://surgmotion.cares-copilot.com/"><img src='https://img.shields.io/badge/Project-Homepage-0A66C2' alt='Project Page'></a>
<a href="https://arxiv.org/abs/2602.05638"><img src='https://img.shields.io/badge/arXiv-2602.05638-b31b1b' alt='arXiv'></a>
<a href="https://github.com/CAIR-HKISI/SurgMotion"><img src='https://img.shields.io/badge/GitHub-Code-blue' alt='GitHub'></a>

</div>

## Model Summary

**SurgMotion** is a video-native surgical foundation model that learns spatiotemporal representations by predicting latent motion rather than reconstructing pixels. Built upon the Video Joint Embedding Predictive Architecture ([V-JEPA 2](https://github.com/facebookresearch/vjepa2)), SurgMotion captures the complex temporal dynamics of surgical procedures without the computational overhead of generative decoding.

Key innovations:
- **Latent motion prediction** — shifts from pixel-level reconstruction to abstract motion forecasting in latent space
- **Flow-Guided Latent Prediction** — a novel objective that prevents feature collapse in homogeneous surgical tissue regions
- **Pre-trained on SurgMotion-15M** — the largest multi-modal surgical video dataset to date (15M frames, 3,658 hours, 13+ anatomical regions)

## Model Variants

| Variant | Backbone | Parameters | Pre-training Data |
|---------|----------|------------|-------------------|
| SurgMotion-L | ViT-Large | 300M | SurgMotion-15M |
| SurgMotion-H | ViT-Huge | 600M | SurgMotion-15M |
| SurgMotion-G | ViT-Giant | 1.01B | SurgMotion-15M |

## Pre-training Data: SurgMotion-15M

| Statistic | Value |
|-----------|-------|
| Total Frames | 15M+ |
| Total Duration | 3,658 hours |
| Anatomical Regions | 13+ |
| Supported Tasks | 6+ (workflow, action, segmentation, triplet, skill, depth) |

SurgMotion-15M spans a diverse range of surgical procedures including laparoscopic, endoscopic, robotic, open, endonasal, neurosurgical, and ophthalmic surgeries.

## Performance Highlights

SurgMotion achieves state-of-the-art results across multiple surgical understanding tasks.

- **Best overall** on 5 out of 6 representative surgical tasks

### Workflow Phase Recognition (Avg F1)

| Dataset | Domain | SurgMotion |
|---------|--------|------------|
| AutoLaparo | Laparoscopic hysterectomy | SOTA |
| Cholec80 | Laparoscopic cholecystectomy | SOTA |
| EgoSurgery | Egocentric open surgery | SOTA |
| M2CAI2016 | Laparoscopic cholecystectomy | SOTA |
| OphNet2024 | Ophthalmic surgery | SOTA |
| PitVis | Pituitary neurosurgery | SOTA |
| PmLR50 | Laparoscopic liver resection | SOTA |
| PolypDiag | GI endoscopy | SOTA |

### Multi-Task Performance

SurgMotion demonstrates strong generalization across diverse surgical understanding tasks beyond phase recognition:

| Task | Description | Result |
|------|-------------|--------|
| Workflow Recognition | Surgical phase identification | Best |
| Action Recognition | Fine-grained action classification | Best |
| Segmentation | Instrument/tissue segmentation | Best |
| Triplet Recognition | Instrument-verb-target triplets | Best |
| Skill Assessment | Surgical skill scoring | Best |
| Depth Estimation | Monocular depth prediction | Competitive |

## Intended Use

SurgMotion is designed for:
- **Surgical workflow analysis** — automated phase and step recognition
- **Downstream fine-tuning** — feature extraction backbone for surgical vision tasks
- **Research benchmarking** — standardized evaluation of surgical video foundation models

## How to Use

### Quick Start

```python
# Clone the repository
git clone https://github.com/CAIR-HKISI/SurgMotion.git
cd SurgMotion
pip install -e .
# Run probing evaluation on a dataset
python -m evals.main \
    --fname configs/foundation_model_probing/dinov3/AutoLaparo/dinov3_vitl_64f_autolaparo.yaml \
    --devices cuda:0
```

For detailed setup, data preparation, and usage instructions, please refer to the [GitHub repository](https://github.com/CAIR-HKISI/SurgMotion).

## Architecture

SurgMotion builds on the V-JEPA 2 framework with the following pipeline:

1. **Video Encoder (ViT)** — processes 64-frame surgical video clips into spatiotemporal token sequences
2. **Latent Predictor** — predicts masked region representations in latent space guided by optical flow
3. **Probing Head** — lightweight temporal classifier for downstream phase recognition

The model learns without pixel-level reconstruction, relying entirely on latent-space self-supervised objectives. The Flow-Guided Latent Prediction mechanism specifically addresses the challenge of homogeneous tissue regions common in surgical videos, where standard masking strategies tend to collapse.

## Training Details

| Aspect | Detail |
|--------|--------|
| Framework | V-JEPA 2 (Video Joint Embedding Predictive Architecture) |
| Objective | Flow-Guided Latent Prediction |
| Input | 64 frames per clip |
| Pre-training Scale | 3,658 hours of surgical video |
| Hardware | Multi-GPU (8x NVIDIA A100/H100) |

## Evaluated Benchmarks

SurgMotion has been evaluated on the following public surgical phase datasets:

| Dataset | Task | Domain |
|---------|------|--------|
| AutoLaparo | Phase recognition | Laparoscopic hysterectomy |
| Cholec80 | Phase recognition | Laparoscopic cholecystectomy |
| EgoSurgery | Phase recognition | Egocentric open surgery |
| M2CAI2016 | Phase recognition | Laparoscopic cholecystectomy |
| OphNet2024 | Phase recognition | Ophthalmic surgery |
| PitVis | Phase recognition | Pituitary neurosurgery |
| PmLR50 | Phase recognition | Laparoscopic liver resection |
| PolypDiag | Binary classification | GI endoscopy |
| SurgicalActions160 | Action recognition | Multi-procedure |

## Limitations

- Performance may vary on surgical domains not represented in SurgMotion-15M
- Frame-level predictions may exhibit temporal fragmentation without post-processing smoothing
- Requires GPU for inference; not optimized for edge deployment

## Citation

If you find SurgMotion useful, please cite:

```bibtex
@misc{wu2026unisurgvideonativefoundationmodel,
      title={UniSurg: A Video-Native Foundation Model for Universal Understanding of Surgical Videos}, 
      author={Jinlin Wu and Felix Holm and Chuxi Chen and An Wang and Yaxin Hu and Xiaofan Ye and Zelin Zang and Miao Xu and Lihua Zhou and Huai Liao and Danny T. M. Chan and Ming Feng and Wai S. Poon and Hongliang Ren and Dong Yi and Nassir Navab and Gaofeng Meng and Jiebo Luo and Hongbin Liu and Zhen Lei},
      year={2026},
      eprint={2602.05638},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.05638}, 
}
```

## Acknowledgement

SurgMotion is built on top of [V-JEPA 2](https://github.com/facebookresearch/vjepa2) by Meta. We thank the authors of the following works whose open-source models were used in our benchmark comparison:

[DINOv2](https://github.com/facebookresearch/dinov2) | [Endo-FM](https://github.com/med-air/Endo-FM) | [EndoMamba](https://github.com/TianCuteQY/EndoMamba) | [EndoSSL](https://github.com/royhirsch/endossl) | [EndoViT](https://github.com/DominikBatic/EndoViT) | [GastroNet](https://ieeexplore.ieee.org/document/10243003) | [GSViT](https://github.com/SamuelSchmidgall/GSViT) | [SelfSupSurg](https://github.com/CAMMA-public/SelfSupSurg) | [SurgeNet](https://github.com/TimJaspers0801/SurgeNet) | [SurgVISTA](https://github.com/isyangshu/SurgVISTA) | [SurgVLP](https://github.com/CAMMA-public/SurgVLP) | [VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2)

---

<div align="center">

**Centre for Artificial Intelligence and Robotics, Hong Kong Institute of Science and Innovation, CAS**

[Project Page](https://surgmotion.cares-copilot.com/) | [Paper](https://arxiv.org/abs/2602.05638) | [GitHub](https://github.com/CAIR-HKISI/SurgMotion)

</div>
