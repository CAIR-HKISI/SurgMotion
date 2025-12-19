# Foundation Models Probing Scope

# Quick Start

```bash
# Submit multiple task at a time, waiting for slurm resourcing
sbatch NSJepa/scripts/srun_foundation_probing_allatonce.sh

# Submit multiple task one after finishing another, queue behaviour
sbatch NSJepa/scripts/srun_foundation_probing_aggregation.sh
```

You may have to adjust params based on the platforms.

# To Customize

To change the test plan, modifying these lines with your new config .yaml:

```bash
# srun_foundation_probing_xxx.sh
TASKS=( # the config files dir NSJepa/configs + ...
    "fdtn_probing/dinov3/OphNet"
    "fdtn_probing/endofm/M2CAI16"
    "fdtn_probing/endofm/OphNet"
)

FNAMES=( # the actual config files name
    "dinov3_vitl_64f_OphNet.yaml"
    "endofm_vitb_64f_M2CAI16.yaml"
    "endofm_vitb_64f_OphNet.yaml"
)
```

Already supported datasets and models:

| Dataset | Atlas | AutoLaparo | CATARACTS | cholec80 | EgoSurgery | M2CAI16 | OphNet | PmLR50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | dinov2 | dinov3 | endovit | gastronet | selfsupsurg | endossl | gsvit | … |

To change the config .yaml:

```yaml
# For example {MODEL} on {DATASET}, you should modify as below
tag: {MODEL}_vitl_64f_{DATASET}
folder: logs/foundation/{MODEL}_vitl_{DATASET}
wandb: ...(skip)
experiment:
  data:
    dataset_train: data/Surge_Frames/{DATASET}/clips_64f/train_dense_64f_detailed.csv
    dataset_val: data/Surge_Frames/{DATASET}/clips_64f/test_dense_64f_detailed.csv
    num_classes: 7 # based on your dataset settings
    resolution: 224 # based on your foundation settings
model_kwargs:
  model_type: {MODEL} # refer to the name listed in NSJepa/evals/foundation_phase_probing/modelcustom/foundation_model_wrapper.py
  encoder:
    model_name: {MODEL}
```

For now in the foundation_model_wrapper.py we support models:

| Model | DINOv2 | DINOv3 | EndoViT | Gastronet | SelfSupSurg | EndoSSL | GSViT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| model_name | dinov2 | dinov3 | endovit | gastronet | selfsupsurg | endossl | gsvit |

To add new models:

```python
# 1.Write a new adapter referring the adapters in NSJepa/evals/foundation_phase_probing/modelcustom/adapters
     Make sure your model adapter process
		(1)Input: Any, For example[B:BatchSize, C:Channels, F:Frames(Times), H:Height, W:Width] 
		(2)Output: [B, F*N, D]
				
# 2.Modify the foundation_model_wrapper.py by adding new model support, e.g.
		if model_type == 'dinov2':
        from .adapters.dinov2_adapter import DINOv2Adapter
        adapter = DINOv2Adapter.from_config(
            resolution=resolution,
            checkpoint=checkpoint,
            model_name=model_name
        )
    ...
    elif model_type == '{Your New Model Name}':
        from .adapters.{Your New Model}_adapter.py import {Your New Model Adapter}
        adapter = {Your New Model Adapter}.from_config(
            resolution=resolution,
            checkpoint=checkpoint,
            model_name=model_name
            
# 3.Create config in NSJepa/configs/fdtn_probing/{Your New Model Name}/{DATASET}

# 4.Add them on NSJepa/scripts/srun_foundation_probing_xxx.sh and Sbatch
```

### Surgical Phase Recognition

| Model Name | Versions | Options |
| --- | --- | --- |
| **DINO v2** | dinov2_vitl14 | *ViT-S, B, L |
| **DINO v3** | dinov3_vitl16 | *ViT-S, B, L or ConvNeXT |
| **EndoViT** | EndoViT_SPR | EndoViT_Seg/ATD/SPR |
| **EndoFM** | Pretrained ViT-B | Pretrained/Finetuned |
| **GastroNet** | ViT-S-DINO | RN50_DINOv1/MOCOv2/SIMCLRv2(200K-5M)-ViTs_DINOv1 |
| **SelfSupSurg** | DINO | MoCo V2/SimCLR/SwAV/DINOv1 |
| **SurgVISTA** | ❌ | Yet to release |
| **EndoSSL** | ViT-L (Laparoscopy&Colonoscopy) | *ViT-S, B, L(Laparoscopy/Colonoscopy) |
| **GSViT** | GSViT | **Only** EfficientViT M5 |
| **Endomamba** | Endomamba_SPR | Pretrained/Fintuned（Class/Seg/SPR） |

### Dinov2 Options

| model | # ofparams | withregisters | ImageNetk-NN | ImageNetlinear | download |
| --- | --- | --- | --- | --- | --- |
| ViT-S/14 distilled | 21 M | ❌ | 79.0% | 81.1% | backbone only |
| ViT-S/14 distilled | 21 M | ✅ | 79.1% | 80.9% | backbone only |
| ViT-B/14 distilled | 86 M | ❌ | 82.1% | 84.5% | backbone only |
| ViT-B/14 distilled | 86 M | ✅ | 82.0% | 84.6% | backbone only |
| ViT-L/14 distilled | 300 M | ❌ | 83.5% | 86.3% | backbone only |
| ViT-L/14 distilled | 300 M | ✅ | 83.8% | 86.7% | backbone only |
| ViT-g/14 | 1,100 M | ❌ | 83.5% | 86.5% | backbone only |
| ViT-g/14 | 1,100 M | ✅ | 83.7% | 87.1% | backbone only |

### Dinov3 Options

| **Model** | **Parameters** | **PretrainingDataset** | **Download** |
| --- | --- | --- | --- |
| ViT-S/16 distilled | 21M | LVD-1689M | [[link]](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) |
| ViT-S+/16 distilled | 29M | LVD-1689M | [[link]](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) |
| ViT-B/16 distilled | 86M | LVD-1689M | [[link]](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) |
| ViT-L/16 distilled | 300M | LVD-1689M | [[link]](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) |
| ViT-H+/16 distilled | 840M | LVD-1689M | [[link]](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) |
| ViT-7B/16 | 6,716M | LVD-1689M | [[link]](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) |
| ConvNeXt Tiny | 29M | LVD-1689M | [[link]](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) |
| ConvNeXt Small | 50M | LVD-1689M | [[link]](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) |
| ConvNeXt Base | 89M | LVD-1689M | [[link]](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) |
| ConvNeXt Large | 198M | LVD-1689M | [[link]](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) |

### EndoViT Options

| **Excluded Data (Test Sets)** | **Checkpoint** |
| --- | --- |
| CholecSeg8k (Segmentation) | [EndoViT_Seg](https://drive.google.com/file/d/1NJ-4ZL40kHA_WZ1NylahaS84FcvnigjF/view?usp=share_link) |
| CholecT45 (Action Triplet Detection) | [EndoViT ATD](https://drive.google.com/file/d/1NReHXlMiBkVJiZcuJAGx6sGWh7pNgg_i/view?usp=share_link) |
| Cholec80 (Surgical Phase Recognition) | [EndoViT_SPR](https://drive.google.com/file/d/1NK8aMb9SaApCn_vLigyDSn3aTI55QVT1/view?usp=share_link) |

### EndoFM Finetuned Options

| **Dataset** | **PolypDiag** | **CVC-12k** | **KUMC** |
| --- | --- | --- | --- |
| Our Paper | 90.7 | 73.9 | 84.1 |
| Released Model | 91.5 | 76.6 | 84.0 |
| Weights | [link](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155167044_link_cuhk_edu_hk/ERSlUP10MGpBuhg1uN5iaHABKqz1SPQSrr03j4sEWey-bw?e=muv8RL) | [link](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155167044_link_cuhk_edu_hk/EePnpTllUCFEqpYp6BFPv0sBQyST4CV4jQ8pvaRynCkD7Q?e=f7LeBx) | [link](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155167044_link_cuhk_edu_hk/EYPkwbFyMfxEirezWtumAGIBSCTQ0EvDN4u99KKiRsaVBA?e=DsrkVG) |

### GastroNet Options

| Backbone | Framework | Scale |  |
| --- | --- | --- | --- |
| **ResNet50** | **Initialized with Billion-Scale weights and pretrained DINOv1** | **Gastronet-5M** | [RN50_Billion-Scale-SWSL+GastroNet-5M_DINOv1.pth](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights/blob/main/RN50_Billion-Scale-SWSL%2BGastroNet-5M_DINOv1.pth)[94.4 MB**xet**](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights/resolve/main/RN50_Billion-Scale-SWSL%2BGastroNet-5M_DINOv1.pth?download=true)[Upload 7 filesJuly 5, 2024 12:48 PM](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights/commit/46fa013ccf3c4e9cacc728018f108be946ad57f7) |
| **ResNet50** | **DINOv1** | **Gastronet-200K** | [RN50_GastroNet-200K_DINOv1.pth](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights/blob/main/RN50_GastroNet-200K_DINOv1.pth)[94.4 MB**xet**](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights/resolve/main/RN50_GastroNet-200K_DINOv1.pth?download=true)[Upload 7 filesJuly 5, 2024 12:48 PM](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights/commit/46fa013ccf3c4e9cacc728018f108be946ad57f7) |
| **ResNet50** | **DINOv1** | **Gastronet-1M** | [RN50_GastroNet-1M_DINOv1.pth](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights/blob/main/RN50_GastroNet-1M_DINOv1.pth)[94.4 MB**xet**](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights/resolve/main/RN50_GastroNet-1M_DINOv1.pth?download=true)[Upload 7 filesJuly 5, 2024 12:48 PM](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights/commit/46fa013ccf3c4e9cacc728018f108be946ad57f7) |
| **ResNet50** | **DINOv1** | **Gastronet-5M** | [RN50_GastroNet-5M_DINOv1.pth](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights/blob/main/RN50_GastroNet-5M_DINOv1.pth)[94.4 MB**xet**](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights/resolve/main/RN50_GastroNet-5M_DINOv1.pth?download=true)[Upload 7 filesJuly 5, 2024 12:48 PM](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights/commit/46fa013ccf3c4e9cacc728018f108be946ad57f7) |
| **ResNet50** | **MOCOv2** | **Gastronet-5M** | [RN50_GastroNet-5M_MOCOv2.pth](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights/blob/main/RN50_GastroNet-5M_MOCOv2.pth)[94.4 MB**xet**](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights/resolve/main/RN50_GastroNet-5M_MOCOv2.pth?download=true)[Upload 7 filesJuly 5, 2024 12:48 PM](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights/commit/46fa013ccf3c4e9cacc728018f108be946ad57f7) |
| **ResNet50** | **SIMCLRv2** | **Gastronet-5M** | [RN50_GastroNet-5M_SIMCLRv2.pth](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights/blob/main/RN50_GastroNet-5M_SIMCLRv2.pth)[94.4 MB**xet**](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights/resolve/main/RN50_GastroNet-5M_SIMCLRv2.pth?download=true)[Upload 7 filesJuly 5, 2024 12:48 PM](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights/commit/46fa013ccf3c4e9cacc728018f108be946ad57f7) |
| **VIT-small** | **DINOv1** | **Gastronet-5M** | [VITS_GastroNet-5M_DINOv1.pth](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights/blob/main/VITS_GastroNet-5M_DINOv1.pth) |

### SelfSupSurg Options

| **Model** | **Model Weights** |
| --- | --- |
| [MoCo V2](https://github.com/CAMMA-public/SelfSupSurg/blob/main/configs/config/hparams/cholec80/pre_training/cholec_to_cholec/series_01/h001.yaml) | [download](https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_moco_v2_surg.torch) |
| [SimCLR](https://github.com/CAMMA-public/SelfSupSurg/blob/main/configs/config/hparams/cholec80/pre_training/cholec_to_cholec/series_01/h002.yaml) | [download](https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_simclr_surg.torch) |
| [SwAV](https://github.com/CAMMA-public/SelfSupSurg/blob/main/configs/config/hparams/cholec80/pre_training/cholec_to_cholec/series_01/h003.yaml) | [download](https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_swav_surg.torch) |
| [DINO](https://github.com/CAMMA-public/SelfSupSurg/blob/main/configs/config/hparams/cholec80/pre_training/cholec_to_cholec/series_01/h004.yaml) | [download](https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_dino_surg.torch) |

### EndoSSL Options

| **Arch** | **Dataset** | **Down-stream results** | **Link** |
| --- | --- | --- | --- |
| ViT-S | Private Laparoscopy | Cholec80 F1: 83.4 | [Link](https://drive.google.com/drive/folders/1CctMDXGo8AlyZQSoWwiVyssMrEBZp3IE?usp=drive_link) |
| ViT-B | Private Laparoscopy | Cholec80 F1: 82.6 | [Link](https://drive.google.com/drive/folders/1zcLKhE7H50GIDeb53chLrE5SUBtBooAR?usp=drive_link) |
| ViT-L | Private Laparoscopy | Cholec80 F1: 84.0 | [Link](https://drive.google.com/drive/folders/11TdNyl4HGvpoi6Ro0zZ28L1qxY4IAJPb?usp=drive_link) |
| - | - | - | - |
| ViT-S | Private Colonoscopy | PolypSet Acc: 78.5 | [Link](https://drive.google.com/drive/folders/1GfBVLh3r6A2ctkJyy_1Uc0onSg8tNykM?usp=drive_link) |
| ViT-B | Private Colonoscopy | PolypSet Acc: 78.2 | [Link](https://drive.google.com/drive/folders/1-ispnt7CElWxntmA61XDDHbbBDwZ6njN?usp=drive_link) |
| ViT-L | Private Colonoscopy | PolypSet Acc: 80.4 | [Link](https://drive.google.com/drive/folders/1eq_KcAY_OQU07Ey8XFpvWlsZhMDfl86K?usp=drive_link) |

### Endomamba

|  | Classification | Segmentation | Surgical Phase Recognition |
| --- | --- | --- | --- |
| Metrics | F1: 96.0 | Dice 85.4 | Acc: 83.3 |
| Weights | [link](https://pan.cstcloud.cn/s/3SrWtTt5TbI) | [link](https://pan.cstcloud.cn/s/0xVTmWnQ4c) | [link](https://pan.cstcloud.cn/s/lZhbMk9GQic) |