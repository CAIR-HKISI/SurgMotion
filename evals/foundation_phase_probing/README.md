# Foundation Models Probing Scope

# Quick Start

```bash
# Submit multiple task at a time
sbatch SurgMotion/scripts/srun_foundation_probing.sh
```

You may have to adjust params based on the platforms.

# To Customize

To change the test plan, modifying these lines with your new config .yaml:

```bash
# srun_foundation_probing_xxx.sh
TASKS=( # the config files dir SurgMotion/configs + ...
    "foundation_model_probing/dinov3/OphNet"
    "foundation_model_probing/endofm/M2CAI16"
    "foundation_model_probing/endofm/OphNet"
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
  model_type: {MODEL} # refer to the name listed in SurgMotion/evals/foundation_phase_probing/modelcustom/foundation_model_wrapper.py
  encoder:
    model_name: {MODEL}
```

For now in the foundation_model_wrapper.py we support models:

| Model | DINOv2 | DINOv3 | EndoViT | Gastronet | SelfSupSurg | EndoSSL | GSViT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| model_name | dinov2 | dinov3 | endovit | gastronet | selfsupsurg | endossl | gsvit |

To add new models:

```python
# 1.Write a new adapter referring the adapters in SurgMotion/evals/foundation_phase_probing/modelcustom/adapters
     Make sure your model adapter process
		(1)Input: Any, For example[B:BatchSize, C:Channels, F:Frames(Times), H:Height, W:Width] 
		(2)Output: [B, F*N, D]
				
# 2.Modify the foundation_model_wrapper.py by adding new model support, e.g.
    elif model_type == '{Your New Model Name}':
        from .adapters.{Your New Model}_adapter.py import {Your New Model Adapter}
        adapter = {Your New Model Adapter}.from_config(
            resolution=resolution,
            checkpoint=checkpoint,
            model_name=model_name
        )
            
# 3.Create config in SurgMotion/configs/foundation_model_probing/{Your New Model Name}/{DATASET}

# 4.Add them on SurgMotion/scripts/srun_foundation_probing_xxx.sh and Sbatch
```