
## install

```bash
conda create -n jepa_torch python=3.12
conda activate jepa_torch
pip install -e .
```

download checkpoint
```bash
mkdir ckpts
cd ckpts
wget https://dl.fbaipublicfiles.com/vjepa2/vitl.pt
wget https://dl.fbaipublicfiles.com/vjepa2/vith.pt
wget https://dl.fbaipublicfiles.com/vjepa2/vitg.pt
wget https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt
```

prepare dataset
``` bash 
mkdir data
## download pitvis

## extract frames
python data_process/pitvis_data_list.py

## make clips for training
python data_process/pitvis_clip_list.py
```

```markdown
data/Surge_Frames/
├── PitVis/                           # 直接下载的视频数据和标注
|   |──video_01.mp4
|   |──video_02.mp4
|   |──annotation_01.csv
├── pitvis/
│   ├── train_metadata.csv
│   └── val_metadata.csv
└── pitvis_clips_64f/
    ├── train_dense_64f.csv           # CSV中的路径: pitvis_clips_64f/clip_dense_64f_info/train/xxx.txt
    ├── val_dense_64f.csv
    └── clip_dense_64f_info/
        ├── train/
        │   └── case001_c000_xxx.txt   # txt中的路径: pitvis/case001/frame_xxx.jpg
        └── val/
            └── case002_c000_xxx.txt
```


probing training
