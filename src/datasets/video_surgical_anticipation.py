"""
SurgicalAnticipationDataset: 支持手术流程预测/器械预测的多目标时间回归 Dataset。

与 SurgicalVideoDataset 共享帧加载逻辑，但监督信号改为多维连续值：
  - target_reg: [num_targets] float tensor
  - vid_id: case_id 哈希
  - data_idx: 样本在 CSV 中的行索引
  - dataset_idx: 多数据源场景下的数据集索引

CSV 需包含列: ant_reg_{TargetName}
若未显式给出 target_names，则会从 CSV 中自动推断 ant_reg_* 列。
"""

import os
import pathlib
from logging import getLogger

import pandas as pd
import torch

from src.datasets.video_surgical import SurgicalVideoDataset
from src.datasets.utils.dataloader import MonitoredDataset, NondeterministicDataLoader
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

logger = getLogger()
ANTICIPATION_PREFIX = "ant_reg_"

CHOLEC80_PHASES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]


class SurgicalAnticipationDataset(SurgicalVideoDataset):
    """
    继承 SurgicalVideoDataset 的帧加载逻辑，
    覆盖 label 加载以支持 anticipation 任务。
    """

    def __init__(self, data_paths, phase_names=None, target_names=None, **kwargs):
        self.target_names = list(target_names or phase_names) if (target_names or phase_names) else None
        self.num_targets = len(self.target_names) if self.target_names is not None else None
        self.anticipation_reg = []
        self._ant_loaded = False

        super().__init__(data_paths=data_paths, **kwargs)

        self._load_anticipation_labels(data_paths)

    @staticmethod
    def _infer_target_names(df):
        return [c[len(ANTICIPATION_PREFIX):] for c in df.columns if c.startswith(ANTICIPATION_PREFIX)]

    def _load_anticipation_labels(self, data_paths):
        if self._ant_loaded:
            return
        if isinstance(data_paths, str):
            data_paths = [data_paths]

        for data_path in data_paths:
            if not data_path.endswith(".csv"):
                if self.num_targets is None:
                    raise ValueError(
                        "无法从非 CSV 数据源自动推断 anticipation 目标维度，请显式提供 target_names"
                    )
                n_samples = sum(1 for _ in open(data_path)) if os.path.exists(data_path) else 0
                for _ in range(n_samples):
                    self.anticipation_reg.append([5.0] * self.num_targets)
                continue

            df = pd.read_csv(data_path)
            inferred_target_names = self._infer_target_names(df)
            if not inferred_target_names:
                raise ValueError(f"{data_path} 中未找到 ant_reg_* 列，无法构建 anticipation 数据集")

            if self.target_names is None:
                self.target_names = inferred_target_names
                self.num_targets = len(self.target_names)
            elif inferred_target_names != self.target_names:
                raise ValueError(
                    f"{data_path} 的 anticipation 目标列与当前配置不一致: "
                    f"expected={self.target_names}, found={inferred_target_names}"
                )

            reg_cols = [f"{ANTICIPATION_PREFIX}{p}" for p in self.target_names]

            has_reg = all(c in df.columns for c in reg_cols)
            if not has_reg:
                logger.warning(
                    f"CSV {data_path} 缺少 anticipation 回归列 (ant_reg_*), "
                    f"将使用默认值 5.0"
                )
                for _ in range(len(df)):
                    self.anticipation_reg.append([5.0] * self.num_targets)
                continue

            for _, row in df.iterrows():
                reg_vals = [float(row[c]) for c in reg_cols]
                self.anticipation_reg.append(reg_vals)

        self._ant_loaded = True
        assert len(self.anticipation_reg) == len(self.samples), \
            f"Anticipation 标签数({len(self.anticipation_reg)}) != 样本数({len(self.samples)})"

        reg_array = torch.tensor(self.anticipation_reg, dtype=torch.float32)
        reg_mean = reg_array.mean(dim=0).tolist()
        print(f"[AnticipationDataset] {len(self.samples)} samples, {self.num_targets} regression targets")
        print(f"[AnticipationDataset] Targets: {self.target_names}")
        print(f"[AnticipationDataset] Mean target values: {dict(zip(self.target_names, [round(v, 4) for v in reg_mean]))}")

    def get_item_frames(self, index):
        result = super().get_item_frames(index)
        if result is None:
            return None
        buffer, label_list, clip_indices = result

        target_reg = torch.tensor(self.anticipation_reg[index], dtype=torch.float32)
        vid_id = label_list[1]  # case_id hash
        data_idx = label_list[2]  # Index
        dataset_idx = label_list[3] if len(label_list) > 3 else 0

        new_label = (target_reg, vid_id, data_idx, dataset_idx)
        return buffer, new_label, clip_indices

    def get_item_image(self, index):
        result = super().get_item_image(index)
        if result is None:
            return None
        buffer, label_list, clip_indices = result

        target_reg = torch.tensor(self.anticipation_reg[index], dtype=torch.float32)
        vid_id = label_list[1]
        data_idx = label_list[2]
        dataset_idx = label_list[3] if len(label_list) > 3 else 0

        new_label = (target_reg, vid_id, data_idx, dataset_idx)
        return buffer, new_label, clip_indices


def make_surgical_anticipation_dataset(
    data_paths,
    batch_size,
    frames_per_clip=8,
    dataset_fpcs=None,
    frame_step=1,
    duration=None,
    fps=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    deterministic=True,
    log_dir=None,
    phase_names=None,
    target_names=None,
):
    dataset = SurgicalAnticipationDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        dataset_fpcs=dataset_fpcs,
        duration=duration,
        fps=fps,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        shared_transform=shared_transform,
        transform=transform,
        phase_names=phase_names,
        target_names=target_names,
    )

    log_dir = pathlib.Path(log_dir) if log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        resource_log_filename = log_dir / f"resource_file_{rank}_%w.csv"
        dataset = MonitoredDataset(
            dataset=dataset,
            log_filename=str(resource_log_filename),
            log_interval=10.0,
            monitor_interval=5.0,
        )

    logger.info("SurgicalAnticipationDataset created")
    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    if deterministic:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )
    else:
        data_loader = NondeterministicDataLoader(
            dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )

    logger.info("SurgicalAnticipationDataset data loader created")
    return dataset, data_loader, dist_sampler
