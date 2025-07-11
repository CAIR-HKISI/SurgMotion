# import math
# import os
# import pathlib
# import warnings
# from logging import getLogger
# import numpy as np
# import pandas as pd
# import torch
# import torchvision
# from PIL import Image

# import sys
# sys.path.append('/data/wjl/vjepa2')

# from src.datasets.utils.dataloader import ConcatIndices, MonitoredDataset, NondeterministicDataLoader
# from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

# _GLOBAL_SEED = 0
# logger = getLogger()


# def make_surgical_videodataset(
#     data_paths,
#     batch_size,
#     frames_per_clip=8,
#     dataset_fpcs=None,
#     frame_step=4,
#     duration=None,
#     fps=None,
#     num_clips=1,
#     random_clip_sampling=True,
#     allow_clip_overlap=False,
#     filter_short_videos=False,
#     filter_long_videos=int(10**9),
#     transform=None,
#     shared_transform=None,
#     rank=0,
#     world_size=1,
#     datasets_weights=None,
#     collator=None,
#     drop_last=True,
#     num_workers=10,
#     pin_mem=True,
#     persistent_workers=True,
#     deterministic=True,
#     log_dir=None,
# ):
#     dataset = SurgicalVideoDataset(
#         data_paths=data_paths,
#         datasets_weights=datasets_weights,
#         frames_per_clip=frames_per_clip,
#         dataset_fpcs=dataset_fpcs,
#         duration=duration,
#         fps=fps,
#         frame_step=frame_step,
#         num_clips=num_clips,
#         random_clip_sampling=random_clip_sampling,
#         allow_clip_overlap=allow_clip_overlap,
#         filter_short_videos=filter_short_videos,
#         filter_long_videos=filter_long_videos,
#         shared_transform=shared_transform,
#         transform=transform,
#     )

#     log_dir = pathlib.Path(log_dir) if log_dir else None
#     if log_dir:
#         log_dir.mkdir(parents=True, exist_ok=True)
#         resource_log_filename = log_dir / f"resource_file_{rank}_%w.csv"
#         dataset = MonitoredDataset(
#             dataset=dataset,
#             log_filename=str(resource_log_filename),
#             log_interval=10.0,
#             monitor_interval=5.0,
#         )

#     logger.info("VideoDataset dataset created")
#     if datasets_weights is not None:
#         dist_sampler = DistributedWeightedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
#     else:
#         dist_sampler = torch.utils.data.distributed.DistributedSampler(
#             dataset, num_replicas=world_size, rank=rank, shuffle=True
#         )

#     if deterministic:
#         data_loader = torch.utils.data.DataLoader(
#             dataset,
#             collate_fn=collator,
#             sampler=dist_sampler,
#             batch_size=batch_size,
#             drop_last=drop_last,
#             pin_memory=pin_mem,
#             num_workers=num_workers,
#             persistent_workers=(num_workers > 0) and persistent_workers,
#         )
#     else:
#         data_loader = NondeterministicDataLoader(
#             dataset,
#             collate_fn=collator,
#             sampler=dist_sampler,
#             batch_size=batch_size,
#             drop_last=drop_last,
#             pin_memory=pin_mem,
#             num_workers=num_workers,
#             persistent_workers=(num_workers > 0) and persistent_workers,
#         )
#     logger.info("VideoDataset unsupervised data loader created")

#     return dataset, data_loader, dist_sampler


# class SurgicalVideoDataset(torch.utils.data.Dataset):
#     """Video classification dataset for pre-extracted frames."""

#     def __init__(
#         self,
#         data_paths,
#         datasets_weights=None,
#         frames_per_clip=16,
#         fps=1,  # 固定为1fps
#         dataset_fpcs=None,
#         frame_step=1,  # 固定为1帧步长
#         num_clips=1,
#         transform=None,
#         shared_transform=None,
#         random_clip_sampling=True,
#         allow_clip_overlap=False,
#         filter_short_videos=False,
#         filter_long_videos=int(10**9),
#         duration=None,  # 忽略duration参数
#     ):
#         self.data_paths = data_paths
#         self.datasets_weights = datasets_weights
#         self.frame_step = frame_step
#         self.num_clips = num_clips
#         self.transform = transform
#         self.shared_transform = shared_transform
#         self.random_clip_sampling = random_clip_sampling
#         self.allow_clip_overlap = allow_clip_overlap
#         self.filter_short_videos = filter_short_videos
#         self.filter_long_videos = filter_long_videos
#         self.duration = duration
#         self.fps = fps

#         # 打印参数设置信息
#         print(f"[VideoDataset] 强制设置 fps={fps}, frame_step={frame_step}")
#         if duration is not None:
#             print(f"[VideoDataset] 忽略传入的 duration={duration} 参数")

#         if isinstance(data_paths, str):
#             data_paths = [data_paths]

#         if dataset_fpcs is None:
#             self.dataset_fpcs = [frames_per_clip for _ in data_paths]
#         else:
#             if len(dataset_fpcs) != len(data_paths):
#                 raise ValueError("Frames per clip not properly specified for data paths")
#             self.dataset_fpcs = dataset_fpcs

#         # 加载数据路径和标签
#         samples, labels = [], []
#         self.num_samples_per_dataset = []
#         for data_path in self.data_paths:
#             if data_path.endswith(".csv"):
#                 try:
#                     data = pd.read_csv(data_path, header=None, delimiter=" ")
#                 except pd.errors.ParserError:
#                     data = pd.read_csv(data_path, header=None, delimiter="::")
#                 samples += list(data.values[:, 0])
#                 labels += list(data.values[:, 1])
#                 self.num_samples_per_dataset.append(len(data))
#             elif data_path.endswith(".npy"):
#                 data = np.load(data_path, allow_pickle=True)
#                 samples += list(map(lambda x: repr(x)[1:-1], data))
#                 labels += [0] * len(data)
#                 self.num_samples_per_dataset.append(len(data))

#         self.per_dataset_indices = ConcatIndices(self.num_samples_per_dataset)
#         self.sample_weights = None
#         if self.datasets_weights is not None:
#             self.sample_weights = [dw / ns for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset) 
#                                  for _ in range(ns)]
#         self.samples = samples
#         self.labels = labels

#     def __getitem__(self, index):
#         sample = self.samples[index]
#         loaded_sample = False
#         while not loaded_sample:
#             if not isinstance(sample, str):
#                 logger.warning("Invalid sample.")
#             else:
#                 if sample.split(".")[-1].lower() in ("txt"):
#                     loaded_sample = self.get_item_frames(index)
#                 elif sample.split(".")[-1].lower() in ("jpg", "png", "jpeg"):
#                     loaded_sample = self.get_item_image(index)
#                 else:
#                     loaded_sample = self.get_item_video(index)  # 保留视频加载逻辑

#             if not loaded_sample:
#                 index = np.random.randint(self.__len__())
#                 sample = self.samples[index]
#         return loaded_sample

#     def get_item_frames(self, index):
#         sample = self.samples[index]
#         dataset_idx, _ = self.per_dataset_indices[index]
#         frames_per_clip = self.dataset_fpcs[dataset_idx]

#         buffer, clip_indices = self.load_frames_from_txt(sample, frames_per_clip)
#         if len(buffer) == 0:
#             return None  # 加载失败时返回None

#         label = self.labels[index]

#         def split_into_clips(video):
#             fpc = frames_per_clip
#             nc = self.num_clips
#             return [video[i * fpc : (i + 1) * fpc] for i in range(nc)]

#         # 应用转换（保持与原代码一致的处理流程）
#         if self.shared_transform is not None:
#             buffer = self.shared_transform(buffer)
#         buffer = split_into_clips(buffer)
#         if self.transform is not None:
#             buffer = [self.transform(clip) for clip in buffer]
        
#         return buffer, label, clip_indices  # 返回列表形式的clip，与原代码一致

#     def get_item_image(self, index):
#         sample = self.samples[index]
#         dataset_idx, _ = self.per_dataset_indices[index]
#         fpc = self.dataset_fpcs[dataset_idx]

#         try:
#             image_tensor = torchvision.io.read_image(sample, torchvision.io.ImageReadMode.RGB)
#         except Exception:
#             return None
#         label = self.labels[index]
#         clip_indices = [np.arange(0, fpc, dtype=np.int32)]

#         # 扩展图像维度为[T, H, W, 3]
#         buffer = image_tensor.unsqueeze(0).repeat(fpc, 1, 1, 1).permute(0, 2, 3, 1)
#         if self.shared_transform is not None:
#             buffer = self.shared_transform(buffer)
#         if self.transform is not None:
#             buffer = [self.transform(buffer)]
        
#         return buffer, label, clip_indices

#     def load_frames_from_txt(self, txt_path, fpc):
#         """从txt文件加载帧并保持与原视频加载一致的buffer结构"""
#         if not os.path.exists(txt_path):
#             warnings.warn(f"txt path not found: {txt_path}")
#             return [], []

#         try:
#             with open(txt_path, 'r') as f:
#                 frame_paths = [line.strip() for line in f if line.strip()]
#         except Exception as e:
#             warnings.warn(f"读取txt文件失败: {txt_path}, 错误: {e}")
#             return [], []

#         # 过滤过短/过长片段
#         if self.filter_short_videos and len(frame_paths) < fpc:
#             warnings.warn(f"片段过短，跳过: {txt_path} (帧数: {len(frame_paths)})")
#             return [], []
#         if len(frame_paths) > self.filter_long_videos:
#             warnings.warn(f"片段过长，跳过: {txt_path} (帧数: {len(frame_paths)})")
#             return [], []

#         fstp = self.frame_step
#         clip_len = fpc * fstp
#         clip_len = min(clip_len, len(frame_paths))  # 确保不超出实际帧数

#         all_indices, clip_indices = [], []
#         partition_len = len(frame_paths) // self.num_clips if self.num_clips > 0 else len(frame_paths)

#         for i in range(self.num_clips):
#             if partition_len > clip_len:
#                 # 随机采样窗口
#                 end_idx = np.random.randint(clip_len, partition_len) if self.random_clip_sampling else clip_len
#                 start_idx = end_idx - clip_len
#                 indices = np.linspace(start_idx, end_idx-1, fpc).astype(np.int64)
#                 indices = np.clip(indices, start_idx, end_idx-1)
#                 indices += i * partition_len
#             else:
#                 # 处理短片段
#                 if not self.allow_clip_overlap and len(frame_paths) < fpc:
#                     indices = np.concatenate([
#                         np.linspace(0, partition_len-1, partition_len//fstp, dtype=np.int64),
#                         np.ones(fpc - partition_len//fstp, dtype=np.int64) * (partition_len-1)
#                     ])
#                     indices += i * partition_len
#                 else:
#                     sample_len = min(clip_len, len(frame_paths)) - 1
#                     indices = np.linspace(0, sample_len, sample_len//fstp, dtype=np.int64)
#                     if len(indices) < fpc:
#                         indices = np.concatenate([indices, np.ones(fpc - len(indices), dtype=np.int64) * sample_len])
#                     clip_step = (len(frame_paths) - clip_len) // (self.num_clips - 1) if self.num_clips > 1 else 0
#                     indices += i * clip_step
#             clip_indices.append(indices)
#             all_indices.extend(indices)

#         # 加载图像并构建buffer [T, H, W, 3]
#         buffer = []
#         for idx in all_indices:
#             idx = min(idx, len(frame_paths)-1)
#             frame_path = frame_paths[idx]
#             if not os.path.exists(frame_path):
#                 buffer.append(np.zeros((224, 224, 3), dtype=np.uint8))
#                 continue
#             try:
#                 img = Image.open(frame_path).convert('RGB')
#                 buffer.append(np.array(img))
#             except Exception:
#                 buffer.append(np.zeros((224, 224, 3), dtype=np.uint8))
#         return np.array(buffer), clip_indices

#     def __len__(self):
#         return len(self.samples)



# if __name__ == "__main__":
#     # 
#     data = SurgicalVideoDataset(
#         data_paths=["/data/wjl/vjepa2/data_process/fixed_64frames_clips.csv"],
#         datasets_weights=[1.0],
#         frames_per_clip=16,
#         fps=None,
#         dataset_fpcs=None,
#         frame_step=1,
#         num_clips=2,
#         transform=None,
#         shared_transform=None,
#         random_clip_sampling=True,
#         allow_clip_overlap=True,
#         filter_short_videos=False,
#         filter_long_videos=int(10**9),
#         duration=None,  # duration in seconds
#     )
#     print(len(data))
    
#     for item in data:
#         import ipdb; ipdb.set_trace()
#         print(item)
        

import math
import os
import pathlib
import warnings
from logging import getLogger
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image

import sys
sys.path.append('/data/wjl/vjepa2')

from src.datasets.utils.dataloader import ConcatIndices, MonitoredDataset, NondeterministicDataLoader
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()


def make_surgical_videodataset(
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
):
    dataset = SurgicalVideoDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        dataset_fpcs=dataset_fpcs,
        duration=duration,
        fps=fps,
        frame_step=frame_step,  # 传递frame_step参数
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        shared_transform=shared_transform,
        transform=transform,
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

    logger.info("VideoDataset dataset created")
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
    logger.info("VideoDataset unsupervised data loader created")

    return dataset, data_loader, dist_sampler


class SurgicalVideoDataset(torch.utils.data.Dataset):
    """Video classification dataset for pre-extracted frames."""

    def __init__(
        self,
        data_paths,
        datasets_weights=None,
        frames_per_clip=16,
        fps=1,  # 固定为1fps
        dataset_fpcs=None,
        frame_step=1,  # 明确设置默认值为1，避免None
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,  # 忽略duration参数
    ):
        self.data_paths = data_paths
        self.datasets_weights = datasets_weights
        # 确保frame_step不为None，若为None则设为1
        self.frame_step = frame_step if frame_step is not None else 1
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration
        self.fps = fps

        # 打印参数设置信息
        print(f"[VideoDataset] 强制设置 fps={fps}, frame_step={self.frame_step}")
        if duration is not None:
            print(f"[VideoDataset] 忽略传入的 duration={duration} 参数")

        if isinstance(data_paths, str):
            data_paths = [data_paths]

        if dataset_fpcs is None:
            self.dataset_fpcs = [frames_per_clip for _ in data_paths]
        else:
            if len(dataset_fpcs) != len(data_paths):
                raise ValueError("Frames per clip not properly specified for data paths")
            self.dataset_fpcs = dataset_fpcs

        # 加载数据路径和标签
        samples, labels = [], []
        self.num_samples_per_dataset = []
        for data_path in self.data_paths:
            if data_path.endswith(".csv"):
                try:
                    data = pd.read_csv(data_path, header=None, delimiter=" ")
                except pd.errors.ParserError:
                    data = pd.read_csv(data_path, header=None, delimiter="::")
                samples += list(data.values[:, 0])
                labels += list(data.values[:, 1])
                self.num_samples_per_dataset.append(len(data))
            elif data_path.endswith(".npy"):
                data = np.load(data_path, allow_pickle=True)
                samples += list(map(lambda x: repr(x)[1:-1], data))
                labels += [0] * len(data)
                self.num_samples_per_dataset.append(len(data))

        self.per_dataset_indices = ConcatIndices(self.num_samples_per_dataset)
        self.sample_weights = None
        if self.datasets_weights is not None:
            self.sample_weights = [dw / ns for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset) 
                                 for _ in range(ns)]
        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        sample = self.samples[index]
        loaded_sample = False
        while not loaded_sample:
            if not isinstance(sample, str):
                logger.warning("Invalid sample.")
            else:
                if sample.split(".")[-1].lower() in ("txt"):
                    loaded_sample = self.get_item_frames(index)
                elif sample.split(".")[-1].lower() in ("jpg", "png", "jpeg"):
                    loaded_sample = self.get_item_image(index)
                else:
                    loaded_sample = self.get_item_video(index)  # 保留视频加载逻辑

            if not loaded_sample:
                index = np.random.randint(self.__len__())
                sample = self.samples[index]
        return loaded_sample

    def get_item_frames(self, index):
        sample = self.samples[index]
        dataset_idx, _ = self.per_dataset_indices[index]
        frames_per_clip = self.dataset_fpcs[dataset_idx]

        buffer, clip_indices = self.load_frames_from_txt(sample, frames_per_clip)
        if len(buffer) == 0:
            return None  # 加载失败时返回None

        label = self.labels[index]

        def split_into_clips(video):
            fpc = frames_per_clip
            nc = self.num_clips
            return [video[i * fpc : (i + 1) * fpc] for i in range(nc)]

        # 应用转换（保持与原代码一致的处理流程）
        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        buffer = split_into_clips(buffer)
        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]
        
        return buffer, label, clip_indices  # 返回列表形式的clip，与原代码一致

    def get_item_image(self, index):
        sample = self.samples[index]
        dataset_idx, _ = self.per_dataset_indices[index]
        fpc = self.dataset_fpcs[dataset_idx]

        try:
            image_tensor = torchvision.io.read_image(sample, torchvision.io.ImageReadMode.RGB)
        except Exception:
            return None
        label = self.labels[index]
        clip_indices = [np.arange(0, fpc, dtype=np.int32)]

        # 扩展图像维度为[T, H, W, 3]
        buffer = image_tensor.unsqueeze(0).repeat(fpc, 1, 1, 1).permute(0, 2, 3, 1)
        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        if self.transform is not None:
            buffer = [self.transform(buffer)]
        
        return buffer, label, clip_indices

    def load_frames_from_txt(self, txt_path, fpc):
        """从txt文件加载帧并保持与原视频加载一致的buffer结构"""
        if not os.path.exists(txt_path):
            warnings.warn(f"txt path not found: {txt_path}")
            return [], []

        try:
            with open(txt_path, 'r') as f:
                frame_paths = [line.strip() for line in f if line.strip()]
        except Exception as e:
            warnings.warn(f"读取txt文件失败: {txt_path}, 错误: {e}")
            return [], []

        # 过滤过短/过长片段
        if self.filter_short_videos and len(frame_paths) < fpc:
            warnings.warn(f"片段过短，跳过: {txt_path} (帧数: {len(frame_paths)})")
            return [], []
        if len(frame_paths) > self.filter_long_videos:
            warnings.warn(f"片段过长，跳过: {txt_path} (帧数: {len(frame_paths)})")
            return [], []

        fstp = self.frame_step  # 现在fstp一定是int类型
        clip_len = fpc * fstp
        clip_len = min(clip_len, len(frame_paths))  # 确保不超出实际帧数

        all_indices, clip_indices = [], []
        partition_len = len(frame_paths) // self.num_clips if self.num_clips > 0 else len(frame_paths)

        for i in range(self.num_clips):
            if partition_len > clip_len:
                # 随机采样窗口
                end_idx = np.random.randint(clip_len, partition_len) if self.random_clip_sampling else clip_len
                start_idx = end_idx - clip_len
                indices = np.linspace(start_idx, end_idx-1, fpc).astype(np.int64)
                indices = np.clip(indices, start_idx, end_idx-1)
                indices += i * partition_len
            else:
                # 处理短片段
                if not self.allow_clip_overlap and len(frame_paths) < fpc:
                    indices = np.concatenate([
                        np.linspace(0, partition_len-1, partition_len//fstp, dtype=np.int64),
                        np.ones(fpc - partition_len//fstp, dtype=np.int64) * (partition_len-1)
                    ])
                    indices += i * partition_len
                else:
                    sample_len = min(clip_len, len(frame_paths)) - 1
                    indices = np.linspace(0, sample_len, sample_len//fstp, dtype=np.int64)
                    if len(indices) < fpc:
                        indices = np.concatenate([indices, np.ones(fpc - len(indices), dtype=np.int64) * sample_len])
                    clip_step = (len(frame_paths) - clip_len) // (self.num_clips - 1) if self.num_clips > 1 else 0
                    indices += i * clip_step
            clip_indices.append(indices)
            all_indices.extend(indices)

        # 加载图像并构建buffer [T, H, W, 3]
        buffer = []
        for idx in all_indices:
            idx = min(idx, len(frame_paths)-1)
            frame_path = frame_paths[idx]
            if not os.path.exists(frame_path):
                buffer.append(np.zeros((224, 224, 3), dtype=np.uint8))
                continue
            try:
                img = Image.open(frame_path).convert('RGB')
                buffer.append(np.array(img))
            except Exception:
                buffer.append(np.zeros((224, 224, 3), dtype=np.uint8))
        return np.array(buffer), clip_indices

    def __len__(self):
        return len(self.samples)



if __name__ == "__main__":
    # 示例用法，确保frame_step被正确传递
    data = SurgicalVideoDataset(
        data_paths=["/data/wjl/vjepa2/data_process/fixed_64frames_clips.csv"],
        datasets_weights=[1.0],
        frames_per_clip=16,
        fps=None,
        dataset_fpcs=None,
        frame_step=1,  # 显式指定frame_step
        num_clips=2,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=True,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,  # duration in seconds
    )
    print(len(data))
    
    for item in data:
        import ipdb; ipdb.set_trace()
        print(item)
       