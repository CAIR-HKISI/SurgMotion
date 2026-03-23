import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


@dataclass
class C3VDSample:
    rgb_path: str
    depth_path: str
    mask_path: Optional[str] = None


def _resolve_path(path: str) -> str:
    return os.path.expanduser(str(path))


def _pick_column(fieldnames, candidates):
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    return None


def _read_c3vd_metadata(annotation_file: str):
    annotation_path = Path(_resolve_path(annotation_file))
    if not annotation_path.exists():
        raise FileNotFoundError(f"C3VD metadata file not found: {annotation_path}")

    with open(annotation_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"C3VD metadata {annotation_path} has no header row.")

        rgb_col = _pick_column(reader.fieldnames, ["rgb_path", "image_path", "rgb", "image", "frame", "frame_path"])
        depth_col = _pick_column(reader.fieldnames, ["depth_path", "depth", "depth_map", "depthmap", "depth_path_png"])
        mask_col = _pick_column(reader.fieldnames, ["mask_path", "mask"])

        if rgb_col is None or depth_col is None:
            raise ValueError(
                f"C3VD metadata {annotation_path} must contain RGB and depth columns. "
                f"Found columns: {reader.fieldnames}"
            )

        samples = []
        for row_idx, row in enumerate(reader, start=2):
            rgb_path = str(row.get(rgb_col, "")).strip()
            depth_path = str(row.get(depth_col, "")).strip()
            if not rgb_path or not depth_path:
                raise ValueError(
                    f"C3VD metadata {annotation_path} has missing RGB/depth path at CSV line {row_idx}."
                )

            mask_path = None
            if mask_col is not None:
                value = str(row.get(mask_col, "")).strip()
                mask_path = value if value else None

            samples.append(C3VDSample(rgb_path=rgb_path, depth_path=depth_path, mask_path=mask_path))

    if not samples:
        raise ValueError(f"C3VD metadata {annotation_path} contains zero samples.")
    return samples


class C3VDDepthDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        annotation_file,
        root=None,
        transform=None,
        depth_scale=1.0,
        min_depth=0.0,
        max_depth=None,
        invalid_depth=0.0,
        return_paths=False,
    ):
        self.samples = _read_c3vd_metadata(annotation_file)
        self.root = _resolve_path(root) if root else None
        self.transform = transform
        self.depth_scale = float(depth_scale)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth) if max_depth is not None else None
        self.invalid_depth = float(invalid_depth) if invalid_depth is not None else None
        self.return_paths = bool(return_paths)

    def __len__(self):
        return len(self.samples)

    def _make_abs_path(self, path):
        path = _resolve_path(path)
        if os.path.isabs(path):
            return path
        if self.root is None:
            return path
        return os.path.join(self.root, path)

    def _load_depth(self, path):
        if path.endswith(".npy"):
            depth = np.load(path).astype(np.float32)
        else:
            with Image.open(path) as depth_im:
                depth = np.array(depth_im, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth *= self.depth_scale
        return depth

    def __getitem__(self, index):
        sample = self.samples[index]
        rgb_path = self._make_abs_path(sample.rgb_path)
        depth_path = self._make_abs_path(sample.depth_path)

        image = Image.open(rgb_path).convert("RGB")
        image_tensor = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1) / 255.0

        depth = self._load_depth(depth_path)
        valid = np.isfinite(depth)
        valid &= depth > max(self.min_depth, 1e-6)
        if self.max_depth is not None:
            valid &= depth <= self.max_depth
        if self.invalid_depth is not None:
            valid &= ~np.isclose(depth, self.invalid_depth)

        depth = np.where(valid, depth, 0.0).astype(np.float32)
        valid_mask = valid.astype(np.float32)

        depth_tensor = torch.from_numpy(depth).unsqueeze(0)
        valid_tensor = torch.from_numpy(valid_mask).unsqueeze(0)

        if sample.mask_path:
            mask_path = self._make_abs_path(sample.mask_path)
            with Image.open(mask_path) as mask_im:
                mask_arr = np.array(mask_im, dtype=np.float32)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[..., 0]
            extra_mask = torch.from_numpy((mask_arr > 0).astype(np.float32)).unsqueeze(0)
            valid_tensor = valid_tensor * extra_mask
            depth_tensor = depth_tensor * valid_tensor

        if self.transform is not None:
            image_tensor, depth_tensor, valid_tensor = self.transform(image_tensor, depth_tensor, valid_tensor)

        if self.return_paths:
            return image_tensor, depth_tensor, valid_tensor, rgb_path, depth_path
        return image_tensor, depth_tensor, valid_tensor


def make_c3vd_depth_dataset(
    data_paths,
    batch_size,
    dataset_root=None,
    transform=None,
    rank=0,
    world_size=1,
    collator=None,
    drop_last=True,
    num_workers=8,
    pin_mem=True,
    persistent_workers=False,
    deterministic=True,
    training=True,
    depth_scale=1.0,
    min_depth=0.0,
    max_depth=None,
    invalid_depth=0.0,
    return_paths=False,
):
    if isinstance(data_paths, (list, tuple)):
        if len(data_paths) != 1:
            raise ValueError("c3vd_depth_dataset expects exactly one metadata CSV path.")
        annotation_file = data_paths[0]
    else:
        annotation_file = data_paths

    dataset = C3VDDepthDataset(
        annotation_file=annotation_file,
        root=dataset_root,
        transform=transform,
        depth_scale=depth_scale,
        min_depth=min_depth,
        max_depth=max_depth,
        invalid_depth=invalid_depth,
        return_paths=return_paths,
    )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=bool(training),
    )

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

    return dataset, data_loader, dist_sampler
