from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import functional as F


@dataclass
class CVC12KSegmentationSample:
    image_path: str
    mask_path: str


def _as_paths(data_paths: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(data_paths, (str, Path)):
        return [str(data_paths)]
    return [str(path) for path in data_paths]


def _read_pair_file(pair_file: Union[str, Path]) -> List[CVC12KSegmentationSample]:
    pair_file = Path(pair_file)
    if not pair_file.exists():
        raise FileNotFoundError(f"Pair file not found: {pair_file}")

    samples: List[CVC12KSegmentationSample] = []
    for raw_line in pair_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if (not line) or line.startswith("#"):
            continue
        parts = [part for part in line.replace(",", " ").split(" ") if part]
        if len(parts) < 2:
            raise ValueError(
                f"Invalid line in pair file {pair_file}: '{line}'. Expected '<image_path> <mask_path>'."
            )
        samples.append(CVC12KSegmentationSample(image_path=parts[0], mask_path=parts[1]))

    return samples


class CVC12KSegmentationDataset(Dataset):
    def __init__(
        self,
        data_paths: Union[str, Sequence[str]],
        root_path: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        strict_size_match: bool = True,
    ):
        pair_files = _as_paths(data_paths)
        samples: List[CVC12KSegmentationSample] = []
        for pair_file in pair_files:
            samples.extend(_read_pair_file(pair_file))

        self.samples = samples
        self.root_path = Path(root_path) if root_path is not None else None
        self.transform = transform
        self.strict_size_match = bool(strict_size_match)

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_path(self, path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute() or self.root_path is None:
            return path
        return self.root_path / path

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_path = self._resolve_path(sample.image_path)
        mask_path = self._resolve_path(sample.mask_path)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.strict_size_match and image.size != mask.size:
            raise ValueError(
                "CVC-12K image/mask size mismatch: "
                f"image={image_path} size={image.size}, mask={mask_path} size={mask.size}"
            )

        mask_np = np.array(mask, dtype=np.uint8)
        mask_tensor = torch.from_numpy((mask_np > 0).astype(np.float32)).unsqueeze(0)

        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image, mask_tensor)
        else:
            image_tensor = F.to_tensor(image)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


def make_cvc12k_segmentation_dataset(
    data_paths,
    batch_size,
    transform=None,
    rank=0,
    world_size=1,
    collator=None,
    drop_last=False,
    num_workers=8,
    pin_mem=True,
    persistent_workers=False,
    training=True,
    deterministic=True,
    root_path=None,
    strict_size_match=True,
):
    if not deterministic:
        raise ValueError("cvc12k_segmentation_dataset currently supports deterministic DataLoader only.")

    dataset = CVC12KSegmentationDataset(
        data_paths=data_paths,
        root_path=root_path,
        transform=transform,
        strict_size_match=strict_size_match,
    )

    dist_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=bool(training))
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=dist_sampler,
        collate_fn=collator,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    return dataset, data_loader, dist_sampler
