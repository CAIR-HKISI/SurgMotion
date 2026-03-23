import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.data_manager import init_data


def _fixture_root() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / "cvc12k_seg"


def _run_main_loader_contract() -> dict:
    fixture_root = _fixture_root()
    pair_file = fixture_root / "pairs_train.txt"

    data_loader, _ = init_data(
        batch_size=2,
        transform=None,
        data="cvc12k_segmentation_dataset",
        pin_mem=False,
        num_workers=0,
        world_size=1,
        rank=0,
        root_path=str(fixture_root),
        image_folder=str(pair_file),
        training=False,
        drop_last=False,
        deterministic=True,
        persistent_workers=False,
    )

    batch = next(iter(data_loader))
    image = batch["image"]
    mask = batch["mask"]

    if image.ndim != 4:
        raise ValueError(f"Expected image batch rank 4, got {image.ndim}")
    if mask.ndim != 4:
        raise ValueError(f"Expected mask batch rank 4, got {mask.ndim}")
    if mask.shape[1] != 1:
        raise ValueError(f"Expected mask channel dimension == 1, got {mask.shape[1]}")

    mask_unique = sorted(float(v) for v in torch.unique(mask).tolist())
    if not set(mask_unique).issubset({0.0, 1.0}):
        raise ValueError(f"Expected binary masks in {{0,1}}, got unique values {mask_unique}")

    return {
        "batch_size": int(image.shape[0]),
        "image_dtype": str(image.dtype),
        "image_shape": [int(v) for v in image.shape],
        "mask_dtype": str(mask.dtype),
        "mask_shape": [int(v) for v in mask.shape],
        "mask_unique": mask_unique,
    }


def _run_size_mismatch_check() -> str:
    fixture_root = _fixture_root()
    mismatch_pair_file = fixture_root / "pairs_mismatch.txt"

    data_loader, _ = init_data(
        batch_size=1,
        transform=None,
        data="cvc12k_segmentation_dataset",
        pin_mem=False,
        num_workers=0,
        world_size=1,
        rank=0,
        root_path=str(fixture_root),
        image_folder=str(mismatch_pair_file),
        training=False,
        drop_last=False,
        deterministic=True,
        persistent_workers=False,
    )

    try:
        _ = next(iter(data_loader))
    except ValueError as err:
        msg = str(err)
        if "size mismatch" not in msg:
            raise ValueError(f"Mismatch check raised unexpected error message: {msg}") from err
        return msg

    raise AssertionError("Expected a controlled image/mask size mismatch error, but loading succeeded.")


def main():
    contract = _run_main_loader_contract()
    mismatch_msg = _run_size_mismatch_check()

    print("CVC12K_SMOKE_CONTRACT", json.dumps(contract, sort_keys=True))
    print("CVC12K_MISMATCH_CHECK", mismatch_msg)


if __name__ == "__main__":
    main()
