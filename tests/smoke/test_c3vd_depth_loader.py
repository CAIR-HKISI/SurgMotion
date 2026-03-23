from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.c3vd_depth_dataset import C3VDDepthDataset
from src.datasets.data_manager import init_data


def _print_batch_contract(batch):
    images, depth_targets, validity_masks = batch
    print(f"image.shape={tuple(images.shape)}")
    print(f"image.dtype={images.dtype}")
    print(f"depth.shape={tuple(depth_targets.shape)}")
    print(f"depth.dtype={depth_targets.dtype}")
    print(f"depth.sum={depth_targets.sum().item():.1f}")
    print(f"valid_mask.shape={tuple(validity_masks.shape)}")
    print(f"valid_mask.dtype={validity_masks.dtype}")
    print(f"valid_mask.sum={validity_masks.sum().item():.1f}")


def main():
    fixture_dir = PROJECT_ROOT / "tests" / "fixtures" / "c3vd_depth"
    valid_meta = fixture_dir / "metadata_valid.csv"
    malformed_meta = fixture_dir / "metadata_malformed.csv"

    data_loader, _ = init_data(
        batch_size=2,
        data="c3vd_depth_dataset",
        num_workers=0,
        world_size=1,
        rank=0,
        root_path=str(valid_meta),
        image_folder=str(fixture_dir),
        training=False,
        drop_last=False,
        deterministic=True,
    )

    batch = next(iter(data_loader))
    _print_batch_contract(batch)

    try:
        C3VDDepthDataset(annotation_file=str(malformed_meta), root=str(fixture_dir))
    except ValueError as error:
        print(f"malformed_metadata_error={error.__class__.__name__}: {error}")
    else:
        raise RuntimeError("Expected malformed metadata check to raise ValueError.")


if __name__ == "__main__":
    main()
