from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.depth_estimation_frozen.losses import DepthCompositeLoss
from evals.depth_estimation_frozen.metrics import compute_depth_metrics
from src.datasets.data_manager import init_data
from src.models.depth import SurgicalDepthHead


def _load_fixture_batch(batch_size: int = 2):
    fixture_dir = PROJECT_ROOT / "tests" / "fixtures" / "c3vd_depth"
    valid_meta = fixture_dir / "metadata_valid.csv"
    data_loader, _ = init_data(
        batch_size=batch_size,
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
    return next(iter(data_loader))


def _run_normal_path():
    images, depth_targets, validity_masks = _load_fixture_batch()

    batch_size, _, height, width = images.shape
    patch_size = 1
    embed_dim = 32
    num_tokens = (height // patch_size) * (width // patch_size)

    synthetic_features = torch.randn(batch_size, num_tokens, embed_dim, dtype=torch.float32)
    head = SurgicalDepthHead(
        embed_dim=embed_dim,
        decoder_channels=(64, 32),
        patch_size=patch_size,
        target_size=(height, width),
        num_feature_levels=1,
        activation="softplus",
    )

    preds = head(synthetic_features)
    criterion = DepthCompositeLoss(l1_weight=1.0, silog_weight=0.0, grad_weight=0.0, ssi_weight=0.0)
    loss, loss_components = criterion(preds, depth_targets, validity_masks, images=images)
    metrics = compute_depth_metrics(preds.detach(), depth_targets, validity_masks, min_depth=1e-3, max_depth=None)

    print(f"pred.shape={tuple(preds.shape)}")
    print(f"loss={loss.item():.6f}")
    print(f"loss_components={sorted(loss_components.keys())}")
    print(f"metrics={{{', '.join(f'{k}: {float(v):.6f}' for k, v in sorted(metrics.items()))}}}")


def _run_invalid_mask_guard_path():
    images, depth_targets, validity_masks = _load_fixture_batch()
    empty_mask = torch.zeros_like(validity_masks)

    criterion = DepthCompositeLoss(l1_weight=1.0, silog_weight=0.0, grad_weight=0.0, ssi_weight=0.0)
    preds = torch.ones_like(depth_targets)

    try:
        criterion(preds, depth_targets, empty_mask, images=images)
    except ValueError as exc:
        print(f"empty_mask_loss_guard={exc.__class__.__name__}: {exc}")
    else:
        raise RuntimeError("Expected empty-mask loss guard to raise ValueError.")

    try:
        compute_depth_metrics(preds, depth_targets, empty_mask, min_depth=1e-3, max_depth=None)
    except ValueError as exc:
        print(f"empty_mask_metric_guard={exc.__class__.__name__}: {exc}")
    else:
        raise RuntimeError("Expected empty-mask metric guard to raise ValueError.")


def main():
    _run_normal_path()
    _run_invalid_mask_guard_path()


if __name__ == "__main__":
    main()
