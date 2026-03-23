import json
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torchvision.transforms import functional as tvf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.segmentation_frozen.eval import CosineWDSchedule, WarmupCosineLRSchedule, run_one_epoch
from evals.segmentation_frozen.losses import BinarySegmentationLoss
from evals.segmentation_frozen.metrics import compute_binary_segmentation_metrics
from src.datasets.data_manager import init_data
from src.models.segmentation import MaskFormerSegmentationHead


class _SegPairTransform:
    def __init__(self, size=32):
        self.size = (int(size), int(size))
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image_pil, mask_tensor):
        image = tvf.to_tensor(image_pil)
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=self.size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        image = tvf.normalize(image, mean=self.mean, std=self.std)

        mask = mask_tensor.float()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=self.size, mode="nearest").squeeze(0)
        mask = (mask > 0.5).float()
        return image, mask


class _DummyFrozenSegEncoder(nn.Module):
    def __init__(self, embed_dim=32, patch_size=16):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(3, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        tokens = self.proj(x)
        return tokens.flatten(2).transpose(1, 2).contiguous()


def _fixture_root() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / "cvc12k_seg"


def _build_loader(batch_size=2):
    fixture_root = _fixture_root()
    pair_file = fixture_root / "pairs_train.txt"
    transform = _SegPairTransform(size=32)
    data_loader, _ = init_data(
        batch_size=batch_size,
        transform=transform,
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
    return data_loader


def _run_one_pass_flow():
    device = torch.device("cpu")
    loader = _build_loader(batch_size=2)

    encoder = _DummyFrozenSegEncoder(embed_dim=32, patch_size=16).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    head = MaskFormerSegmentationHead(
        embed_dim=32,
        patch_size=16,
        target_size=32,
        num_feature_levels=1,
        hidden_dim=64,
        mask_dim=64,
        num_queries=1,
        num_decoder_layers=2,
        nheads=4,
        dim_feedforward=128,
        decoder_channels=(64, 32),
        return_aux=True,
    ).to(device)

    criterion = BinarySegmentationLoss(structure_weight=0.5, aux_weight=0.25)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = WarmupCosineLRSchedule(
        optimizer=optimizer,
        iterations_per_epoch=len(loader),
        num_epochs=1,
        start_lr=1e-3,
        ref_lr=1e-3,
        final_lr=5e-4,
        warmup_ratio=0.0,
    )
    wd_scheduler = CosineWDSchedule(
        optimizer=optimizer,
        iterations_per_epoch=len(loader),
        num_epochs=1,
        ref_wd=0.01,
        final_wd=0.005,
    )

    loss, metrics, comps, _ = run_one_epoch(
        device=device,
        encoder=encoder,
        head=head,
        criterion=criterion,
        data_loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        wd_scheduler=wd_scheduler,
        scaler=None,
        use_bfloat16=False,
        training=True,
        global_step=0,
        writer=None,
        phase="smoke_train",
        log_interval=1,
        metrics_threshold=0.5,
    )

    if not torch.isfinite(torch.tensor(loss)):
        raise ValueError(f"Expected finite training loss, got {loss}")
    required_metric_keys = {"iou", "dice", "precision", "recall", "specificity", "pixel_acc", "mae"}
    if not required_metric_keys.issubset(set(metrics.keys())):
        raise ValueError(f"Missing metrics keys: {sorted(required_metric_keys - set(metrics.keys()))}")
    for key in required_metric_keys:
        if not torch.isfinite(torch.tensor(float(metrics[key]))):
            raise ValueError(f"Metric {key} is not finite: {metrics[key]}")

    batch = next(iter(loader))
    if set(batch.keys()) < {"image", "mask"}:
        raise ValueError(f"Expected CVC-12K batch dict keys image/mask, got keys={list(batch.keys())}")

    return {
        "loss": float(loss),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "loss_components": {k: float(v) for k, v in comps.items()},
        "batch_image_shape": [int(v) for v in batch["image"].shape],
        "batch_mask_shape": [int(v) for v in batch["mask"].shape],
    }


def _run_all_background_guard():
    logits = torch.full((2, 1, 32, 32), fill_value=-12.0, dtype=torch.float32)
    all_bg_target = torch.zeros((2, 1, 32, 32), dtype=torch.float32)

    metrics = compute_binary_segmentation_metrics({"logits": logits}, all_bg_target, threshold=0.5)
    for key, value in metrics.items():
        if not torch.isfinite(value):
            raise ValueError(f"All-background guard failed: metric {key} is non-finite ({value})")

    if abs(float(metrics["iou"]) - 1.0) > 1e-6:
        raise ValueError(f"Expected IoU=1.0 for empty prediction on all-background target, got {float(metrics['iou'])}")
    if abs(float(metrics["dice"]) - 1.0) > 1e-6:
        raise ValueError(f"Expected Dice=1.0 for empty prediction on all-background target, got {float(metrics['dice'])}")

    criterion = BinarySegmentationLoss()
    total_loss, _ = criterion({"logits": logits}, all_bg_target)
    if not torch.isfinite(total_loss):
        raise ValueError(f"All-background loss is non-finite: {total_loss}")

    return {
        "iou": float(metrics["iou"]),
        "dice": float(metrics["dice"]),
        "loss": float(total_loss),
    }


def main():
    forward_stats = _run_one_pass_flow()
    bg_guard = _run_all_background_guard()
    print("SEG_FORWARD_SMOKE", json.dumps(forward_stats, sort_keys=True))
    print("SEG_ALL_BACKGROUND_GUARD", json.dumps(bg_guard, sort_keys=True))


if __name__ == "__main__":
    main()
