from __future__ import annotations

import torch

__all__ = [
    "compute_depth_metrics",
    "compute_depth_metrics_median_scaled",
]


def _validate_mask(mask: torch.Tensor, *, threshold: float = 0.5) -> torch.Tensor:
    valid = mask > threshold
    if not valid.any():
        raise ValueError("Depth metrics received an empty/invalid mask (no valid pixels).")
    return valid


def _compute_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, min_depth: float, max_depth: float | None):
    eps = 1e-6
    pred = torch.clamp(pred, min=min_depth)
    target = torch.clamp(target, min=min_depth)
    if max_depth is not None:
        pred = torch.clamp(pred, max=max_depth)
        target = torch.clamp(target, max=max_depth)

    valid = _validate_mask(mask)
    denom = torch.clamp(valid.sum(), min=eps)

    diff = pred - target
    abs_diff = torch.abs(diff)
    abs_rel = ((abs_diff / torch.clamp(target, min=eps)) * valid).sum() / denom

    sq_rel = (((diff**2) / torch.clamp(target, min=eps)) * valid).sum() / denom
    sq_rel = sq_rel * 1000.0

    rmse = torch.sqrt(((diff**2) * valid).sum() / denom)

    ratio = torch.max(
        pred / torch.clamp(target, min=eps),
        target / torch.clamp(pred, min=eps),
    )
    delta1 = ((ratio < 1.25).float().mul(valid).sum()) / denom
    delta1_1 = ((ratio < 1.1).float().mul(valid).sum()) / denom

    return {
        "abs_rel": abs_rel,
        "rmse": rmse,
        "sq_rel": sq_rel,
        "delta1": delta1,
        "delta1_1": delta1_1,
    }


def compute_depth_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    min_depth: float = 1e-3,
    max_depth: float | None = None,
):
    return _compute_metrics(pred, target, mask, min_depth=min_depth, max_depth=max_depth)


def compute_depth_metrics_median_scaled(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    min_depth: float = 1e-3,
    max_depth: float | None = None,
):
    eps = 1e-6
    valid = _validate_mask(mask)
    batch_size = pred.shape[0]
    scaled = []
    for b in range(batch_size):
        p = pred[b]
        t = target[b]
        m = valid[b]
        if m.any():
            ratio = t[m] / torch.clamp(p[m], min=eps)
            scale = torch.median(ratio)
        else:
            scale = torch.tensor(1.0, device=pred.device, dtype=pred.dtype)
        scaled.append(p * scale)
    pred_scaled = torch.stack(scaled, dim=0)
    return _compute_metrics(pred_scaled, target, valid.float(), min_depth=min_depth, max_depth=max_depth)
