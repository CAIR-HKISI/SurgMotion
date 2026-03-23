from __future__ import annotations

from typing import Dict, List

import torch


def _unpack_logits(preds: torch.Tensor | Dict[str, torch.Tensor | List[torch.Tensor]]) -> torch.Tensor:
    if isinstance(preds, torch.Tensor):
        return preds
    logits = preds.get("logits")
    if logits is None:
        raise KeyError("Expected preds to contain key 'logits'.")
    return logits


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor, eps: float) -> torch.Tensor:
    return numerator / torch.clamp(denominator, min=eps)


def compute_binary_segmentation_metrics(
    preds: torch.Tensor | Dict[str, torch.Tensor | List[torch.Tensor]],
    targets: torch.Tensor,
    *,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    logits = _unpack_logits(preds)
    probs = torch.sigmoid(logits)
    pred = (probs > float(threshold)).float()
    target = (targets > 0.5).float()

    dims = (1, 2, 3)
    mae = torch.abs(probs - target).mean(dim=dims)
    tp = (pred * target).sum(dim=dims)
    fp = (pred * (1.0 - target)).sum(dim=dims)
    fn = ((1.0 - pred) * target).sum(dim=dims)
    tn = ((1.0 - pred) * (1.0 - target)).sum(dim=dims)

    target_pos = target.sum(dim=dims)
    pred_pos = pred.sum(dim=dims)

    iou = _safe_ratio(tp, tp + fp + fn, eps)
    dice = _safe_ratio(2.0 * tp, 2.0 * tp + fp + fn, eps)
    precision = _safe_ratio(tp, tp + fp, eps)
    recall = _safe_ratio(tp, tp + fn, eps)
    specificity = _safe_ratio(tn, tn + fp, eps)
    acc = _safe_ratio(tp + tn, tp + tn + fp + fn, eps)

    empty_target = target_pos <= eps
    empty_pred = pred_pos <= eps
    perfect_empty = (empty_target & empty_pred).float()

    iou = torch.where(empty_target, perfect_empty, iou)
    dice = torch.where(empty_target, perfect_empty, dice)
    recall = torch.where(empty_target, torch.ones_like(recall), recall)
    precision = torch.where(empty_pred, torch.ones_like(precision), precision)

    return {
        "iou": iou.mean(),
        "dice": dice.mean(),
        "precision": precision.mean(),
        "recall": recall.mean(),
        "sensitivity": recall.mean(),
        "specificity": specificity.mean(),
        "pixel_acc": acc.mean(),
        "mae": mae.mean(),
    }
