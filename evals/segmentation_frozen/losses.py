from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    bsz = probs.shape[0]
    probs = probs.view(bsz, -1)
    targets = targets.view(bsz, -1)
    targets = (targets > 0.5).float()
    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return (1.0 - dice).mean()


def structure_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    kernel_size: int = 31,
    weight_factor: float = 5.0,
    smooth: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    if kernel_size <= 1:
        raise ValueError(f"kernel_size must be > 1, got {kernel_size}.")
    if logits.ndim != 4 or targets.ndim != 4:
        raise ValueError(f"Expected logits/targets as [B,1,H,W]. Got {logits.shape=} {targets.shape=}.")
    if logits.shape[1] != 1 or targets.shape[1] != 1:
        raise ValueError(f"Expected logits/targets channel=1 for binary segmentation. Got {logits.shape=} {targets.shape=}.")

    logits_f = logits.float()
    targets_f = (targets > 0.5).float()

    pad = kernel_size // 2
    avg = F.avg_pool2d(targets_f, kernel_size=kernel_size, stride=1, padding=pad)
    weight = 1.0 + float(weight_factor) * torch.abs(avg - targets_f)

    bce = F.binary_cross_entropy_with_logits(logits_f, targets_f, reduction="none")
    wbce = (weight * bce).sum(dim=(2, 3)) / torch.clamp(weight.sum(dim=(2, 3)), min=eps)

    probs = torch.sigmoid(logits_f)
    inter = ((probs * targets_f) * weight).sum(dim=(2, 3))
    union = ((probs + targets_f) * weight).sum(dim=(2, 3))
    wiou = 1.0 - (inter + smooth) / (union - inter + smooth)

    return (wbce + wiou).mean()


def _unpack_predictions(preds: torch.Tensor | Dict[str, torch.Tensor | List[torch.Tensor]]):
    if isinstance(preds, torch.Tensor):
        return preds, []
    logits = preds.get("logits")
    if logits is None:
        raise KeyError("Expected preds to contain key 'logits'.")
    aux = preds.get("aux_logits", [])
    if aux is None:
        aux = []
    if isinstance(aux, torch.Tensor):
        aux = [aux]
    return logits, list(aux)


class BinarySegmentationLoss(nn.Module):
    def __init__(
        self,
        *,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        structure_weight: float = 0.0,
        structure_kernel_size: int = 31,
        structure_weight_factor: float = 5.0,
        structure_smooth: float = 1.0,
        aux_weight: float = 0.0,
        pos_weight: float | None = None,
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.structure_weight = float(structure_weight)
        self.structure_kernel_size = int(structure_kernel_size)
        self.structure_weight_factor = float(structure_weight_factor)
        self.structure_smooth = float(structure_smooth)
        self.aux_weight = float(aux_weight)
        self.register_buffer(
            "_pos_weight",
            torch.tensor([float(pos_weight)], dtype=torch.float32) if pos_weight is not None else None,
            persistent=False,
        )
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight if pos_weight is not None else None)

    def forward(
        self,
        preds: torch.Tensor | Dict[str, torch.Tensor | List[torch.Tensor]],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits, aux_logits = _unpack_predictions(preds)
        targets = (targets > 0.5).float()

        loss_bce = self.bce(logits, targets)
        loss_dice = dice_loss_from_logits(logits, targets)
        if self.structure_weight > 0.0:
            loss_struct = structure_loss_from_logits(
                logits,
                targets,
                kernel_size=self.structure_kernel_size,
                weight_factor=self.structure_weight_factor,
                smooth=self.structure_smooth,
            )
        else:
            loss_struct = logits.new_tensor(0.0)

        total = self.bce_weight * loss_bce + self.dice_weight * loss_dice + self.structure_weight * loss_struct

        comps: Dict[str, torch.Tensor] = {
            "bce": loss_bce.detach(),
            "dice": loss_dice.detach(),
            "structure": loss_struct.detach(),
        }

        if self.aux_weight > 0.0 and len(aux_logits) > 0:
            aux_losses = []
            for aux in aux_logits:
                lb = self.bce(aux, targets)
                ld = dice_loss_from_logits(aux, targets)
                if self.structure_weight > 0.0:
                    ls = structure_loss_from_logits(
                        aux,
                        targets,
                        kernel_size=self.structure_kernel_size,
                        weight_factor=self.structure_weight_factor,
                        smooth=self.structure_smooth,
                    )
                else:
                    ls = aux.new_tensor(0.0)
                aux_losses.append(self.bce_weight * lb + self.dice_weight * ld + self.structure_weight * ls)
            aux_loss = torch.stack(aux_losses).mean()
            total = total + self.aux_weight * aux_loss
            comps["aux"] = aux_loss.detach()

        return total, comps
