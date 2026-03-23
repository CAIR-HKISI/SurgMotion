from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_valid_mask(mask: torch.Tensor, *, threshold: float = 0.5) -> torch.Tensor:
    valid = mask > threshold
    if not valid.any():
        raise ValueError("Depth loss received an empty/invalid mask (no valid pixels).")
    return valid.float()


def _masked_mean(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    denom = torch.clamp(mask.sum(), min=eps)
    return (value * mask).sum() / denom


def _to_scales(scales: Iterable[float] | None) -> Tuple[float, ...]:
    if scales is None:
        return (1.0,)
    scales = tuple(float(s) for s in scales)
    if len(scales) == 0:
        return (1.0,)
    return scales


class ScaleInvariantLoss(nn.Module):
    def __init__(self, lam: float = 0.15, eps: float = 1e-6):
        super().__init__()
        self.lam = lam
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(pred, min=self.eps)
        target = torch.clamp(target, min=self.eps)
        diff = torch.log(pred) - torch.log(target)
        diff = diff * mask
        denom = torch.clamp(mask.sum(), min=self.eps)
        mean = diff.sum() / denom
        mean_sq = (diff**2).sum() / denom
        return torch.sqrt(torch.clamp(mean_sq - self.lam * (mean**2), min=0.0))


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        def masked_mean(x, m):
            num = torch.clamp(m.sum(dim=(1, 2, 3), keepdim=True), min=self.eps)
            return (x * m).sum(dim=(1, 2, 3), keepdim=True) / num

        mu_p = masked_mean(pred, mask)
        mu_t = masked_mean(target, mask)

        p_centered = (pred - mu_p) * mask
        t_centered = (target - mu_t) * mask

        num_s = (p_centered * t_centered).sum(dim=(1, 2, 3), keepdim=True)
        den_s = (p_centered**2).sum(dim=(1, 2, 3), keepdim=True)

        s = num_s / torch.clamp(den_s, min=self.eps)
        b = mu_t - s * mu_p

        aligned_pred = s * pred + b
        diff = (aligned_pred - target) * mask

        num_pixels = torch.clamp(mask.sum(dim=(1, 2, 3), keepdim=True), min=self.eps)
        loss_per_sample = (diff**2).sum(dim=(1, 2, 3), keepdim=True) / num_pixels
        return loss_per_sample.mean()


class GradientLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        def gradients(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            dx = torch.abs(x[..., :, 1:] - x[..., :, :-1])
            dy = torch.abs(x[..., 1:, :] - x[..., :-1, :])
            return dx, dy

        pred_dx, pred_dy = gradients(pred)
        target_dx, target_dy = gradients(target)
        mask_dx = mask[..., :, 1:] * mask[..., :, :-1]
        mask_dy = mask[..., 1:, :] * mask[..., :-1, :]
        loss_x = _masked_mean(torch.abs(pred_dx - target_dx), mask_dx, eps=self.eps)
        loss_y = _masked_mean(torch.abs(pred_dy - target_dy), mask_dy, eps=self.eps)
        return loss_x + loss_y


class EdgeAwareSmoothnessLoss(nn.Module):
    def __init__(
        self,
        *,
        scales: Iterable[float] | None = (1.0,),
        image_grad_scale: float = 10.0,
        depth_mode: str = "depth",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.scales = _to_scales(scales)
        self.image_grad_scale = float(image_grad_scale)
        self.depth_mode = depth_mode
        self.eps = float(eps)

    def _prep_depth(self, depth: torch.Tensor) -> torch.Tensor:
        if self.depth_mode == "depth":
            return depth
        if self.depth_mode == "inv_depth":
            return 1.0 / torch.clamp(depth, min=self.eps)
        if self.depth_mode == "log_depth":
            return torch.log(torch.clamp(depth, min=self.eps))
        raise ValueError(f"Unsupported depth_mode='{self.depth_mode}'.")

    def forward(self, pred: torch.Tensor, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if images is None:
            raise ValueError("EdgeAwareSmoothnessLoss requires 'images' but got None.")

        pred = self._prep_depth(pred)
        total = pred.new_tensor(0.0)
        for s in self.scales:
            if s == 1.0:
                p, img, m = pred, images, mask
            else:
                size = (max(1, int(pred.shape[-2] * s)), max(1, int(pred.shape[-1] * s)))
                p = F.interpolate(pred, size=size, mode="bilinear", align_corners=False)
                img = F.interpolate(images, size=size, mode="bilinear", align_corners=False)
                m = F.interpolate(mask, size=size, mode="nearest")

            p_dx = torch.abs(p[..., :, 1:] - p[..., :, :-1])
            p_dy = torch.abs(p[..., 1:, :] - p[..., :-1, :])

            i_dx = torch.mean(torch.abs(img[..., :, 1:] - img[..., :, :-1]), dim=1, keepdim=True)
            i_dy = torch.mean(torch.abs(img[..., 1:, :] - img[..., :-1, :]), dim=1, keepdim=True)
            w_x = torch.exp(-self.image_grad_scale * i_dx)
            w_y = torch.exp(-self.image_grad_scale * i_dy)

            m_dx = m[..., :, 1:] * m[..., :, :-1]
            m_dy = m[..., 1:, :] * m[..., :-1, :]
            total = total + _masked_mean(p_dx * w_x, m_dx, eps=self.eps) + _masked_mean(p_dy * w_y, m_dy, eps=self.eps)

        return total / float(len(self.scales))


class DepthCompositeLoss(nn.Module):
    def __init__(
        self,
        l1_weight: float = 1.0,
        silog_weight: float = 0.15,
        grad_weight: float = 0.1,
        ssi_weight: float = 0.0,
        data_term: str = "l1",
        huber_delta: float = 0.01,
        charbonnier_eps: float = 1e-3,
        berhu_threshold: float = 0.2,
        use_edge_aware_smoothness: bool = False,
        smooth_scales: Iterable[float] | None = (1.0,),
        smooth_image_grad_scale: float = 10.0,
        smooth_depth_mode: str = "depth",
        weight_schedules: dict | None = None,
    ):
        super().__init__()
        self.l1_weight = float(l1_weight)
        self.silog_weight = float(silog_weight)
        self.grad_weight = float(grad_weight)
        self.ssi_weight = float(ssi_weight)

        self._cur_l1_weight = float(l1_weight)
        self._cur_silog_weight = float(silog_weight)
        self._cur_grad_weight = float(grad_weight)
        self._cur_ssi_weight = float(ssi_weight)

        self.weight_schedules = weight_schedules or {}
        self.data_term = data_term
        self.huber_delta = huber_delta
        self.charbonnier_eps = charbonnier_eps
        self.berhu_threshold = berhu_threshold
        self.use_edge_aware_smoothness = use_edge_aware_smoothness

        self.silog = ScaleInvariantLoss()
        self.grad = GradientLoss()
        self.ssi = ScaleAndShiftInvariantLoss()
        self.edge_smooth = EdgeAwareSmoothnessLoss(
            scales=smooth_scales,
            image_grad_scale=smooth_image_grad_scale,
            depth_mode=smooth_depth_mode,
        )

    @staticmethod
    def _schedule_value(sched: dict, *, epoch: int, num_epochs: int, default: float) -> float:
        if not isinstance(sched, dict):
            return default
        typ = str(sched.get("type", "constant")).lower()
        start = float(sched.get("start", default))
        end = float(sched.get("end", start))
        start_epoch = float(sched.get("start_epoch", 0))
        end_epoch = float(sched.get("end_epoch", num_epochs))

        if end_epoch <= start_epoch:
            return end

        e = float(epoch)
        if e <= start_epoch:
            t = 0.0
        elif e >= end_epoch:
            t = 1.0
        else:
            t = (e - start_epoch) / (end_epoch - start_epoch)

        if typ == "constant":
            return start
        if typ == "linear":
            return start + (end - start) * t
        if typ == "cosine":
            import math

            return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * t))
        raise ValueError(f"Unsupported schedule type '{typ}'.")

    def set_epoch(self, epoch: int, num_epochs: int) -> Dict[str, float]:
        self._cur_l1_weight = self._schedule_value(
            self.weight_schedules.get("l1_weight", None), epoch=epoch, num_epochs=num_epochs, default=self.l1_weight
        )
        self._cur_silog_weight = self._schedule_value(
            self.weight_schedules.get("silog_weight", None), epoch=epoch, num_epochs=num_epochs, default=self.silog_weight
        )
        self._cur_grad_weight = self._schedule_value(
            self.weight_schedules.get("grad_weight", None), epoch=epoch, num_epochs=num_epochs, default=self.grad_weight
        )
        self._cur_ssi_weight = self._schedule_value(
            self.weight_schedules.get("ssi_weight", None), epoch=epoch, num_epochs=num_epochs, default=self.ssi_weight
        )
        return {
            "l1_weight": self._cur_l1_weight,
            "silog_weight": self._cur_silog_weight,
            "grad_weight": self._cur_grad_weight,
            "ssi_weight": self._cur_ssi_weight,
        }

    def _data_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Tuple[str, torch.Tensor]:
        eps = 1e-6
        diff = pred - target
        abs_diff = torch.abs(diff)
        term = self.data_term

        if term == "l1":
            return "l1", _masked_mean(abs_diff, mask, eps=eps)
        if term == "huber":
            return "huber", _masked_mean(F.smooth_l1_loss(pred, target, reduction="none", beta=self.huber_delta), mask, eps=eps)
        if term == "charbonnier":
            return "charbonnier", _masked_mean(torch.sqrt(diff * diff + (self.charbonnier_eps**2)), mask, eps=eps)
        if term == "berhu":
            valid_abs = abs_diff * mask
            c = torch.clamp(valid_abs.max(), min=eps) * float(self.berhu_threshold)
            berhu = torch.where(abs_diff <= c, abs_diff, (diff * diff + c * c) / (2.0 * c))
            return "berhu", _masked_mean(berhu, mask, eps=eps)
        if term == "log_l1":
            pred_c = torch.clamp(pred, min=eps)
            tgt_c = torch.clamp(target, min=eps)
            return "log_l1", _masked_mean(torch.abs(torch.log(pred_c) - torch.log(tgt_c)), mask, eps=eps)
        if term == "abs_rel":
            return "abs_rel", _masked_mean(abs_diff / torch.clamp(target, min=eps), mask, eps=eps)

        raise ValueError(f"Unsupported data_term='{term}'.")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        *,
        images: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pred = pred.float()
        target = target.float()
        mask = _ensure_valid_mask(mask.float().detach())
        if images is not None:
            images = images.float()

        losses = {}
        total = pred.new_tensor(0.0)

        if self._cur_l1_weight > 0:
            name, data_loss = self._data_loss(pred, target, mask)
            losses[name] = data_loss
            total = total + self._cur_l1_weight * data_loss
        if self._cur_silog_weight > 0:
            silog_loss = self.silog(pred, target, mask)
            losses["silog"] = silog_loss
            total = total + self._cur_silog_weight * silog_loss
        if self._cur_grad_weight > 0:
            if self.use_edge_aware_smoothness:
                smooth = self.edge_smooth(pred, images, mask)
                losses["edge_smooth"] = smooth
                total = total + self._cur_grad_weight * smooth
            else:
                grad_loss = self.grad(pred, target, mask)
                losses["grad"] = grad_loss
                total = total + self._cur_grad_weight * grad_loss
        if self._cur_ssi_weight > 0:
            ssi_loss = self.ssi(pred, target, mask)
            losses["ssi"] = ssi_loss
            total = total + self._cur_ssi_weight * ssi_loss

        return total, {k: v.detach() for k, v in losses.items()}
