from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from evals.depth_estimation_frozen.losses import DepthCompositeLoss
from evals.depth_estimation_frozen.metrics import compute_depth_metrics, compute_depth_metrics_median_scaled
from evals.image_classification_frozen.models import init_module
from src.models.depth import SurgicalDepthHead
from src.datasets.data_manager import init_data
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import AllReduce, init_distributed
from src.utils.logging import AverageMeter, CSVLogger

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_NORMALIZATION: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = (
    (0.485, 0.456, 0.406),
    (0.229, 0.224, 0.225),
)


def _to_hw(size: int | Sequence[int]) -> Tuple[int, int]:
    if isinstance(size, Sequence) and not isinstance(size, (str, bytes)):
        if len(size) == 1:
            side = int(size[0])
            return side, side
        if len(size) >= 2:
            return int(size[0]), int(size[1])
    side = int(size)
    return side, side


class DepthTensorTransform:
    def __init__(
        self,
        *,
        img_size: int | Sequence[int],
        normalization: Tuple[Sequence[float], Sequence[float]] = DEFAULT_NORMALIZATION,
        random_horizontal_flip: bool = False,
        training: bool = False,
    ):
        self.target_hw = _to_hw(img_size)
        self.mean = torch.tensor(normalization[0], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(normalization[1], dtype=torch.float32).view(3, 1, 1)
        self.random_horizontal_flip = bool(random_horizontal_flip)
        self.training = bool(training)

    def __call__(self, image: torch.Tensor, depth: torch.Tensor, mask: torch.Tensor):
        image = image.float()
        depth = depth.float()
        mask = mask.float()

        if image.shape[-2:] != self.target_hw:
            image = F.interpolate(image.unsqueeze(0), size=self.target_hw, mode="bilinear", align_corners=False).squeeze(0)
        if depth.shape[-2:] != self.target_hw:
            depth = F.interpolate(depth.unsqueeze(0), size=self.target_hw, mode="bilinear", align_corners=False).squeeze(0)
        if mask.shape[-2:] != self.target_hw:
            mask = F.interpolate(mask.unsqueeze(0), size=self.target_hw, mode="nearest").squeeze(0)

        if self.training and self.random_horizontal_flip and torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=(-1,))
            depth = torch.flip(depth, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))

        image = (image - self.mean.to(image.device)) / self.std.to(image.device)
        mask = (mask > 0.5).float()
        depth = depth * mask
        return image, depth, mask


def build_dataloader(
    *,
    annotation_file: str,
    dataset_root: str | None,
    img_size: int | Sequence[int],
    batch_size: int,
    num_workers: int,
    world_size: int,
    rank: int,
    training: bool,
    normalization: Tuple[Sequence[float], Sequence[float]],
    random_horizontal_flip: bool,
    persistent_workers: bool = True,
    pin_memory: bool = True,
):
    transform = DepthTensorTransform(
        img_size=img_size,
        normalization=normalization,
        random_horizontal_flip=random_horizontal_flip,
        training=training,
    )
    loader, sampler = init_data(
        batch_size=batch_size,
        data="c3vd_depth_dataset",
        transform=transform,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        root_path=annotation_file,
        image_folder=dataset_root,
        training=training,
        drop_last=training,
        persistent_workers=persistent_workers,
        pin_mem=pin_memory,
        deterministic=not training,
    )
    return loader, sampler


def main(args_eval, resume_preempt=False):
    val_only = args_eval.get("val_only", False)
    pretrain_folder = args_eval.get("folder", None)
    eval_tag = args_eval.get("tag", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    head_checkpoint = args_eval.get("head_checkpoint", None)
    num_workers = args_eval.get("num_workers", 8)

    args_pretrain = args_eval.get("model_kwargs", {})
    args_model = args_pretrain.get("pretrain_kwargs", {})
    args_wrapper = args_pretrain.get("wrapper_kwargs", {})

    args_exp = args_eval.get("experiment", {})
    args_data = args_exp.get("data", {})
    args_head = args_exp.get("depth_head", {})
    args_opt = args_exp.get("optimization", {})
    args_loss = args_exp.get("loss", {})
    args_logging = args_exp.get("logging", {})
    args_metrics = args_exp.get("metrics", {})

    dataset_root = args_data.get("dataset_root")
    train_annotation = args_data.get("train_annotation")
    val_annotation = args_data.get("val_annotation")
    if val_annotation is None:
        raise ValueError("Validation annotation file must be provided for depth evaluation.")
    img_size = args_data.get("img_size", 256)
    normalization = args_data.get("normalization", DEFAULT_NORMALIZATION)
    random_horizontal_flip = args_data.get("random_horizontal_flip", True)
    min_depth = float(args_data.get("min_depth", 1e-3))
    max_depth = args_data.get("max_depth", None)

    batch_size = args_opt.get("batch_size", 2)
    num_epochs = args_opt.get("num_epochs", 20)
    use_bfloat16 = args_opt.get("use_bfloat16", False)
    base_lr = args_opt.get("lr", 1e-4)
    final_lr = args_opt.get("final_lr", base_lr * 0.1)
    base_wd = args_opt.get("weight_decay", 0.01)
    final_wd = args_opt.get("final_weight_decay", base_wd)
    warmup = args_opt.get("warmup", 0.05)

    decoder_channels = args_head.get("decoder_channels", [512, 256, 128, 64])
    num_feature_levels = args_head.get("num_feature_levels", max(1, len(args_wrapper.get("out_layers", [0]))))
    patch_size = args_head.get("patch_size", args_model.get("encoder", {}).get("patch_size", 16))
    target_size = args_head.get("target_size", img_size)
    activation = args_head.get("activation", "softplus")
    upsample_mode = args_head.get("upsample_mode", "bilinear")

    log_interval = args_logging.get("log_interval", 10)

    metrics_mode = args_metrics.get("mode", "absolute")
    if metrics_mode not in {"absolute", "median_scaled"}:
        raise ValueError(f"Unsupported depth metrics mode '{metrics_mode}'.")
    metric_fn = compute_depth_metrics if metrics_mode == "absolute" else compute_depth_metrics_median_scaled

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()

    folder = Path(pretrain_folder) / "depth_estimation_frozen"
    if eval_tag is not None:
        folder = folder / eval_tag
    folder.mkdir(parents=True, exist_ok=True)

    latest_path = folder / "latest.pt"
    best_path = folder / "best.pt"
    config_path = folder / "config_resolved.json"

    if rank == 0:
        with open(config_path, "w") as f:
            json.dump(args_eval, f, indent=2)
        csv_logger = CSVLogger(
            str(folder / f"log_r{rank}.csv"),
            ("%d", "epoch"),
            ("%.5f", "train_loss"),
            ("%.5f", "val_loss"),
            ("%.5f", "val_abs_rel"),
            ("%.5f", "val_sq_rel"),
            ("%.5f", "val_rmse"),
            ("%.5f", "val_delta1"),
            ("%.5f", "val_delta1_1"),
        )

    encoder = init_module(
        module_name=args_pretrain.get("module_name"),
        resolution=img_size if isinstance(img_size, int) else max(img_size),
        checkpoint=args_pretrain.get("checkpoint"),
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )

    depth_head = SurgicalDepthHead(
        embed_dim=encoder.embed_dim,
        decoder_channels=decoder_channels,
        patch_size=patch_size,
        target_size=target_size,
        num_feature_levels=num_feature_levels,
        activation=activation,
        upsample_mode=upsample_mode,
        min_depth=min_depth,
        max_depth=max_depth,
    ).to(device)

    if world_size > 1:
        depth_head = DistributedDataParallel(depth_head, static_graph=True)

    train_loader, train_sampler = (None, None)
    if not val_only:
        if train_annotation is None:
            raise ValueError("Training annotation file must be provided when not running val_only.")
        train_loader, train_sampler = build_dataloader(
            annotation_file=train_annotation,
            dataset_root=dataset_root,
            img_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            training=True,
            normalization=normalization,
            random_horizontal_flip=random_horizontal_flip,
        )

    val_loader, _ = build_dataloader(
        annotation_file=val_annotation,
        dataset_root=dataset_root,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        training=False,
        normalization=normalization,
        random_horizontal_flip=False,
    )

    iterations_per_epoch = len(train_loader) if train_loader is not None else len(val_loader)
    optimizer = torch.optim.AdamW(depth_head.parameters(), lr=base_lr, weight_decay=base_wd)
    scheduler = WarmupCosineLRSchedule(
        optimizer=optimizer,
        iterations_per_epoch=iterations_per_epoch,
        num_epochs=num_epochs,
        start_lr=base_lr,
        ref_lr=base_lr,
        final_lr=final_lr,
        warmup_ratio=warmup,
    )
    wd_scheduler = CosineWDSchedule(
        optimizer=optimizer,
        iterations_per_epoch=iterations_per_epoch,
        num_epochs=num_epochs,
        ref_wd=base_wd,
        final_wd=final_wd,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    criterion = DepthCompositeLoss(**args_loss)

    start_epoch = 0
    if head_checkpoint:
        depth_head, start_epoch = load_head_checkpoint(device=device, r_path=str(head_checkpoint), head=depth_head)
    elif resume_checkpoint and latest_path.exists():
        depth_head, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=str(latest_path),
            head=depth_head,
            optimizer=optimizer,
            scaler=scaler,
        )

    best_abs_rel = float("inf")

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        torch.save(
            {
                "head": depth_head.state_dict(),
                "opt": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch,
                "batch_size": batch_size,
                "world_size": world_size,
            },
            path,
        )

    for epoch in range(start_epoch, num_epochs):
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch=epoch, num_epochs=num_epochs)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = -1.0
        if not val_only:
            train_loss, _, _, _ = run_one_epoch(
                device=device,
                encoder=encoder,
                head=depth_head,
                criterion=criterion,
                metric_fn=metric_fn,
                data_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                scaler=scaler,
                use_bfloat16=use_bfloat16,
                training=True,
                min_depth=min_depth,
                max_depth=max_depth,
                log_interval=log_interval,
            )

        val_loss, val_metrics, _, _ = run_one_epoch(
            device=device,
            encoder=encoder,
            head=depth_head,
            criterion=criterion,
            metric_fn=metric_fn,
            data_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            scaler=scaler,
            use_bfloat16=use_bfloat16,
            training=False,
            min_depth=min_depth,
            max_depth=max_depth,
            log_interval=log_interval,
        )

        if rank == 0:
            logger.info(
                "epoch=%d train_loss=%.5f val_loss=%.5f abs_rel=%.5f rmse=%.5f",
                epoch + 1,
                train_loss,
                val_loss,
                val_metrics["abs_rel"],
                val_metrics["rmse"],
            )
            csv_logger.log(
                epoch + 1,
                train_loss,
                val_loss,
                val_metrics["abs_rel"],
                val_metrics["sq_rel"],
                val_metrics["rmse"],
                val_metrics["delta1"],
                val_metrics["delta1_1"],
            )

        if val_only:
            return

        save_checkpoint(epoch + 1, latest_path)
        if rank == 0 and val_metrics["abs_rel"] < best_abs_rel:
            best_abs_rel = val_metrics["abs_rel"]
            save_checkpoint(epoch + 1, best_path)


def run_one_epoch(
    *,
    device,
    encoder,
    head,
    criterion,
    metric_fn,
    data_loader,
    optimizer,
    scheduler,
    wd_scheduler,
    scaler,
    use_bfloat16,
    training,
    min_depth,
    max_depth,
    global_step=0,
    log_interval=10,
):
    del scaler
    head.train(mode=training)

    loss_meter = AverageMeter()
    metric_meters = {k: AverageMeter() for k in ("abs_rel", "sq_rel", "rmse", "delta1", "delta1_1")}
    loss_component_meters: Dict[str, AverageMeter] = {}
    step = global_step

    amp_enabled = bool(use_bfloat16)
    amp_dtype = torch.bfloat16

    for itr, (images, depths, masks) in enumerate(data_loader):
        if training:
            scheduler.step()
            wd_scheduler.step()

        images = images.to(device, non_blocking=True)
        depths = depths.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=amp_enabled):
            with torch.no_grad():
                features = encoder(images)
            preds = head(features)
            loss, loss_components = criterion(preds, depths, masks, images=images)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = images.shape[0]
        reduced_loss = float(AllReduce.apply(loss.detach()))
        loss_meter.update(reduced_loss, batch_size)
        step += 1

        for name, value in loss_components.items():
            reduced = float(AllReduce.apply(value.detach()))
            if name not in loss_component_meters:
                loss_component_meters[name] = AverageMeter()
            loss_component_meters[name].update(reduced, batch_size)

        metrics = metric_fn(preds.detach(), depths, masks, min_depth=min_depth, max_depth=max_depth)
        for name, value in metrics.items():
            reduced = float(AllReduce.apply(value.detach()))
            metric_meters[name].update(reduced, batch_size)

        if itr % max(1, log_interval) == 0:
            logger.info(
                "[%5d] %s loss=%.5f abs_rel=%.5f rmse=%.5f",
                itr,
                "train" if training else "val",
                loss_meter.avg,
                metric_meters["abs_rel"].avg,
                metric_meters["rmse"].avg,
            )

    agg_metrics = {k: meter.avg for k, meter in metric_meters.items()}
    agg_losses = {k: meter.avg for k, meter in loss_component_meters.items()}
    return loss_meter.avg, agg_metrics, agg_losses, step


def load_checkpoint(device, r_path, head, optimizer, scaler):
    checkpoint = robust_checkpoint_loader(r_path, map_location=device)
    head.load_state_dict(checkpoint["head"])
    optimizer.load_state_dict(checkpoint["opt"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return head, optimizer, scaler, checkpoint.get("epoch", 0)


def load_head_checkpoint(device, r_path, head):
    checkpoint = robust_checkpoint_loader(r_path, map_location=device)
    head.load_state_dict(checkpoint["head"])
    return head, checkpoint.get("epoch", 0)


class WarmupCosineLRSchedule:
    def __init__(self, optimizer, iterations_per_epoch, num_epochs, start_lr, ref_lr, final_lr, warmup_ratio):
        self.optimizer = optimizer
        self.total_steps = iterations_per_epoch * num_epochs
        self.warmup_steps = int(warmup_ratio * self.total_steps)
        self.ref_lr = ref_lr
        self.start_lr = start_lr
        self.final_lr = final_lr
        self._step = 0

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / max(1, self.warmup_steps)
            lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            progress = float(self._step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = lr


class CosineWDSchedule:
    def __init__(self, optimizer, iterations_per_epoch, num_epochs, ref_wd, final_wd):
        self.optimizer = optimizer
        self.total_steps = iterations_per_epoch * num_epochs
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self._step = 0

    def step(self):
        self._step += 1
        progress = float(self._step) / max(1, self.total_steps)
        wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["weight_decay"] = wd
