import json
import logging
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torchvision.transforms import functional as tvf

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from evals.image_classification_frozen.models import init_module
from evals.segmentation_frozen.losses import BinarySegmentationLoss
from evals.segmentation_frozen.metrics import compute_binary_segmentation_metrics
from src.datasets.data_manager import init_data
from src.models.segmentation import Mask2FormerSegmentationHead, Mask2FormerSegmentationHeadPaper, MaskFormerSegmentationHead
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import AllReduce, init_distributed
from src.utils.logging import AverageMeter, CSVLogger

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

DEFAULT_NORMALIZATION = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class SegmentationPairTransform:
    def __init__(self, img_size=384, normalization=DEFAULT_NORMALIZATION):
        if isinstance(img_size, int):
            self.img_size = (int(img_size), int(img_size))
        else:
            self.img_size = (int(img_size[0]), int(img_size[1]))
        mean, std = normalization
        self.mean = list(mean)
        self.std = list(std)

    def __call__(self, image_pil, mask_tensor):
        image = tvf.to_tensor(image_pil)
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=self.img_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        image = tvf.normalize(image, mean=self.mean, std=self.std)

        mask = mask_tensor.float()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=self.img_size, mode="nearest").squeeze(0)
        mask = (mask > 0.5).float()
        return image, mask


def build_dataloader(
    *,
    annotation_file,
    dataset_root,
    img_size,
    normalization,
    batch_size,
    num_workers,
    world_size,
    rank,
    training,
    drop_last,
    pin_memory,
    persistent_workers,
    dataset_type,
):
    if annotation_file is None:
        return None, None
    transform = SegmentationPairTransform(img_size=img_size, normalization=normalization)
    return init_data(
        batch_size=batch_size,
        transform=transform,
        data=dataset_type,
        pin_mem=pin_memory,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        root_path=dataset_root,
        image_folder=annotation_file,
        training=training,
        drop_last=drop_last,
        deterministic=True,
        persistent_workers=persistent_workers,
    )


def main(args_eval, resume_preempt=False):
    val_only = args_eval.get("val_only", False)
    pretrain_folder = args_eval.get("folder", None)
    eval_tag = args_eval.get("tag", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    head_checkpoint = args_eval.get("head_checkpoint", None)
    num_workers = args_eval.get("num_workers", 8)

    args_pretrain = args_eval.get("model_kwargs")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")

    args_exp = args_eval.get("experiment", {})
    args_data = args_exp.get("data", {})
    args_head = args_exp.get("segmentation_head", {})
    args_opt = args_exp.get("optimization", {})
    args_loss = args_exp.get("loss", {})
    args_logging = args_exp.get("logging", {})
    args_metrics = args_exp.get("metrics", {})

    dataset_root = args_data.get("dataset_root")
    annotation_file = args_data.get("annotation_file", None)
    train_annotation = args_data.get("train_annotation", None)
    val_annotation = args_data.get("val_annotation", None)
    if val_annotation is None:
        val_annotation = annotation_file

    img_size = args_data.get("img_size", 384)
    normalization = args_data.get("normalization", DEFAULT_NORMALIZATION)
    persistent_workers = args_data.get("persistent_workers", True)
    pin_memory = args_data.get("pin_memory", True)
    dataset_type = args_data.get("dataset_type", "cvc12k_segmentation_dataset")

    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs", 50)
    use_bfloat16 = args_opt.get("use_bfloat16", False)
    base_lr = args_opt.get("lr", 5e-4)
    final_lr = args_opt.get("final_lr", base_lr * 0.1)
    base_wd = args_opt.get("weight_decay", 0.01)
    final_wd = args_opt.get("final_weight_decay", base_wd)
    warmup = args_opt.get("warmup", 0.1)

    patch_size = args_head.get("patch_size", args_model["encoder"].get("patch_size", 16))
    target_size = args_head.get("target_size", img_size)
    num_feature_levels = args_head.get("num_feature_levels", max(1, len(args_wrapper.get("out_layers", [0]))))
    hidden_dim = args_head.get("hidden_dim", 256)
    mask_dim = args_head.get("mask_dim", hidden_dim)
    num_queries = args_head.get("num_queries", 1)
    num_decoder_layers = args_head.get("num_decoder_layers", 4)
    nheads = args_head.get("nheads", 8)
    dim_feedforward = args_head.get("dim_feedforward", 1024)
    dropout = args_head.get("dropout", 0.0)
    decoder_channels = args_head.get("decoder_channels", [hidden_dim, 128, 64])
    upsample_mode = args_head.get("upsample_mode", "bilinear")
    return_aux = args_head.get("return_aux", True)

    metrics_threshold = float(args_metrics.get("threshold", 0.5))
    log_interval = int(args_logging.get("log_interval", 10))
    enable_tensorboard = bool(args_logging.get("enable_tensorboard", True))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    folder = Path(pretrain_folder) / "segmentation_frozen"
    if eval_tag is not None:
        folder = folder / eval_tag
    folder.mkdir(parents=True, exist_ok=True)
    log_file = folder / f"log_r{rank}.csv"
    latest_path = folder / "latest.pt"
    best_path = folder / "best.pt"
    config_path = folder / "config_resolved.json"

    if rank == 0:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(args_eval, f, indent=2)
        csv_logger = CSVLogger(
            str(log_file),
            ("%d", "epoch"),
            ("%.6f", "train_loss"),
            ("%.6f", "train_bce"),
            ("%.6f", "train_dice_loss"),
            ("%.6f", "train_structure_loss"),
            ("%.6f", "train_aux_loss"),
            ("%.6f", "val_loss"),
            ("%.6f", "val_bce"),
            ("%.6f", "val_dice_loss"),
            ("%.6f", "val_structure_loss"),
            ("%.6f", "val_aux_loss"),
            ("%.6f", "val_iou"),
            ("%.6f", "val_dice"),
            ("%.6f", "val_precision"),
            ("%.6f", "val_recall"),
            ("%.6f", "val_specificity"),
            ("%.6f", "val_pixel_acc"),
            ("%.6f", "val_mae"),
        )
        writer = SummaryWriter(log_dir=str(folder / "tensorboard")) if (enable_tensorboard and SummaryWriter is not None) else None
    else:
        csv_logger = None
        writer = None

    encoder = init_module(
        module_name=args_pretrain.get("module_name"),
        resolution=img_size if isinstance(img_size, int) else max(img_size),
        checkpoint=args_pretrain.get("checkpoint"),
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )

    head_type = str(args_head.get("type", "maskformer")).lower()
    head_map = {
        "maskformer": MaskFormerSegmentationHead,
        "mask2former": Mask2FormerSegmentationHead,
        "mask2former_paper": Mask2FormerSegmentationHeadPaper,
    }
    if head_type not in head_map:
        raise ValueError(f"Unsupported segmentation_head.type='{head_type}'. Use one of: {sorted(list(head_map.keys()))}")

    head_cls = head_map[head_type]
    head_kwargs = dict(
        embed_dim=encoder.embed_dim,
        patch_size=patch_size,
        target_size=target_size,
        num_feature_levels=num_feature_levels,
        hidden_dim=hidden_dim,
        mask_dim=mask_dim,
        num_queries=num_queries,
        num_decoder_layers=num_decoder_layers,
        nheads=nheads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        decoder_channels=decoder_channels,
        upsample_mode=upsample_mode,
        return_aux=return_aux,
    )
    if head_type in {"mask2former", "mask2former_paper"}:
        head_kwargs["attn_mask_threshold"] = args_head.get("attn_mask_threshold", 0.5)
    seg_head = head_cls(**head_kwargs).to(device)
    if world_size > 1:
        seg_head = DistributedDataParallel(seg_head, static_graph=True)

    if val_annotation is None:
        raise ValueError("Validation annotation file must be provided (val_annotation or annotation_file).")

    train_loader, train_sampler = (None, None)
    if not val_only:
        if train_annotation is None:
            raise ValueError("Training requested but train_annotation is None.")
        train_loader, train_sampler = build_dataloader(
            annotation_file=train_annotation,
            dataset_root=dataset_root,
            img_size=img_size,
            normalization=normalization,
            batch_size=batch_size,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            training=True,
            drop_last=True,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            dataset_type=dataset_type,
        )

    val_loader, _ = build_dataloader(
        annotation_file=val_annotation,
        dataset_root=dataset_root,
        img_size=img_size,
        normalization=normalization,
        batch_size=batch_size,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        training=False,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        dataset_type=dataset_type,
    )

    iterations_per_epoch = len(train_loader) if train_loader is not None else len(val_loader)
    optimizer = torch.optim.AdamW(seg_head.parameters(), lr=base_lr, weight_decay=base_wd)
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
    criterion = BinarySegmentationLoss(**args_loss)

    start_epoch = 0
    if head_checkpoint:
        ckpt_path = head_checkpoint
        if isinstance(ckpt_path, str) and ckpt_path.lower() in {"best", "best.pt"}:
            ckpt_path = str(best_path)
        elif isinstance(ckpt_path, str) and ckpt_path.lower() in {"latest", "latest.pt"}:
            ckpt_path = str(latest_path)
        seg_head, start_epoch = load_head_checkpoint(device=device, r_path=str(ckpt_path), head=seg_head)
    elif resume_checkpoint and latest_path.exists():
        seg_head, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=str(latest_path),
            head=seg_head,
            optimizer=optimizer,
            scaler=scaler,
        )
        for _ in range(start_epoch * iterations_per_epoch):
            scheduler.step()
            wd_scheduler.step()

    best_iou = float("-inf")

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        torch.save(
            {
                "head": seg_head.state_dict(),
                "opt": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch,
                "batch_size": batch_size,
                "world_size": world_size,
            },
            path,
        )

    train_step = 0
    val_step = 0
    for epoch in range(start_epoch, num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if val_only:
            train_loss, train_metrics, train_comps = -1.0, {}, {}
        else:
            train_loss, train_metrics, train_comps, train_step = run_one_epoch(
                device=device,
                encoder=encoder,
                head=seg_head,
                criterion=criterion,
                data_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                scaler=scaler,
                use_bfloat16=use_bfloat16,
                training=True,
                global_step=train_step,
                writer=writer if rank == 0 else None,
                phase="train",
                log_interval=log_interval,
                metrics_threshold=metrics_threshold,
            )

        val_loss, val_metrics, val_comps, val_step = run_one_epoch(
            device=device,
            encoder=encoder,
            head=seg_head,
            criterion=criterion,
            data_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            scaler=scaler,
            use_bfloat16=use_bfloat16,
            training=False,
            global_step=val_step,
            writer=writer if rank == 0 else None,
            phase="val",
            log_interval=log_interval,
            metrics_threshold=metrics_threshold,
        )

        if rank == 0:
            logger.info(
                "[%5d] train_loss: %.5f | val_loss: %.5f | iou: %.5f | dice: %.5f | prec: %.5f | rec: %.5f | spe: %.5f | acc: %.5f | mae: %.5f",
                epoch + 1,
                train_loss,
                val_loss,
                val_metrics["iou"],
                val_metrics["dice"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["specificity"],
                val_metrics["pixel_acc"],
                val_metrics["mae"],
            )
            csv_logger.log(
                epoch + 1,
                train_loss,
                float(train_comps.get("bce", 0.0)),
                float(train_comps.get("dice", 0.0)),
                float(train_comps.get("structure", 0.0)),
                float(train_comps.get("aux", 0.0)),
                val_loss,
                float(val_comps.get("bce", 0.0)),
                float(val_comps.get("dice", 0.0)),
                float(val_comps.get("structure", 0.0)),
                float(val_comps.get("aux", 0.0)),
                val_metrics["iou"],
                val_metrics["dice"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["specificity"],
                val_metrics["pixel_acc"],
                val_metrics["mae"],
            )
            if writer is not None:
                if not val_only:
                    writer.add_scalar("epoch/train_loss", train_loss, epoch + 1)
                    for k, v in train_comps.items():
                        writer.add_scalar(f"epoch/train_{k}_loss", float(v), epoch + 1)
                    for k, v in train_metrics.items():
                        writer.add_scalar(f"epoch/train_{k}", float(v), epoch + 1)
                writer.add_scalar("epoch/val_loss", val_loss, epoch + 1)
                for k, v in val_comps.items():
                    writer.add_scalar(f"epoch/val_{k}_loss", float(v), epoch + 1)
                for k, v in val_metrics.items():
                    writer.add_scalar(f"epoch/val_{k}", float(v), epoch + 1)

        if val_only:
            break

        save_checkpoint(epoch + 1, latest_path)
        if rank == 0 and not math.isnan(val_metrics["iou"]) and val_metrics["iou"] > best_iou:
            best_iou = float(val_metrics["iou"])
            logger.info("New best IoU %.5f. Saving best checkpoint.", best_iou)
            save_checkpoint(epoch + 1, best_path)

    if writer is not None:
        writer.close()


def run_one_epoch(
    *,
    device,
    encoder,
    head,
    criterion,
    data_loader,
    optimizer,
    scheduler,
    wd_scheduler,
    scaler,
    use_bfloat16,
    training,
    global_step=0,
    writer=None,
    phase="train",
    log_interval=10,
    metrics_threshold: float = 0.5,
):
    del scaler
    head.train(mode=training)
    loss_meter = AverageMeter()
    loss_component_meters = {}
    metric_meters = {
        k: AverageMeter()
        for k in ("iou", "dice", "precision", "recall", "sensitivity", "specificity", "pixel_acc", "mae")
    }
    step = global_step

    amp_enabled = bool(use_bfloat16) and device.type == "cuda"

    for itr, batch in enumerate(data_loader):
        if training:
            scheduler.step()
            wd_scheduler.step()

        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        amp_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=amp_enabled) if amp_enabled else nullcontext()
        with amp_ctx:
            with torch.no_grad():
                features = encoder(images)
            preds = head(features)
            loss, loss_components = criterion(preds, masks)

        if not torch.isfinite(loss).all():
            if itr % max(1, log_interval) == 0:
                logger.error(f"[{itr:5d}] {phase} non-finite loss detected; skipping step.")
            if training:
                optimizer.zero_grad(set_to_none=True)
            continue

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        reduced_loss = float(AllReduce.apply(loss.detach()))
        loss_meter.update(reduced_loss, batch_size)
        step += 1

        for name, value in loss_components.items():
            reduced_comp = float(AllReduce.apply(value.detach()))
            if name not in loss_component_meters:
                loss_component_meters[name] = AverageMeter()
            loss_component_meters[name].update(reduced_comp, batch_size)
            if writer is not None:
                writer.add_scalar(f"{phase}/{name}_iter", reduced_comp, step)

        if writer is not None:
            writer.add_scalar(f"{phase}/loss_iter", reduced_loss, step)

        metrics = compute_binary_segmentation_metrics(preds, masks, threshold=metrics_threshold)
        for name, value in metrics.items():
            metric_value = float(AllReduce.apply(value.detach()))
            metric_meters[name].update(metric_value, batch_size)
            if writer is not None:
                writer.add_scalar(f"{phase}/{name}_iter", metric_value, step)

        if itr % max(1, log_interval) == 0:
            logger.info(
                "[%5d] %s loss: %.5f | iou: %.5f | dice: %.5f | prec: %.5f | rec: %.5f | spe: %.5f | acc: %.5f | mae: %.5f",
                itr,
                "train" if training else "val",
                loss_meter.avg,
                metric_meters["iou"].avg,
                metric_meters["dice"].avg,
                metric_meters["precision"].avg,
                metric_meters["recall"].avg,
                metric_meters["specificity"].avg,
                metric_meters["pixel_acc"].avg,
                metric_meters["mae"].avg,
            )

    agg_metrics = {k: meter.avg for k, meter in metric_meters.items()}
    agg_loss_components = {k: meter.avg for k, meter in loss_component_meters.items()}
    return loss_meter.avg, agg_metrics, agg_loss_components, step


def load_checkpoint(device, r_path, head, optimizer, scaler):
    checkpoint = robust_checkpoint_loader(r_path, map_location=device)
    logger.info(f"read-path: {r_path}")
    msg = head.load_state_dict(checkpoint["head"])
    logger.info(f"Loaded head with msg: {msg}")
    optimizer.load_state_dict(checkpoint["opt"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    epoch = checkpoint["epoch"]
    return head, optimizer, scaler, epoch


def load_head_checkpoint(device, r_path, head):
    checkpoint = robust_checkpoint_loader(r_path, map_location=device)
    logger.info(f"read-path: {r_path}")
    msg = head.load_state_dict(checkpoint["head"])
    logger.info(f"Loaded head with msg: {msg}")
    epoch = checkpoint.get("epoch", 0)
    return head, epoch


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
