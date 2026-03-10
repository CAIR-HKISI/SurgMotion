import os
import logging
import math
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import wandb
from torch.nn.parallel import DistributedDataParallel

from evals.surgical_video_classification_frozen.models import init_module
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.data_manager import init_data
from src.models.attentive_pooler import AttentiveRegressor
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import AllReduce, init_distributed
from src.utils.logging import AverageMeter

import torch.distributed as dist


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


# ----------------------
# Anticipation Metrics
# ----------------------
ANTICIPATION_BACKGROUND_VALUE = 5.0
DEFAULT_METRIC_HORIZONS = [2.0, 3.0, 5.0]


def _nanmean(values):
    valid_values = [v for v in values if not np.isnan(v)]
    return float(np.mean(valid_values)) if valid_values else float("nan")


def _masked_mae(errors, mask):
    if np.any(mask):
        return float(errors[mask].mean())
    return float("nan")


def _pair_mean(v1, v2):
    if np.isnan(v1) or np.isnan(v2):
        return float("nan")
    return float((v1 + v2) / 2.0)


def _format_horizon_tag(horizon):
    if float(horizon).is_integer():
        return f"h{int(horizon)}"
    return f"h{str(horizon).replace('.', 'p')}"


def compute_horizon_metrics(predictions, targets, horizon):
    """
    Compute B-style anticipation metrics under horizon h.

    out_MAE: t == h
    in_MAE: 0 < t < h
    wMAE: (out_MAE + in_MAE) / 2
    eMAE: 0 < t < 0.1h
    mMAE: 0.1h < t < 0.9h
    dMAE: 0.9h < t < h
    """
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    clipped_predictions = np.clip(predictions, 0.0, horizon)
    clipped_targets = np.clip(targets, 0.0, horizon)
    abs_error = np.abs(clipped_predictions - clipped_targets)

    out_mask = np.isclose(clipped_targets, horizon)
    in_mask = (clipped_targets > 0.0) & (clipped_targets < horizon)
    e_mask = (clipped_targets > 0.0) & (clipped_targets < 0.1 * horizon)
    m_mask = (clipped_targets > 0.1 * horizon) & (clipped_targets < 0.9 * horizon)
    d_mask = (clipped_targets > 0.9 * horizon) & (clipped_targets < horizon)

    out_mae = _masked_mae(abs_error, out_mask)
    in_mae = _masked_mae(abs_error, in_mask)

    return {
        "out_MAE": out_mae,
        "in_MAE": in_mae,
        "wMAE": _pair_mean(out_mae, in_mae),
        "eMAE": _masked_mae(abs_error, e_mask),
        "mMAE": _masked_mae(abs_error, m_mask),
        "dMAE": _masked_mae(abs_error, d_mask),
        "Clipped_MAE": float(abs_error.mean()),
    }


def compute_anticipation_metrics(predictions, targets, metric_horizons):
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    metrics = {
        "Raw_MAE": float(np.mean(np.abs(predictions - targets))),
    }
    for horizon in metric_horizons:
        prefix = _format_horizon_tag(horizon)
        for metric_name, metric_value in compute_horizon_metrics(predictions, targets, horizon).items():
            metrics[f"{prefix}_{metric_name}"] = metric_value
    return metrics


def evaluate_per_video_anticipation(predictions_df, target_names=None, metric_horizons=None):
    """
    Evaluate per-video anticipation metrics.

    Args:
        predictions_df: DataFrame with columns
            [data_idx, vid, target_idx, target_name, prediction, label]
        Returns:
            per_video: List of per-video metrics
            stats: Aggregated statistics
    """
    metric_horizons = metric_horizons or DEFAULT_METRIC_HORIZONS
    primary_horizon = max(metric_horizons)
    primary_tag = _format_horizon_tag(primary_horizon)

    predictions_df = predictions_df.sort_values(["vid", "data_idx", "target_idx"])
    if target_names is None:
        if "target_name" in predictions_df.columns:
            target_names = list(
                predictions_df.sort_values("target_idx")["target_name"].drop_duplicates().tolist()
            )
        else:
            target_names = [f"target_{idx}" for idx in sorted(predictions_df["target_idx"].unique())]

    per_video = []
    for vid_name, vid_data in predictions_df.groupby("vid"):
        gt = vid_data["label"].values
        pred = vid_data["prediction"].values

        metrics = compute_anticipation_metrics(pred, gt, metric_horizons=metric_horizons)
        per_target_metrics = {}
        for target_name, target_df in vid_data.groupby("target_name"):
            target_metrics = compute_horizon_metrics(
                target_df["prediction"].values,
                target_df["label"].values,
                primary_horizon,
            )
            per_target_metrics[target_name] = target_metrics
            metrics[f"{target_name}_{primary_tag}_wMAE"] = target_metrics["wMAE"]

        metrics[f"{primary_tag}_TargetAvg_wMAE"] = _nanmean([v["wMAE"] for v in per_target_metrics.values()])
        metrics["vid"] = vid_name
        per_video.append(metrics)

    all_gt = predictions_df["label"].values
    all_pred = predictions_df["prediction"].values
    overall_metrics = compute_anticipation_metrics(all_pred, all_gt, metric_horizons=metric_horizons)
    overall_per_target = {}
    for target_name, target_df in predictions_df.groupby("target_name"):
        overall_per_target[target_name] = compute_horizon_metrics(
            target_df["prediction"].values,
            target_df["label"].values,
            primary_horizon,
        )
    overall_metrics[f"{primary_tag}_TargetAvg_wMAE"] = _nanmean([v["wMAE"] for v in overall_per_target.values()])

    summary_metric_keys = ["Raw_MAE"]
    for horizon in metric_horizons:
        prefix = _format_horizon_tag(horizon)
        summary_metric_keys.extend(
            [f"{prefix}_{metric_name}" for metric_name in ["out_MAE", "in_MAE", "wMAE", "eMAE", "mMAE", "dMAE", "Clipped_MAE"]]
        )
    summary_metric_keys.append(f"{primary_tag}_TargetAvg_wMAE")

    stats = {}
    for metric in summary_metric_keys:
        stats[f"{metric}_Mean"] = _nanmean([v[metric] for v in per_video])

    for metric in summary_metric_keys:
        stats[f"Overall_{metric}"] = overall_metrics[metric]
    stats["per_target"] = overall_per_target
    stats["primary_horizon"] = primary_horizon
    stats["primary_tag"] = primary_tag
    return per_video, stats


# ----------------------
# 主入口
# ----------------------
def normalize_task_type(task_type):
    task_type = (task_type or "anticipation").lower()
    if task_type in {"anticipation", "regression", "prediction"}:
        return "anticipation"
    raise ValueError(
        f"`evals/surgical_phase_anticipation` 仅支持 anticipation/prediction，收到: '{task_type}'"
    )


def validate_anticipation_data_paths(paths, split_name):
    if isinstance(paths, str):
        paths = [paths]

    validated_paths = []
    for path in paths:
        if not isinstance(path, str):
            raise TypeError(
                f"`{split_name}` 中的每个数据路径都必须是字符串，收到: {type(path).__name__}"
            )
        if not path.endswith(".csv"):
            raise ValueError(
                f"`{split_name}` 仅接受明确的 CSV 文件路径，不再支持目录自动解析: {path}"
            )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"`{split_name}` 指定的 CSV 不存在: {path}")
        validated_paths.append(path)
    return validated_paths


def infer_anticipation_output_dims(data_paths, target_names=None):
    if target_names is not None:
        return [len(target_names)] * len(data_paths)

    output_dims = []
    for data_path in data_paths:
        if not data_path.endswith(".csv"):
            raise ValueError(
                "anticipation 任务需要 CSV 数据源，或显式通过 target_names/num_classes 指定输出维度"
            )
        df = pd.read_csv(data_path, nrows=1)
        ant_cols = [c for c in df.columns if c.startswith("ant_reg_")]
        if not ant_cols:
            raise ValueError(f"{data_path} 中未找到 ant_reg_* 列，无法推断回归输出维度")
        output_dims.append(len(ant_cols))
    return output_dims


def main(args_eval, resume_preempt=False):

    val_only = args_eval.get("val_only", False)
    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)
    num_workers = args_eval.get("num_workers", 12)

    # Quick run / debug mode configuration
    quick_run = args_eval.get("quick_run", False)
    quick_run_num_videos = args_eval.get("quick_run_num_videos", 2)

    task_type = normalize_task_type(args_eval.get("task_type", "anticipation"))

    metric_horizons = [float(h) for h in args_eval.get("metric_horizons", DEFAULT_METRIC_HORIZONS)]

    # wandb configuration
    wandb_config = args_eval.get("wandb", {})
    wandb_project = wandb_config.get("project", "nsjepa-surgical-probing")
    wandb_entity = wandb_config.get("entity", None)
    wandb_name = wandb_config.get("name", None)
    wandb_tags = wandb_config.get("tags", [])
    wandb_group = wandb_config.get("group", None)
    wandb_notes = wandb_config.get("notes", None)

    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")

    args_exp = args_eval.get("experiment")
    args_classifier = args_exp.get("classifier")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 16)

    args_data = args_exp.get("data")
    dataset_type = args_data.get("dataset_type", "VideoDataset")
    target_names = args_data.get("target_names")

    # Support both single and multiple datasets
    train_data_path = args_data.get("dataset_train")
    if isinstance(train_data_path, str):
        train_data_path = [train_data_path]

    val_data_path = args_data.get("dataset_val")
    if isinstance(val_data_path, str):
        val_data_path = [val_data_path]

    train_data_path = validate_anticipation_data_paths(train_data_path, "dataset_train")
    val_data_path = validate_anticipation_data_paths(val_data_path, "dataset_val")

    # Support datasets_weights for sampling from multiple datasets
    datasets_weights = args_data.get("datasets_weights", None)

    num_classes = args_data.get("num_classes")
    if isinstance(num_classes, int):
        num_classes_list = [num_classes] * len(train_data_path)
    else:
        num_classes_list = num_classes
    if num_classes_list is None:
        num_classes_list = infer_anticipation_output_dims(train_data_path, target_names=target_names)

    # Support head-to-dataset mapping (which dataset each head trains on)
    head_to_dataset_map = args_data.get("head_to_dataset_map", None)

    resolution = args_data.get("resolution", 224)
    num_segments = args_data.get("num_segments", 1)
    frames_per_clip = args_data.get("frames_per_clip", 16)
    frame_step = args_data.get("frame_step", 1)
    duration = args_data.get("clip_duration", None)
    num_views_per_segment = args_data.get("num_views_per_segment", 1)
    normalization = args_data.get("normalization", None)

    args_opt = args_exp.get("optimization")
    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    use_bfloat16 = args_opt.get("use_bfloat16")
    
    opt_kwargs = args_opt.get("multihead_kwargs")

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    world_size, rank = init_distributed()

    # Quick run mode: create subset CSV files (only on rank 0)
    if quick_run and rank == 0:
        logger.info(f"Quick run mode enabled: using {quick_run_num_videos} video(s)")

        # Create subset for training data
        train_data_path_subset = []
        for path in train_data_path:
            if path.endswith('.csv'):
                subset_path = create_quick_run_subset(path, num_videos=quick_run_num_videos)
                train_data_path_subset.append(subset_path)
            else:
                logger.warning(f"Quick run mode only supports CSV files, skipping: {path}")
                train_data_path_subset.append(path)

        # Create subset for validation data
        val_data_path_subset = []
        for path in val_data_path:
            if path.endswith('.csv'):
                subset_path = create_quick_run_subset(path, num_videos=quick_run_num_videos)
                val_data_path_subset.append(subset_path)
            else:
                logger.warning(f"Quick run mode only supports CSV files, skipping: {path}")
                val_data_path_subset.append(path)

        # Use subset paths
        train_data_path = train_data_path_subset
        val_data_path = val_data_path_subset

    # Sync across ranks if using distributed training
    if dist.is_initialized():
        dist.barrier()

    # Initialize wandb (only on rank 0)
    if rank == 0:
        # Extract model name (parent directory of checkpoint) for logging
        if checkpoint:
            checkpoint_dir = os.path.dirname(checkpoint)
            model_name = os.path.basename(checkpoint_dir)
        else:
            model_name = "unknown"

        # Prepare config dict for wandb
        wandb_run_config = {
            "eval_name": args_eval.get("eval_name"),
            "tag": eval_tag,
            "dataset": args_eval.get("dataset", "unknown"),  # Dataset name for easy identification
            "task_type": task_type,
            "model": model_name,  # Model/checkpoint directory name
            "checkpoint_path": checkpoint,  # Full checkpoint path
            "num_workers": num_workers,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "use_bfloat16": use_bfloat16,
            "num_probe_blocks": num_probe_blocks,
            "num_heads": num_heads,
            "frames_per_clip": frames_per_clip,
            "resolution": resolution,
            "num_classes": num_classes_list,
            "target_names": target_names,
            "metric_horizons": metric_horizons,
            "quick_run": quick_run,
            "quick_run_num_videos": quick_run_num_videos if quick_run else None,
        }

        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            config=wandb_run_config,
            tags=wandb_tags,
            group=wandb_group,
            notes=wandb_notes,
            resume="allow"
        )
        logger.info(f"wandb initialized: {wandb.run.url}")

    # checkpoint 路径
    folder = pretrain_folder
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    os.makedirs(folder, exist_ok=True)
    latest_path = os.path.join(folder, "latest.pt")

    # 构建 encoder
    encoder = init_module(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )

    classifiers = [
        AttentiveRegressor(
            embed_dim=encoder.embed_dim,
            num_heads=num_heads,
            depth=num_probe_blocks,
            num_outputs=(
                num_classes_list[head_to_dataset_map[idx]]
                if head_to_dataset_map is not None
                else num_classes_list[0]
            ),
            use_activation_checkpointing=True,
        ).to(device)
        for idx, _ in enumerate(opt_kwargs)
    ]

    # Only use DistributedDataParallel if distributed is initialized
    if dist.is_initialized():
        classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers]
        logger.info(f"Wrapped {task_type} heads with DistributedDataParallel")
    else:
        logger.info(f"Running {task_type} in single-process mode (no DDP wrapping)")

    train_loader, train_sampler = make_dataloader(
        dataset_type=dataset_type,
        root_path=train_data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        eval_duration=duration,
        num_segments=num_segments,
        num_views_per_segment=1,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True,
        num_workers=num_workers,
        normalization=normalization,
        datasets_weights=datasets_weights,
        target_names=target_names,
    )
    val_loader, _ = make_dataloader(
        dataset_type=dataset_type,
        root_path=val_data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_segments=num_segments,
        eval_duration=duration,
        num_views_per_segment=num_views_per_segment,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=False,
        num_workers=num_workers,
        normalization=normalization,
        target_names=target_names,
    )

    ipe = len(train_loader)

    # 多头优化器
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifiers=classifiers,
        opt_kwargs=opt_kwargs,
        iterations_per_epoch=ipe,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )

    # 断点恢复
    start_epoch = 0
    if resume_checkpoint and os.path.exists(latest_path):
        encoder, classifiers, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=latest_path,
            encoder=encoder,
            classifiers=classifiers,
            opt=optimizer,
            scaler=scaler,
            val_only=val_only,
        )
        for _ in range(start_epoch * ipe):
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

    def save_checkpoint(epoch):
        save_dict = {
            "encoder": encoder.state_dict(),
            "classifiers": [c.module.state_dict() if hasattr(c, 'module') else c.state_dict() for c in classifiers],
            "opt": [o.state_dict() for o in optimizer],
            "scaler": [None if s is None else s.state_dict() for s in scaler],
            "epoch": epoch,
        }
        if rank == 0:
            torch.save(save_dict, latest_path)

    # ----------------
    # 训练循环
    # ----------------
    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)

        if not val_only:
            train_metrics = run_one_epoch(
                device=device,
                training=True,
                encoder=encoder,
                classifiers=classifiers,
                scaler=scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                data_loader=train_loader,
                use_bfloat16=use_bfloat16,
                epoch=epoch,
                head_to_dataset_map=head_to_dataset_map,
                metric_horizons=metric_horizons,
                rank=rank,
            )
        else:
            train_metrics = None

        val_metrics = run_one_epoch(
            device=device,
            training=False,
            encoder=encoder,
            classifiers=classifiers,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
            epoch=epoch,
            save_predictions=True,
            folder=folder,
            head_to_dataset_map=head_to_dataset_map,
            metric_horizons=metric_horizons,
            rank=rank,
            train_loader_len=len(train_loader),
        )

        logger.info(f"Epoch {epoch+1}: train={train_metrics} val={val_metrics}")

        if val_only:
            if dist.is_initialized():
                dist.destroy_process_group()
            return

        save_checkpoint(epoch + 1)

    # Finish wandb run (only on rank 0)
    if rank == 0:
        wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


# ----------------------
# 单个 epoch 训练/验证
# ----------------------
def run_one_epoch(
    device,
    training,
    encoder,
    classifiers,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    epoch=0,
    folder=None,
    save_predictions=False,
    log_interval=20,
    head_to_dataset_map=None,
    metric_horizons=None,
    rank=0,
    train_loader_len=None,
):
    metric_horizons = metric_horizons or DEFAULT_METRIC_HORIZONS
    primary_horizon = max(metric_horizons)
    primary_tag = _format_horizon_tag(primary_horizon)

    for c in classifiers:
        c.train(mode=training)

    base_dataset = data_loader.dataset.dataset if hasattr(data_loader.dataset, "dataset") else data_loader.dataset
    loader_target_names = getattr(base_dataset, "target_names", None)
    criterion = torch.nn.MSELoss()

    acc_meters = [AverageMeter() for _ in classifiers]
    loss_meters = [AverageMeter() for _ in classifiers]

    if not training:
        all_predictions = []

    for itr, data in enumerate(data_loader):
        if training:
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_bfloat16):
            clips = [[dij.to(device) for dij in di] for di in data[0]]
            clip_indices = [d.to(device) for d in data[2]]
            labels = data[1][0].to(device)
            batch_size = len(labels)

            # Extract dataset_idx if available (for multi-dataset training)
            if len(data[1]) > 3:
                dataset_indices = data[1][3].to(device)
            else:
                dataset_indices = None

            with torch.no_grad():
                features = encoder(clips, clip_indices)

            outputs = [[c(f) for f in features] for c in classifiers]
            losses = []
            has_samples = []
            labels_float = labels.float()

            if head_to_dataset_map is not None and dataset_indices is not None:
                for head_idx, coutputs in enumerate(outputs):
                    assigned_dataset = head_to_dataset_map[head_idx]
                    head_losses = []
                    head_has_samples = False
                    for o in coutputs:
                        mask = (dataset_indices == assigned_dataset)
                        if mask.sum() > 0:
                            loss = criterion(o[mask], labels_float[mask])
                            head_has_samples = True
                        else:
                            loss = torch.tensor(0.0, device=device, requires_grad=True)
                        head_losses.append(loss)
                    losses.append(head_losses)
                    has_samples.append(head_has_samples)
            else:
                losses = [[criterion(o, labels_float) for o in coutputs] for coutputs in outputs]
                has_samples = [True] * len(losses)

        if training:
            if use_bfloat16:
                for s, li, o, has_sample in zip(scaler, losses, optimizer, has_samples):
                    if has_sample:
                        for lij in li:
                            s.scale(lij).backward(retain_graph=True)
                        s.step(o)
                        s.update()
                        o.zero_grad()
                    else:
                        # Skip optimizer step if no samples for this head in batch
                        o.zero_grad()
            else:
                for li, o, has_sample in zip(losses, optimizer, has_samples):
                    if has_sample:
                        for lij in li:
                            lij.backward(retain_graph=True)
                        o.step()
                        o.zero_grad()
                    else:
                        # Skip optimizer step if no samples for this head in batch
                        o.zero_grad()

        with torch.no_grad():
            for idx, coutputs in enumerate(outputs):
                preds = torch.stack(coutputs).mean(0)

                if head_to_dataset_map is not None and dataset_indices is not None:
                    assigned_dataset = head_to_dataset_map[idx]
                    mask = (dataset_indices == assigned_dataset)
                    if mask.sum() > 0:
                        masked_preds = preds[mask]
                        masked_labels = labels[mask]
                        acc = torch.abs(masked_preds - masked_labels.float()).mean()
                        acc = float(AllReduce.apply(acc))
                        acc_meters[idx].update(acc, mask.sum().item())
                else:
                    acc = torch.abs(preds - labels.float()).mean()
                    acc = float(AllReduce.apply(acc))
                    acc_meters[idx].update(acc)

                loss_val = torch.stack([lij.detach() for lij in losses[idx]]).mean().item()
                loss_meters[idx].update(loss_val, batch_size)

                if not training:
                    vid_ids, data_idxs = data[1][1], data[1][2]
                    cur_target_names = loader_target_names
                    if cur_target_names is None:
                        cur_target_names = [f"target_{i}" for i in range(preds.shape[-1])]
                    if dataset_indices is not None:
                        batch_iter = zip(
                            preds.cpu(),
                            labels.cpu(),
                            vid_ids.cpu(),
                            data_idxs.cpu(),
                            dataset_indices.cpu(),
                        )
                    else:
                        batch_iter = (
                            (pred, label, vid, did, torch.tensor(0))
                            for pred, label, vid, did in zip(
                                preds.cpu(), labels.cpu(), vid_ids.cpu(), data_idxs.cpu()
                            )
                        )
                    for pred, label, vid, did, ds_idx in batch_iter:
                        for target_idx, target_name in enumerate(cur_target_names):
                            all_predictions.append([
                                idx,
                                did.item(),
                                vid.item(),
                                target_idx,
                                target_name,
                                float(pred[target_idx].item()),
                                float(label[target_idx].item()),
                                ds_idx.item(),
                            ])

        if itr % log_interval == 0:
            if training:
                logger.info(
                    f"[Train][Epoch {epoch}][Iter {itr}/{len(data_loader)}] "
                    + " ".join([f"Head{h}: MAE={am.avg:.4f}, Loss={lm.avg:.4f}"
                                for h, (am, lm) in enumerate(zip(acc_meters, loss_meters))])
                )

                if rank == 0:
                    global_step = epoch * len(data_loader) + itr
                    wandb_metrics = {}
                    for h, (am, lm) in enumerate(zip(acc_meters, loss_meters)):
                        wandb_metrics[f"train/head_{h}/MAE"] = am.avg
                        wandb_metrics[f"train/head_{h}/Loss"] = lm.avg
                    wandb.log(wandb_metrics, step=global_step)
            else:
                logger.info(
                    f"[Val][Epoch {epoch}][Iter {itr}/{len(data_loader)}] "
                    + " ".join([f"Head{h}: MAE={am.avg:.4f}" for h, am in enumerate(acc_meters)])
                )

    metrics = {f"head_{i}": {"MAE": acc_meters[i].avg, "Loss": loss_meters[i].avg} for i in range(len(classifiers))}

    if not training and len(all_predictions) > 0:
        df = pd.DataFrame(
            all_predictions,
            columns=["head", "data_idx", "vid", "target_idx", "target_name", "prediction", "label", "dataset_idx"],
        )
        results = {}

        global_step = (epoch + 1) * train_loader_len

        for head_id, g in df.groupby("head"):
            _, stats = evaluate_per_video_anticipation(
                g,
                target_names=loader_target_names,
                metric_horizons=metric_horizons,
            )
            results[f"head_{head_id}"] = stats

            if rank == 0:
                wandb_metrics = {}
                for horizon in metric_horizons:
                    tag = _format_horizon_tag(horizon)
                    for metric_name in ["out_MAE", "in_MAE", "wMAE", "eMAE", "mMAE", "dMAE"]:
                        wandb_metrics[f"val/head_{head_id}/{tag}/{metric_name}"] = stats[f"{tag}_{metric_name}_Mean"]
                    wandb_metrics[f"val/head_{head_id}/{tag}/Clipped_MAE"] = stats[f"{tag}_Clipped_MAE_Mean"]
                wandb_metrics[f"val/head_{head_id}/Raw_MAE"] = stats["Raw_MAE_Mean"]
                wandb_metrics[f"val/head_{head_id}/{primary_tag}/TargetAvg_wMAE"] = stats[f"{primary_tag}_TargetAvg_wMAE_Mean"]
                wandb.log(wandb_metrics, step=global_step)

        logger.info("=== Evaluation per head ===")
        for k, v in results.items():
            logger.info(
                f"{k}: "
                + " ".join(
                    [
                        f"{_format_horizon_tag(h)}/wMAE={v[f'{_format_horizon_tag(h)}_wMAE_Mean']:.4f}"
                        for h in metric_horizons
                    ]
                )
            )
            if "per_target" in v:
                logger.info(f"  Per-target {primary_tag} wMAE for {k}:")
                for target_name, target_metrics in v["per_target"].items():
                    logger.info(f"    {target_name}: wMAE={target_metrics['wMAE']:.4f}")

        best_head_name = min(results.items(), key=lambda x: x[1][f"{primary_tag}_wMAE_Mean"])[0]
        best_head_stats = results[best_head_name]

        logger.info("\n" + "=" * 70)
        logger.info(f"BEST HEAD: {best_head_name} ({primary_tag}/wMAE={best_head_stats[f'{primary_tag}_wMAE_Mean']:.4f})")
        logger.info("=" * 70)
        logger.info(
            + " ".join(
                [
                    f"{_format_horizon_tag(h)}/wMAE={best_head_stats[f'{_format_horizon_tag(h)}_wMAE_Mean']:.4f}"
                    for h in metric_horizons
                ]
            )
        )
        logger.info("=" * 70)

        if rank == 0:
            wandb_best_metrics = {
                "val/best_head/Raw_MAE": best_head_stats["Raw_MAE_Mean"],
                f"val/best_head/{primary_tag}/TargetAvg_wMAE": best_head_stats[f"{primary_tag}_TargetAvg_wMAE_Mean"],
            }
            for horizon in metric_horizons:
                tag = _format_horizon_tag(horizon)
                for metric_name in ["out_MAE", "in_MAE", "wMAE", "eMAE", "mMAE", "dMAE"]:
                    wandb_best_metrics[f"val/best_head/{tag}/{metric_name}"] = best_head_stats[f"{tag}_{metric_name}_Mean"]
            wandb.log(wandb_best_metrics, step=global_step)
            wandb.log({"val/best_head_id": int(best_head_name.split("_")[1])}, step=global_step)

        if head_to_dataset_map is not None:
            logger.info("\n=== Evaluation per dataset ===")
            dataset_results = {}
            for ds_idx in df["dataset_idx"].unique():
                ds_df = df[df["dataset_idx"] == ds_idx]
                assigned_heads = [h for h, d in enumerate(head_to_dataset_map) if d == ds_idx]
                logger.info(f"\nDataset {ds_idx} (Heads: {assigned_heads}):")
                for head_id in assigned_heads:
                    head_df = ds_df[ds_df["head"] == head_id]
                    if len(head_df) == 0:
                        continue

                    _, stats = evaluate_per_video_anticipation(
                        head_df,
                        target_names=loader_target_names,
                        metric_horizons=metric_horizons,
                    )
                    if rank == 0:
                        wandb_metrics = {}
                        for horizon in metric_horizons:
                            tag = _format_horizon_tag(horizon)
                            wandb_metrics[f"val/dataset_{ds_idx}/head_{head_id}/{tag}/wMAE"] = stats[f"{tag}_wMAE_Mean"]
                        wandb.log(wandb_metrics, step=global_step)
                    logger.info(
                        f"  Head {head_id}: "
                        + " ".join(
                            [
                                f"{_format_horizon_tag(h)}/wMAE={stats[f'{_format_horizon_tag(h)}_wMAE_Mean']:.4f}"
                                for h in metric_horizons
                            ]
                        )
                    )

                    dataset_results[f"dataset_{ds_idx}_head_{head_id}"] = stats
            metrics.update(dataset_results)

        if save_predictions and folder is not None:
            pred_file = os.path.join(folder, f"predictions_epoch_{epoch}.csv")
            df.to_csv(pred_file, index=False)
            logger.info(f"Saved predictions to {pred_file}")

    return metrics


# ----------------------
# checkpoint 加载
# ----------------------
def load_checkpoint(device, r_path, encoder, classifiers, opt, scaler, val_only=False):
    checkpoint = robust_checkpoint_loader(r_path, map_location="cpu")

    encoder.load_state_dict(checkpoint["encoder"])
    for c, state in zip(classifiers, checkpoint["classifiers"]):
        if hasattr(c, 'module'):
            c.module.load_state_dict(state)
        else:
            c.load_state_dict(state)

    if val_only:
        return encoder, classifiers, opt, scaler, 0

    epoch = checkpoint["epoch"]
    for o, state in zip(opt, checkpoint["opt"]):
        o.load_state_dict(state)
    for s, state in zip(scaler, checkpoint["scaler"]):
        if s is not None and state is not None:
            s.load_state_dict(state)
    return encoder, classifiers, opt, scaler, epoch


# ----------------------
# Quick run / debug utilities
# ----------------------
def create_quick_run_subset(csv_path, num_videos=1, output_dir=None):
    """
    Create a subset CSV file with only N videos for quick debugging runs.

    Args:
        csv_path: Path to original CSV file
        num_videos: Number of videos to include in subset
        output_dir: Directory to save subset CSV (default: same as original)

    Returns:
        Path to subset CSV file
    """
    import pandas as pd
    import os

    df = pd.read_csv(csv_path)

    # Get unique video IDs (case_id column)
    unique_videos = df['case_id'].unique()

    if len(unique_videos) < num_videos:
        logger.warning(
            f"Requested {num_videos} videos but only {len(unique_videos)} available. "
            f"Using all {len(unique_videos)} videos."
        )
        num_videos = len(unique_videos)

    # Select first N videos
    selected_videos = unique_videos[:num_videos]
    subset_df = df[df['case_id'].isin(selected_videos)]

    # Create output path
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)

    basename = os.path.basename(csv_path)
    name_without_ext = os.path.splitext(basename)[0]
    subset_path = os.path.join(f"{name_without_ext}_quick_{num_videos}vid.csv")

    # Save subset
    subset_df.to_csv(subset_path, index=False)
    logger.info(
        f"Created quick run subset: {subset_path} "
        f"({len(subset_df)} samples from {num_videos} video(s))"
    )

    return subset_path


# ----------------------
# dataloader
# ----------------------
DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type="VideoDataset",
    img_size=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=12,
    subset_file=None,
    normalization=None,
    datasets_weights=None,
    target_names=None,
):
    if normalization is None:
        normalization = DEFAULT_NORMALIZATION

    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=img_size,
        normalize=normalization,
    )

    data_loader, data_sampler = init_data(
        data=dataset_type,
        root_path=root_path,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        clip_len=frames_per_clip,
        frame_sample_rate=frame_step,
        duration=eval_duration,
        num_clips=num_segments,
        allow_clip_overlap=allow_segment_overlap,
        num_workers=num_workers,
        drop_last=False,
        subset_file=subset_file,
        datasets_weights=datasets_weights,
        target_names=target_names,
    )
    return data_loader, data_sampler


# ----------------------
# optimizer + scheduler
# ----------------------
def init_opt(classifiers, opt_kwargs, iterations_per_epoch, num_epochs, use_bfloat16=False):
    optimizers, schedulers, wd_schedulers, scalers = [], [], [], []
    for c, kwargs in zip(classifiers, opt_kwargs):
        param_groups = [{
            "params": c.parameters(),
            "mc_warmup_steps": int(kwargs.get("warmup") * iterations_per_epoch),
            "mc_start_lr": kwargs.get("start_lr"),
            "mc_ref_lr": kwargs.get("lr"),
            "mc_final_lr": kwargs.get("final_lr"),
            "mc_ref_wd": kwargs.get("weight_decay"),
            "mc_final_wd": kwargs.get("final_weight_decay"),
        }]
        optim = torch.optim.AdamW(param_groups)
        schedulers.append(WarmupCosineLRSchedule(optim, T_max=int(num_epochs * iterations_per_epoch)))
        wd_schedulers.append(CosineWDSchedule(optim, T_max=int(num_epochs * iterations_per_epoch)))
        optimizers.append(optim)
        scalers.append(torch.amp.GradScaler('cuda') if use_bfloat16 else None)
    return optimizers, scalers, schedulers, wd_schedulers


class WarmupCosineLRSchedule:
    def __init__(self, optimizer, T_max, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0

    def step(self):
        self._step += 1
        for group in self.optimizer.param_groups:
            ref_lr = group.get("mc_ref_lr")
            final_lr = group.get("mc_final_lr")
            start_lr = group.get("mc_start_lr")
            warmup_steps = group.get("mc_warmup_steps")
            T_max = self.T_max - warmup_steps
            if self._step < warmup_steps:
                progress = self._step / max(1, warmup_steps)
                new_lr = start_lr + progress * (ref_lr - start_lr)
            else:
                progress = (self._step - warmup_steps) / max(1, T_max)
                new_lr = max(
                    final_lr, final_lr + (ref_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
                )
            group["lr"] = new_lr


class CosineWDSchedule:
    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        for group in self.optimizer.param_groups:
            ref_wd = group.get("mc_ref_wd")
            final_wd = group.get("mc_final_wd")
            new_wd = final_wd + (ref_wd - final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
            if final_wd <= ref_wd:
                new_wd = max(final_wd, new_wd)
            else:
                new_wd = min(final_wd, new_wd)
            group["weight_decay"] = new_wd

